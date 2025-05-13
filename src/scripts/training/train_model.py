#!/usr/bin/env python
"""
Unified script for training face recognition models.
Can be used to train both server and client models with similar architecture.
"""

import os
import sys
import warnings
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import Activation, AveragePooling2D, Conv2D, Multiply, Reshape, Lambda, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import logging
import argparse
import time
import matplotlib.pyplot as plt
import shutil
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Train a face recognition model')
    parser.add_argument('--mode', type=str, required=True, choices=['server', 'client'],
                       help='Training mode: server or client')
    parser.add_argument('--client-id', type=str, default=None,
                       help='Client ID (required for client mode)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing the training data')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save the trained model')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Image size for training (square)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with data validation')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'client' and args.client_id is None:
        parser.error("--client-id is required for client mode")
    
    # Set default data directory if not provided
    if args.data_dir is None:
        if args.mode == 'server':
            args.data_dir = 'data/partitioned/server'
        else:
            args.data_dir = f'data/partitioned/{args.client_id}'
    
    return args

# Squeeze and Excitation block
def squeeze_excitation_block(input_tensor, ratio=16):
    """Implements Squeeze and Excitation block"""
    filters = input_tensor.shape[-1]
    se_shape = (1, 1, filters)
    
    # Squeeze operation (global average pooling)
    squeeze = GlobalAveragePooling2D()(input_tensor)
    squeeze = Reshape(se_shape)(squeeze)
    
    # Excitation operation (2 FC layers)
    excitation = Dense(filters // ratio, activation='relu', 
                      kernel_initializer='he_normal', use_bias=False)(squeeze)
    excitation = Dense(filters, activation='sigmoid', 
                      kernel_initializer='he_normal', use_bias=False)(excitation)
    
    # Scale - multiply input feature maps with attention weights
    scale = Multiply()([input_tensor, excitation])
    
    return scale

# Efficient Channel Attention block
def eca_block(input_tensor, b=1, gamma=2):
    """Implements Efficient Channel Attention (ECA) block"""
    channels = input_tensor.shape[-1]
    # Calculate adaptive kernel size
    t = int(abs((np.log2(channels)/gamma) + b/gamma))
    k = t if t % 2 else t + 1
    
    # Global average pooling
    y = GlobalAveragePooling2D()(input_tensor)
    
    # Reshape for 1D convolution (using Lambda to avoid shape issues)
    y = Reshape((channels, 1))(y)
    
    # Use Conv1D instead of Conv2D with proper parameters
    y = Conv1D(filters=1, kernel_size=k, padding='same', use_bias=False)(y)
    
    # Apply sigmoid activation
    y = Activation('sigmoid')(y)
    
    # Reshape to broadcast correctly
    y = Reshape((1, 1, channels))(y)
    
    # Scale the input tensor with attention weights
    output = Multiply()([input_tensor, y])
    
    return output

def resnet50_se_eca(input_shape, num_classes):
    """Create a ResNet50 model for facial recognition"""
    # Create a base model with ResNet50 architecture
    input_tensor = Input(shape=input_shape)
    
    # Load ResNet50 model without top layers
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor
    )
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Fine-tune: Make all layers trainable
    for layer in model.layers:
        layer.trainable = True
    
    return model

def validate_dataset(data_dir):
    """Validate dataset integrity and save sample images for debugging"""
    print(f"Validating dataset in {data_dir}...")
    
    # Check directory structure
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist!")
    
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not class_dirs:
        raise ValueError(f"No class directories found in {data_dir}!")
    
    print(f"Found {len(class_dirs)} class directories")
    
    # Check image counts in each class
    total_images = 0
    empty_classes = []
    small_classes = []
    
    for class_name in class_dirs:
        class_path = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        img_count = len(images)
        
        if img_count == 0:
            empty_classes.append(class_name)
        elif img_count < 5:
            small_classes.append((class_name, img_count))
        
        total_images += img_count
    
    print(f"Total images found: {total_images}")
    
    if empty_classes:
        print(f"WARNING: Found {len(empty_classes)} empty class directories: {empty_classes[:5]}...")
    
    if small_classes:
        print(f"WARNING: Found {len(small_classes)} classes with fewer than 5 images: {small_classes[:5]}...")
    
    # Remove empty class directories
    if empty_classes:
        print(f"Removing {len(empty_classes)} empty class directories...")
        for class_name in empty_classes:
            shutil.rmtree(os.path.join(data_dir, class_name))
    
    # Sample and save some images for visual inspection
    debug_dir = os.path.join("debug", f"{os.path.basename(data_dir)}_samples")
    os.makedirs(debug_dir, exist_ok=True)
    
    sample_classes = min(5, len(class_dirs))
    for class_name in class_dirs[:sample_classes]:
        class_path = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not images:
            continue
        
        sample_count = min(2, len(images))
        for i, img_file in enumerate(images[:sample_count]):
            src_path = os.path.join(class_path, img_file)
            dst_path = os.path.join(debug_dir, f"{class_name}_{i}.jpg")
            shutil.copy(src_path, dst_path)
    
    print(f"Saved sample images to {debug_dir} for visual inspection")
    return len(class_dirs), total_images

def plot_sample_images(generator, class_names, n_samples=5):
    """Plot sample images from generator to verify data is correct"""
    
    debug_dir = os.path.join("debug", "processed_samples")
    os.makedirs(debug_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 5*n_samples))
    batch = next(generator)
    images, labels = batch[0], batch[1]
    
    for i in range(min(n_samples, len(images))):
        plt.subplot(n_samples, 3, i*3 + 1)
        plt.imshow((images[i] * 255).astype(np.uint8))
        class_idx = np.argmax(labels[i])
        plt.title(f"Class: {class_names[class_idx]}")
        
        # Save the processed image for inspection
        plt.imsave(os.path.join(debug_dir, f"sample_{i}.jpg"), (images[i] * 255).astype(np.uint8))
    
    plt.savefig(os.path.join(debug_dir, "batch_samples.jpg"))
    print(f"Saved processed image samples to {debug_dir}")

def main():
    args = parse_args()
    
    # Create debug directory
    os.makedirs("debug", exist_ok=True)
    
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Suppress all warnings and info messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR
    warnings.filterwarnings('ignore')
    tf.get_logger().setLevel(logging.ERROR)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    # Validate dataset if in debug mode
    if args.debug:
        num_classes, total_images = validate_dataset(args.data_dir)
        print(f"Dataset validation complete. Found {num_classes} classes with {total_images} total images.")

    # Configure GPU memory growth to avoid memory errors
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"Found {len(physical_devices)} GPU(s), enabled memory growth")
        except:
            print("Failed to configure GPU memory growth")

    # Setup data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=lambda x: x / 255.0
    )
    
    val_datagen = ImageDataGenerator(
        preprocessing_function=lambda x: x / 255.0
    )

    # Load data
    print(f"Loading data from {args.data_dir}...")
    train_generator = train_datagen.flow_from_directory(
        args.data_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        args.data_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    num_classes = len(train_generator.class_indices)
    print(f"Found {num_classes} classes")
    
    # Plot sample images if in debug mode
    if args.debug:
        class_names = [k for k, v in sorted(train_generator.class_indices.items(), key=lambda item: item[1])]
        plot_sample_images(train_generator, class_names)

    # Build model
    print("Building model...")
    model = resnet50_se_eca((args.img_size, args.img_size, 3), num_classes)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    checkpoint_path = os.path.join(
        args.model_dir, 
        f"{args.mode}_{os.path.basename(args.data_dir)}_best.h5"
    )
    
    model_checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    callbacks = [early_stopping, reduce_lr, model_checkpoint]

    # Train model
    print(f"Training {args.mode} model...")
    start_time = time.time()
    
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Save final model
    final_model_path = os.path.join(
        args.model_dir, 
        f"{args.mode}_{os.path.basename(args.data_dir)}_final.h5"
    )
    
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save class indices
    class_indices_path = os.path.join(
        args.model_dir, 
        f"{args.mode}_{os.path.basename(args.data_dir)}_classes.json"
    )
    
    with open(class_indices_path, 'w') as f:
        json.dump(train_generator.class_indices, f, indent=4)
    
    print(f"Class indices saved to {class_indices_path}")
    
    # Plot and save training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    
    history_path = os.path.join(
        args.model_dir, 
        f"{args.mode}_{os.path.basename(args.data_dir)}_history.png"
    )
    
    plt.savefig(history_path)
    print(f"Training history plot saved to {history_path}")
    
    print(f"{args.mode.capitalize()} model training complete!")

if __name__ == "__main__":
    main() 