import os
import sys
import warnings
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import Activation, AveragePooling2D, Conv2D, Multiply, Reshape, Lambda
from tensorflow.keras.layers import Layer, GlobalAveragePooling1D, Dense, Reshape, Permute, Multiply
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import logging
import argparse
import time
import datetime
import matplotlib.pyplot as plt
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Train a client model for facial recognition')
    parser.add_argument('--client-id', type=str, required=True,
                        help='Client ID (client1 or client2)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing the client training data (default: data/partitioned/{client_id})')
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
    return parser.parse_args()

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
    
    # 1D convolution for cross-channel interaction with adaptive kernel size
    y = Reshape((channels, 1))(y)
    y = Conv2D(1, kernel_size=(k, 1), padding='same', 
               use_bias=False, activation='sigmoid')(y)
    y = Reshape((1, 1, channels))(y)
    
    # Scale the input tensor with attention weights
    output = Multiply()([input_tensor, y])
    
    return output

def resnet50_se_eca(input_shape, num_classes):
    """Create a ResNet50 model with SE and ECA blocks"""
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Make all layers trainable for fine-tuning
    for layer in base_model.layers:
        layer.trainable = True
    
    # Get the output of each ResNet block to add attention
    x = base_model.input
    outputs = []
    
    # Find the residual block outputs to apply attention to
    block_names = [layer.name for layer in base_model.layers 
                  if 'add' in layer.name and 'res' in layer.name]
    
    # Iterate through all blocks and apply attention
    for i, layer in enumerate(base_model.layers):
        x = layer(x)
        
        # Apply SE and ECA attention to residual blocks
        if layer.name in block_names:
            # Apply SE block
            se_out = squeeze_excitation_block(x)
            # Apply ECA block
            eca_out = eca_block(se_out)
            # Add skip connection (residual with attention)
            x = eca_out
            
        # Store the last layer's output
        if i == len(base_model.layers) - 1:
            outputs = x
    
    # Build classifier head
    x = GlobalAveragePooling2D()(outputs)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=output)
    
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
    
    # Validate client ID
    if args.client_id not in ['client1', 'client2']:
        print(f"Error: client-id must be 'client1' or 'client2', got '{args.client_id}'")
        sys.exit(1)
    
    # Set default data directory if not provided
    if args.data_dir is None:
        args.data_dir = f'data/partitioned/{args.client_id}'
    
    # Create debug directory
    os.makedirs("debug", exist_ok=True)
    
    # Suppress all warnings and info messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore')
    tf.get_logger().setLevel(logging.ERROR)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    # Validate dataset if in debug mode
    if args.debug:
        num_classes, total_images = validate_dataset(args.data_dir)
        print(f"Dataset validation complete. Found {num_classes} classes with {total_images} total images.")

    # Enable mixed precision training for faster training
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print('Mixed precision training enabled')
    except:
        print('Mixed precision training not supported on this platform')

    # Disable GPU memory preallocation to avoid warnings
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print('Memory growth enabled on GPU devices')
        except Exception as e:
            print(f'Could not set memory growth: {e}')

    # Create models directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    log_dir = os.path.join(args.model_dir, "logs", f"{args.client_id}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)

    # Define constants
    IMG_SIZE = args.img_size
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    MODEL_PATH = os.path.join(args.model_dir, f'{args.client_id}_model.h5')
    INIT_LR = args.lr

    # Configure data generators with essential, simpler augmentation to start
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    # Use simple rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    try:
        # Load dataset
        print(f'Loading {args.client_id} dataset from {args.data_dir}...')
        train_generator = train_datagen.flow_from_directory(
            args.data_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )

        validation_generator = val_datagen.flow_from_directory(
            args.data_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        num_classes = len(train_generator.class_indices)
        print(f'Training {args.client_id} model with {num_classes} classes')
        
        # Verify if we have approximately the correct number of classes
        expected_classes = 200  # Equal 1/3 split with 200 classes each
        if num_classes > expected_classes * 2:  # Much more than expected
            print(f'WARNING: Found {num_classes} classes, but expected around {expected_classes}.')
            print('This may indicate the partitioning step did not work correctly or you might be using the wrong directory.')
            user_input = input('Continue anyway? (y/n): ')
            if user_input.lower() != 'y':
                print('Exiting as requested.')
                sys.exit(1)
                
        # Check which classes are actually present in the directory
        class_indices = train_generator.class_indices
        class_names = {v: k for k, v in class_indices.items()}
        print(f"Classes in {args.client_id} partition: {sorted(list(class_indices.keys()))[:5]}...")
        print(f"Total samples in training set: {train_generator.samples}")
        print(f"Total samples in validation set: {validation_generator.samples}")
        print(f"Steps per epoch: {train_generator.samples // BATCH_SIZE}")
        
        # In debug mode, plot some sample images to verify data loading
        if args.debug:
            plot_sample_images(train_generator, class_names)
        
        # Build the enhanced ResNet50 with SE and ECA blocks
        print(f'Building ResNet50 with SE and ECA blocks for {args.client_id} model...')
        
        # Create model with SE and ECA attention blocks
        model = resnet50_se_eca(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            num_classes=num_classes
        )

        # Compile with higher initial learning rate to get faster progress
        optimizer = Adam(learning_rate=INIT_LR * 5)  # Higher LR for attention model
        
        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Show model architecture summary
        model.summary()
        
        # Set up callbacks with longer patience to allow learning to progress
        callbacks = [
            ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=0.000001, verbose=1),
            TensorBoard(log_dir=log_dir)
        ]

        # Train the model
        print(f'Training {args.client_id} model with attention mechanisms...')
        try:
            # Train with the enhanced model
            history = model.fit(
                train_generator,
                steps_per_epoch=train_generator.samples // BATCH_SIZE if train_generator.samples > BATCH_SIZE else 1,
                validation_data=validation_generator,
                validation_steps=validation_generator.samples // BATCH_SIZE if validation_generator.samples > BATCH_SIZE else 1,
                epochs=EPOCHS,
                callbacks=callbacks,
                verbose=1
            )
            
            # Plot training history
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
            plt.savefig(os.path.join(log_dir, 'training_history.png'))
            
        except KeyboardInterrupt:
            print('\nTraining interrupted by user. Saving current model state...')

        # Save the final model
        model.save(MODEL_PATH)
        print(f'{args.client_id} model saved to {MODEL_PATH}')

        # Save class indices for inference
        import json
        with open(os.path.join(args.model_dir, f'{args.client_id}_class_indices.json'), 'w') as f:
            json.dump(train_generator.class_indices, f)

        # Print final metrics if training completed
        if 'history' in locals() and len(history.history.get('accuracy', [])) > 0:
            final_accuracy = history.history['accuracy'][-1]
            final_val_accuracy = history.history['val_accuracy'][-1]
            best_val_accuracy = max(history.history['val_accuracy'])
            print(f'Final {args.client_id} model training accuracy: {final_accuracy:.4f}')
            print(f'Final {args.client_id} model validation accuracy: {final_val_accuracy:.4f}')
            print(f'Best validation accuracy achieved: {best_val_accuracy:.4f}')
        else:
            print('Training was interrupted, final metrics not available')

    except Exception as e:
        print(f'Error training {args.client_id} model: {str(e)}')
        import traceback
        traceback.print_exc()
        print(f'Skipping {args.client_id} model training')

if __name__ == '__main__':
    main() 