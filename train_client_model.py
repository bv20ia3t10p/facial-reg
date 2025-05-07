import os
import sys
import warnings
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import logging
import argparse
import time
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import Activation, AveragePooling2D, Conv2D
import datetime

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
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Image size for training (square)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate client ID
    if args.client_id not in ['client1', 'client2']:
        print(f"Error: client-id must be 'client1' or 'client2', got '{args.client_id}'")
        sys.exit(1)
    
    # Set default data directory if not provided
    if args.data_dir is None:
        args.data_dir = f'data/partitioned/{args.client_id}'
    
    # Suppress all warnings and info messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore')
    tf.get_logger().setLevel(logging.ERROR)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

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

    # Configure data generators with stronger, more diverse augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest',
        validation_split=0.2,
        # Add these to further improve robustness
        channel_shift_range=0.2,
        preprocessing_function=lambda x: tf.image.random_contrast(x, lower=0.8, upper=1.2)
    )
    
    # Use less augmentation for validation to get more realistic metrics
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
        print(f"Classes in {args.client_id} partition: {sorted(list(train_generator.class_indices.keys()))[:5]}...")
        print(f"Total samples in training set: {train_generator.samples}")
        print(f"Total samples in validation set: {validation_generator.samples}")
        
        # Build improved model with EfficientNetB0 (better than MobileNetV2)
        print('Building enhanced EfficientNetB0 model for facial recognition...')
        
        # Use EfficientNetB0 with pretrained ImageNet weights as base
        base_model = EfficientNetB0(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )

        # Two-phase training approach
        # Phase 1: Train only the top layers (freeze base model)
        for layer in base_model.layers:
            layer.trainable = False
            
        # Build improved classifier head with batch normalization
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dense(1024, kernel_initializer=HeNormal(), kernel_regularizer=l2(0.0001)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.3),
            Dense(512, kernel_initializer=HeNormal(), kernel_regularizer=l2(0.0001)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax', kernel_initializer=HeNormal())
        ])

        # Compile the model with a higher initial learning rate for the classifier head
        optimizer = Adam(learning_rate=INIT_LR)
        
        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Show model architecture summary
        model.summary()
        print(f"Phase 1: Training only the classifier head, base model frozen for {args.client_id}")

        # Set up callbacks for phase 1
        callbacks_phase1 = [
            ModelCheckpoint(os.path.join(args.model_dir, f'{args.client_id}_model_phase1.h5'), 
                          monitor='val_accuracy', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=0.00001, verbose=1),
            TensorBoard(log_dir=os.path.join(log_dir, 'phase1'))
        ]

        # Train Phase 1
        print(f'Training Phase 1: Top layers only for {args.client_id}...')
        history_phase1 = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // BATCH_SIZE,
            epochs=10,  # Shorter phase 1
            callbacks=callbacks_phase1,
            verbose=1
        )
        
        # Phase 2: Fine-tune the model by unfreezing more layers
        # Unfreeze the top layers of the base model (fine-tuning)
        for layer in base_model.layers[-30:]:  # Unfreeze the top 30 layers
            layer.trainable = True
            
        print(f'Phase 2: Fine-tuning {args.client_id} model with {sum(1 for layer in model.layers if layer.trainable)} trainable layers')
            
        # Compile with a lower learning rate for fine-tuning
        optimizer_phase2 = Adam(learning_rate=INIT_LR/10)  # Lower learning rate for fine-tuning
        model.compile(
            optimizer=optimizer_phase2,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Set up callbacks for phase 2
        callbacks_phase2 = [
            ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=8, verbose=1, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.000001, verbose=1),
            TensorBoard(log_dir=os.path.join(log_dir, 'phase2'))
        ]

        # Train Phase 2
        print(f'Training Phase 2: Fine-tuning {args.client_id} model...')
        try:
            history_phase2 = model.fit(
                train_generator,
                steps_per_epoch=train_generator.samples // BATCH_SIZE,
                validation_data=validation_generator,
                validation_steps=validation_generator.samples // BATCH_SIZE,
                epochs=EPOCHS,
                callbacks=callbacks_phase2,
                verbose=1,
                initial_epoch=len(history_phase1.history['accuracy'])  # Start from where phase 1 ended
            )
            history = history_phase2  # Use phase 2 for final reporting
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