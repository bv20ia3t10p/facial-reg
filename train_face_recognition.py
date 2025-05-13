import os
import time
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from model import create_model
from utils import parse_args

def main():
    args = parse_args()
    
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Configure GPU memory growth to avoid memory errors
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"Found {len(physical_devices)} GPU(s), enabled memory growth")
        except:
            print("Failed to configure GPU memory growth")

    # Create a smaller dataset for faster training
    print(f"Preparing data from {args.data_dir}...")
    
    # Create a temporary directory for the sample dataset
    sample_data_dir = os.path.join(args.model_dir, "sample_data")
    if os.path.exists(sample_data_dir):
        import shutil
        shutil.rmtree(sample_data_dir)
    os.makedirs(sample_data_dir, exist_ok=True)
    
    # Get class directories
    class_dirs = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    
    # Limit to 10 classes maximum for faster training
    max_classes = min(10, len(class_dirs))
    max_images_per_class = 20  # Limit images per class
    
    print(f"Creating sample dataset with {max_classes} classes, {max_images_per_class} images per class")
    
    # Copy a subset of the data
    for i, class_dir in enumerate(class_dirs[:max_classes]):
        src_dir = os.path.join(args.data_dir, class_dir)
        dst_dir = os.path.join(sample_data_dir, class_dir)
        os.makedirs(dst_dir, exist_ok=True)
        
        # Get image files
        image_files = [f for f in os.listdir(src_dir) 
                       if os.path.isfile(os.path.join(src_dir, f)) and 
                       f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit number of images
        for img_file in image_files[:max_images_per_class]:
            src_path = os.path.join(src_dir, img_file)
            dst_path = os.path.join(dst_dir, img_file)
            import shutil
            shutil.copy2(src_path, dst_path)
    
    # Use the sample data directory instead of original
    data_dir = sample_data_dir
    
    # Setup data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,  # Use 20% for validation
        preprocessing_function=lambda x: x / 255.0
    )

    # Load data
    print(f"Loading data from sample directory...")
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    num_classes = len(train_generator.class_indices)
    print(f"Found {num_classes} classes")
    
    # Build model
    print("Building model...")
    model = create_model((args.img_size, args.img_size, 3), num_classes)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
    
    checkpoint_path = os.path.join(
        args.model_dir, 
        f"face_recognition_best.h5"
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
    print(f"Training model...")
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

if __name__ == "__main__":
    main()

