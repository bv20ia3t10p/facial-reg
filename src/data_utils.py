"""
Utilities for loading and preprocessing the CASIA-WebFace dataset.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATASET_PATH, IMG_SIZE, BATCH_SIZE, VALIDATION_SPLIT

def preprocess_image(img_path):
    """
    Preprocess a single image for facial recognition.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Preprocessed image as numpy array
    """
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to target dimensions
    img = cv2.resize(img, IMG_SIZE)
    
    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img

def create_dataset_index(dataset_path=DATASET_PATH):
    """
    Create an index of images and labels from the CASIA-WebFace dataset.
    
    Args:
        dataset_path: Path to the CASIA-WebFace dataset
        
    Returns:
        DataFrame with columns: image_path, person_id, label_idx
    """
    image_paths = []
    person_ids = []
    
    print("Indexing CASIA-WebFace dataset...")
    for person_id in tqdm(os.listdir(dataset_path)):
        person_dir = os.path.join(dataset_path, person_id)
        if not os.path.isdir(person_dir):
            continue
            
        for img_name in os.listdir(person_dir):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(person_dir, img_name)
                image_paths.append(img_path)
                person_ids.append(person_id)
    
    # Create DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'person_id': person_ids
    })
    
    # Create numeric labels
    unique_ids = df['person_id'].unique()
    id_to_label = {id_: idx for idx, id_ in enumerate(unique_ids)}
    df['label_idx'] = df['person_id'].map(id_to_label)
    
    print(f"Dataset indexed: {len(df)} images, {len(unique_ids)} unique individuals")
    return df

def load_dataset(index_df=None):
    """
    Load and prepare the CASIA-WebFace dataset for training.
    
    Args:
        index_df: DataFrame with dataset index, if None, it will be created
        
    Returns:
        Training dataset, validation dataset, number of classes
    """
    if index_df is None:
        index_df = create_dataset_index()
    
    # Split into training and validation
    train_df, val_df = train_test_split(
        index_df, 
        test_size=VALIDATION_SPLIT,
        stratify=index_df['label_idx'],
        random_state=42
    )
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=lambda x: x
    )
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator(
        preprocessing_function=lambda x: x
    )
    
    # Create data generators
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='image_path',
        y_col='label_idx',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        color_mode='rgb'
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        x_col='image_path',
        y_col='label_idx',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        color_mode='rgb',
        shuffle=False
    )
    
    num_classes = len(index_df['label_idx'].unique())
    
    return train_generator, val_generator, num_classes

def create_tf_dataset(index_df=None):
    """
    Create a TensorFlow dataset from the CASIA-WebFace dataset.
    This is more suitable for using with differential privacy.
    
    Args:
        index_df: DataFrame with dataset index, if None, it will be created
        
    Returns:
        TensorFlow training dataset, validation dataset, number of classes
    """
    if index_df is None:
        index_df = create_dataset_index()
    
    # Split into training and validation
    train_df, val_df = train_test_split(
        index_df, 
        test_size=VALIDATION_SPLIT,
        stratify=index_df['label_idx'],
        random_state=42
    )
    
    num_classes = len(index_df['label_idx'].unique())
    
    def _parse_function(img_path, label):
        img_str = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img_str, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    
    # Create training dataset
    train_ds = tf.data.Dataset.from_tensor_slices((
        train_df['image_path'].values, 
        train_df['label_idx'].values
    ))
    train_ds = train_ds.map(_parse_function, 
                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=10000)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    # Create validation dataset
    val_ds = tf.data.Dataset.from_tensor_slices((
        val_df['image_path'].values, 
        val_df['label_idx'].values
    ))
    val_ds = val_ds.map(_parse_function, 
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return train_ds, val_ds, num_classes 