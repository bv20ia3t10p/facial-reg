"""
Facial recognition model architecture with privacy features.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization, 
    Activation, Flatten, Dense, Dropout, GlobalAveragePooling2D
)
from tensorflow.keras.regularizers import l2

from config import EMBEDDING_SIZE

def build_face_recognition_model(num_classes, training=True):
    """
    Build a facial recognition model based on a modified ResNet architecture.
    
    Args:
        num_classes: Number of identity classes in the dataset
        training: Whether the model is for training (True) or inference (False)
        
    Returns:
        Keras model for face recognition
    """
    weight_decay = 5e-4
    input_shape = (112, 112, 3)
    
    inputs = Input(shape=input_shape)
    
    # First block
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Second block
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Third block
    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Fourth block
    x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Feature extraction part
    x = GlobalAveragePooling2D()(x)
    
    # Face embedding layer
    embeddings = Dense(EMBEDDING_SIZE, kernel_regularizer=l2(weight_decay), name='embeddings')(x)
    
    if training:
        # For training we need the classification head
        x = Dropout(0.5)(embeddings)
        outputs = Dense(num_classes, activation='softmax', name='classifier')(x)
        model = Model(inputs=inputs, outputs=outputs)
    else:
        # For inference we only need the embeddings
        model = Model(inputs=inputs, outputs=embeddings)
    
    return model

def build_dp_model(num_classes):
    """
    Build a simpler model architecture that works better with differential privacy.
    DP training can struggle with very deep networks.
    
    Args:
        num_classes: Number of identity classes in the dataset
        
    Returns:
        Keras model for differentially private training
    """
    weight_decay = 5e-4
    input_shape = (112, 112, 3)
    
    inputs = Input(shape=input_shape)
    
    # Simpler architecture for DP training
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Feature extraction
    x = GlobalAveragePooling2D()(x)
    
    # Face embedding layer
    embeddings = Dense(EMBEDDING_SIZE, kernel_regularizer=l2(weight_decay), name='embeddings')(x)
    
    # Classification head
    outputs = Dense(num_classes, activation='softmax', name='classifier')(embeddings)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def get_embedding_model(trained_model):
    """
    Extract the embedding part of a trained model.
    
    Args:
        trained_model: Trained classification model
        
    Returns:
        Model that outputs embeddings only
    """
    embedding_layer = trained_model.get_layer('embeddings').output
    embedding_model = Model(inputs=trained_model.input, outputs=embedding_layer)
    return embedding_model 