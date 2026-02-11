"""
CNN Model Architecture for Pneumonia Detection

This module contains the Convolutional Neural Network architecture
used for binary classification of pneumonia from chest X-ray images.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple


def create_cnn_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    dropout_rate: float = 0.5
) -> keras.Model:
    """
    Create a CNN model for pneumonia classification.
    
    Architecture:
    - Convolutional layers with filters: 64, 128, 256
    - Max pooling for dimensionality reduction
    - Dropout layers for regularization
    - Fully connected layers for classification
    - Binary cross-entropy loss for binary classification
    
    Args:
        input_shape: Input image dimensions (height, width, channels)
        num_classes: Number of output classes (2 for binary classification)
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model ready for training
        
    Example:
        >>> model = create_cnn_model()
        >>> model.summary()
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            input_shape=input_shape,
            name='conv2d_1'
        ),
        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            name='conv2d_2'
        ),
        layers.MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1'),
        layers.Dropout(dropout_rate, name='dropout_1'),
        
        # Second Convolutional Block
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            name='conv2d_3'
        ),
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            name='conv2d_4'
        ),
        layers.MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_2'),
        layers.Dropout(dropout_rate, name='dropout_2'),
        
        # Third Convolutional Block
        layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            name='conv2d_5'
        ),
        layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            name='conv2d_6'
        ),
        layers.MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_3'),
        layers.Dropout(dropout_rate, name='dropout_3'),
        
        # Fully Connected Layers
        layers.Flatten(name='flatten'),
        layers.Dense(512, activation='relu', name='dense_1'),
        layers.Dropout(dropout_rate, name='dropout_4'),
        layers.Dense(128, activation='relu', name='dense_2'),
        layers.Dropout(dropout_rate, name='dropout_5'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    metrics: list = None
) -> keras.Model:
    """
    Compile the CNN model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
        metrics: List of metrics to track during training
        
    Returns:
        Compiled model
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall']
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy()
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model


def create_binary_cnn_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.5
) -> keras.Model:
    """
    Create a simplified CNN model for binary classification.
    
    Optimized for binary classification (Normal vs Pneumonia).
    Uses binary_crossentropy loss instead of categorical_crossentropy.
    
    Args:
        input_shape: Input image dimensions
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model


if __name__ == "__main__":
    # Create and display model
    model = create_cnn_model()
    print(model.summary())
    
    # Print model architecture
    print("\nModel created successfully!")
    print(f"Parameters: {model.count_params():,}")
