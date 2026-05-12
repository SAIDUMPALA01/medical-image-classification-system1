
Copy

"""
data_loader.py
--------------
Data loading and augmentation utilities for the Pneumonia Detection CNN.
Dataset: Chest X-Ray Images (Pneumonia) - Kaggle
"""
 
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
 
 
# ── Constants ────────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
 
 
def get_data_generators(data_dir: str, batch_size: int = BATCH_SIZE):
    """
    Build train, validation, and test data generators with augmentation.
 
    Args:
        data_dir (str): Root directory containing 'train', 'val', 'test' subdirs.
        batch_size (int): Number of images per batch.
 
    Returns:
        tuple: (train_gen, val_gen, test_gen)
    """
    # Training — augmentation applied
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
 
    # Validation & test — only rescale
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)
 
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
    )
 
    val_generator = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )
 
    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )
 
    return train_generator, val_generator, test_generator
 
 
def load_single_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a single image for inference.
 
    Args:
        image_path (str): Path to the X-ray image file.
 
    Returns:
        np.ndarray: Preprocessed image array of shape (1, 224, 224, 3).
    """
    from tensorflow.keras.preprocessing import image as keras_image
 
    img = keras_image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = keras_image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)
