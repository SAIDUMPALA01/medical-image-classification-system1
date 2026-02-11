"""
Pneumonia Detection CNN Package

A deep learning-based solution for automated pneumonia detection from chest X-ray images.
"""

__version__ = "1.0.0"
__author__ = "Sai Ram"
__email__ = "your.email@example.com"

from .model import create_cnn_model
from .preprocessing import preprocess_image, ImagePreprocessor
from .data_loader import DataLoader

__all__ = [
    "create_cnn_model",
    "preprocess_image",
    "ImagePreprocessor",
    "DataLoader",
]
