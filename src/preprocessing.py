"""
Image Preprocessing Module

Handles all image preprocessing tasks including resizing, normalization,
and augmentation for pneumonia detection model.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class ImagePreprocessor:
    """
    Image preprocessing class for medical image processing.
    
    Provides methods for resizing, normalizing, and augmenting
    chest X-ray images for model training and inference.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize pixel values
        """
        self.target_size = target_size
        self.normalize = normalize
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array or None if error
        """
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image array
            
        Returns:
            Resized image array
        """
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        return resized
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to [0, 1] range.
        
        Args:
            image: Input image array
            
        Returns:
            Normalized image array
        """
        if image.max() > 1:
            normalized = image.astype(np.float32) / 255.0
        else:
            normalized = image.astype(np.float32)
        return normalized
    
    def convert_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Convert grayscale image to RGB.
        
        Args:
            image: Grayscale image array
            
        Returns:
            RGB image array
        """
        if len(image.shape) == 2:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = image
        return rgb_image
    
    def preprocess(self, image_path: str) -> Optional[np.ndarray]:
        """
        Full preprocessing pipeline.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array or None if error
        """
        # Load image
        image = self.load_image(image_path)
        if image is None:
            return None
        
        # Resize
        image = self.resize_image(image)
        
        # Convert to RGB
        image = self.convert_to_rgb(image)
        
        # Normalize
        if self.normalize:
            image = self.normalize_image(image)
        
        return image


def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> Optional[np.ndarray]:
    """
    Standalone function to preprocess a single image.
    
    Args:
        image_path: Path to image file
        target_size: Target image dimensions
        normalize: Whether to normalize pixel values
        
    Returns:
        Preprocessed image array or None if error
        
    Example:
        >>> image = preprocess_image("chest_xray.jpg")
        >>> print(image.shape)
        (224, 224, 3)
    """
    preprocessor = ImagePreprocessor(target_size, normalize)
    return preprocessor.preprocess(image_path)


def augment_image(
    image: np.ndarray,
    rotation_range: int = 15,
    zoom_range: float = 0.2,
    horizontal_flip: bool = True
) -> np.ndarray:
    """
    Apply data augmentation to image.
    
    Args:
        image: Input image array
        rotation_range: Rotation angle range in degrees
        zoom_range: Zoom range factor
        horizontal_flip: Whether to apply horizontal flip
        
    Returns:
        Augmented image array
    """
    # Rotation
    if np.random.rand() < 0.5:
        angle = np.random.uniform(-rotation_range, rotation_range)
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    # Zoom
    if np.random.rand() < 0.5:
        zoom_factor = np.random.uniform(1 - zoom_range, 1 + zoom_range)
        height, width = image.shape[:2]
        new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
        image_resized = cv2.resize(image, (new_width, new_height))
        
        if new_height > height and new_width > width:
            y_start = (new_height - height) // 2
            x_start = (new_width - width) // 2
            image = image_resized[y_start:y_start + height, x_start:x_start + width]
        else:
            new_image = np.zeros_like(image)
            y_start = (height - new_height) // 2
            x_start = (width - new_width) // 2
            new_image[y_start:y_start + new_height, x_start:x_start + new_width] = image_resized
            image = new_image
    
    # Horizontal flip
    if horizontal_flip and np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
    
    return image


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    # Create preprocessor
    preprocessor = ImagePreprocessor()
    
    # Test with a sample image if available
    sample_images = list(Path("data").glob("**/*.jpg"))
    if sample_images:
        image = preprocessor.preprocess(str(sample_images[0]))
        if image is not None:
            print(f"Preprocessed image shape: {image.shape}")
            print(f"Pixel range: [{image.min():.2f}, {image.max():.2f}]")
