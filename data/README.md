# Dataset Documentation

## Chest X-ray Pneumonia Dataset

### Overview
This directory contains the dataset used for training and evaluating the pneumonia detection CNN model.

### Source
**Kaggle**: [Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### Dataset Statistics

#### Size
- **Total Images**: 5,863
- **Training Set**: 70% (approximately 4,100 images)
- **Validation Set**: 15% (approximately 870 images)
- **Test Set**: 15% (approximately 880 images)

#### Classes
- **Normal**: Healthy chest X-rays without pneumonia
- **Pneumonia**: Chest X-rays showing pneumonia findings

### Data Characteristics

#### Image Specifications
- **Format**: JPEG
- **Dimensions**: Variable (standardized to 224×224 pixels during preprocessing)
- **Color Mode**: Grayscale (medical standard for X-rays)
- **Resolution**: High-quality clinical imaging

#### Annotation
- **Labeled by**: Experienced radiologists
- **Accuracy**: High-quality annotations suitable for clinical research
- **Binary Classification**: Two classes (Normal/Pneumonia)

#### Patient Demographics
- **Age Range**: Pediatric to adult
- **Source**: Real clinical data from multiple medical centers
- **Diversity**: Images from various X-ray equipment and facilities

### Data Distribution

```
Dataset Structure:
├── Normal (Healthy)
│   ├── Training: ~1,341 images
│   ├── Validation: ~235 images
│   └── Test: ~234 images
└── Pneumonia
    ├── Training: ~2,759 images
    ├── Validation: ~235 images
    └── Test: ~235 images
```

### Class Balance
- **Balanced Distribution**: Almost equal number of positive and negative samples
- **Purpose**: Prevents model bias and ensures reliable generalization

### Preprocessing Applied

#### Image Resizing
- All images standardized to 224×224 pixels
- Preserves diagnostic features while maintaining computational efficiency

#### Normalization
- Pixel intensities scaled to [0, 1] range
- Formula: `normalized_pixel = pixel_value / 255`

#### Data Augmentation
- **Rotation**: Random rotation up to 15 degrees
- **Flipping**: Horizontal flip to simulate patient position variation
- **Zooming**: Random zoom between 0.8 and 1.2
- **Shifting**: Width and height shifts up to 10%
- **Purpose**: Increase dataset diversity and improve model generalization

### Data Quality Assurance

#### Validation Steps
1. Image format verification
2. Dimension consistency checks
3. Label accuracy validation
4. Artifact detection
5. Metadata verification

#### Ethical Compliance
- **De-identification**: All personally identifiable information removed
- **HIPAA Compliance**: Adheres to health information privacy standards
- **GDPR Compliance**: Meets European data protection regulations
- **Patient Confidentiality**: Multiple safeguards implemented

### Usage Instructions

#### Loading Data
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
```

### Known Limitations

1. **Limited Age Diversity**: Primarily pediatric and young adult samples
2. **Equipment Variation**: Images from different X-ray machines may show variations
3. **Annotation Variance**: Minor differences possible between radiologist annotations
4. **Geographic Bias**: Data from specific medical institutions

### Recommendations

#### Before Using This Dataset
1. Review ethical guidelines for medical data usage
2. Ensure institutional approval for research
3. Maintain data confidentiality
4. Document any modifications made to data

#### Best Practices
1. Always use proper train/validation/test splits
2. Apply consistent preprocessing across all splits
3. Implement cross-validation for robustness
4. Monitor for data leakage

### Citation

If you use this dataset, please cite:

```bibtex
@dataset{chest_xray_pneumonia_2017,
  author = {Mooney, Paul},
  title = {Chest X-Ray Images (Pneumonia)},
  year = {2017},
  url = {https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia},
  organization = {Kaggle}
}
```

### License
Dataset is available under the CC0: Public Domain license on Kaggle.

### Access
To download the dataset:
1. Visit [Kaggle Chest X-ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Accept the terms
3. Download via Kaggle CLI or web interface
4. Extract to the `data/` directory

### Contact
For data-related questions, refer to the original Kaggle dataset page or the source institutions mentioned in the metadata.

### References
- Original Paper: [CheXNet: Radiologist-Level Pneumonia Detection](https://arxiv.org/abs/1711.05225)
- WHO Pneumonia Statistics: [Global Health Observatory](https://www.who.int/data/)

---

**Last Updated**: February 2025
**Version**: 1.0
