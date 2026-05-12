# Medical Image Classification System
![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue) ![License: MIT](https://img.shields.io/badge/License-MIT-green) ![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange)

A comprehensive deep learning-based medical imaging solution for automated pneumonia detection from chest X-ray images using Convolutional Neural Networks (CNNs). This is the final Master's project from Birmingham City University.

---

## 📋 Overview

This project implements an advanced image processing system integrated with machine learning algorithms to improve the diagnostic process for pneumonia detection. The system automates the analysis of chest X-ray images, providing radiologists with accurate and timely insights to enhance clinical decision-making.

### Key Features

- **High Test Accuracy:** Achieves 96.2% test accuracy in pneumonia classification
- **Strong Validation Performance:** 90.67% validation accuracy during training
- **Robust Performance:** Precision of 95.7%, Recall of 94.8%, F1-score of 95.2%
- **Research-Oriented System:** Designed to assist clinical decision-making
- **Strong Discrimination:** AUC-ROC score of 0.977 demonstrates excellent classification capability
- **Data Augmentation:** Comprehensive preprocessing and augmentation techniques
- **Comprehensive Evaluation:** Multiple metrics including confusion matrix and cross-validation

---

## 🎯 Project Objectives

- **Analyze Effectiveness of ML Algorithms:** Compare different ML algorithms (CNNs vs SVMs) for medical image analysis
- **Evaluate Preprocessing Techniques:** Investigate the impact of normalization, augmentation, and segmentation on model performance
- **Create Prototype System:** Develop a user-friendly demonstration system for clinical pneumonia diagnosis
- **Enhance Diagnostic Accuracy:** Reduce human error and improve consistency in pneumonia detection

---

## 📊 Model Performance

### Validation Metrics (During Training)

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 90.67% |
| Precision | 90.5%  |
| Recall    | 90.7%  |
| F1-score  | 90.5%  |
| AUC-ROC   | 0.977  |

### Test Metrics (Final Evaluation)

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 96.2%  |
| Precision | 95.7%  |
| Recall    | 94.8%  |
| F1-score  | 95.2%  |
| AUC-ROC   | 0.977  |

### Comparison with Other Models

| Model              | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| SIFT               | 70.59%   | 64.29%    | 100%   | 78.26%   |
| LBP                | 82.35%   | 80%       | 88.89% | 84.21%   |
| RBM                | 58.82%   | 56.25%    | 100%   | 72%      |
| CNN (Baseline)     | 83.21%   | 87.11%    | 93.33% | 90.11%   |
| **CNN (This Project)** | **96.2%** | **95.7%** | **94.8%** | **95.2%** |

> **Note:** The baseline CNN row represents a standard unoptimized CNN. "This Project" refers to the fine-tuned model with data augmentation and optimized hyperparameters.

---

## 🏗️ Architecture

### CNN Architecture

- **Input Layer:** 224×224 pixel images (normalized to [0, 1])
- **Convolutional Layers:** Multiple layers with 64, 128, and 256 filters
- **Pooling Layers:** Max pooling for dimensionality reduction
- **Fully Connected Layers:** Dense layers with ReLU activation
- **Dropout Layer:** 0.5 dropout for regularization
- **Output Layer:** Sigmoid activation for binary classification

### Data Flow

```
Raw X-ray Images
        ↓
Image Preprocessing (Resizing, Normalization)
        ↓
Data Augmentation (Rotation, Flip, Zoom)
        ↓
Train / Validation / Test Split (70% / 15% / 15%)
        ↓
CNN Model Training
        ↓
Model Evaluation & Validation
```

---

## 📦 Dataset Split

- **Training Set:** 70% (~4,100 images)
- **Validation Set:** 15% (~880 images)
- **Test Set:** 15% (~880 images)

### Data Characteristics

- Balanced class distribution to prevent model bias
- High-quality annotations by experienced radiologists
- Images from diverse patient demographics and acquisition equipment

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- GPU support (recommended for faster training)

### Setup

```bash
# Clone the repository
git clone https://github.com/SAIDUMPALA01/medical-image-classification-system1.git
cd medical-image-classification-system1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📚 Usage

### Training the Model

```bash
python src/train.py --epochs 50 --batch-size 32 --learning-rate 0.001
```

### Making Predictions

```bash
python src/predict.py --image-path path/to/xray.jpg
```

### Running Jupyter Notebook

```bash
jupyter notebook notebooks/Medical_Image_Classification.ipynb
```

---

## 📁 Project Structure

```
pneumonia-detection-cnn/
├── README.md                           # This file
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package setup configuration
│
├── notebooks/                          # Jupyter notebooks
│   └── Medical_Image_Classification.ipynb
│
├── src/                                # Source code
│   ├── __init__.py
│   ├── model.py                        # CNN model architecture
│   ├── data_loader.py                  # Data loading utilities
│   ├── preprocessing.py                # Image preprocessing
│   ├── train.py                        # Training script
│   ├── evaluate.py                     # Evaluation metrics
│   └── predict.py                      # Inference script
│
├── models/                             # Trained model weights
│   └── pneumonia_detector.h5
│
├── data/                               # Data files
│   ├── All_models___.csv               # Model comparison results
│   └── README.md                       # Data documentation
│
├── docs/                               # Documentation
│   ├── Research_Paper.pdf              # Full research paper
│   ├── Project_Report.docx             # Detailed report
│   ├── METHODOLOGY.md                  # Research methodology
│   └── RESULTS.md                      # Results analysis
│
└── tests/                              # Unit tests
    ├── __init__.py
    ├── test_preprocessing.py
    ├── test_model.py
    └── test_evaluation.py
```

---

## 🔧 Configuration

### Hyperparameters

```python
# Training Configuration
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15

# Model Configuration
INPUT_SIZE = (224, 224)
DROPOUT_RATE = 0.5
FILTERS = [64, 128, 256]

# Data Augmentation
ROTATION_RANGE = 15
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True
```

---

## 📈 Results

### Key Findings

1. **High Test Accuracy:** The CNN model achieved 96.2% accuracy on the test set (90.67% on the validation set), outperforming traditional methods like SIFT and LBP.

2. **Low False Negative Rate:** With 94.8% recall, the model effectively identifies pneumonia cases, minimizing missed diagnoses.

3. **Balanced Performance:** The F1-score of 95.2% demonstrates excellent balance between precision and recall.

4. **Superior ROC-AUC:** The AUC-ROC of 0.977 indicates excellent discrimination between pneumonia-positive and pneumonia-negative cases.

### Confusion Matrix (Validation Set)

```
Predicted:   Normal   Pneumonia
Actual Normal:   14         4
Actual Pneumonia: 3        54

Precision: 0.905
Recall:    0.907
F1-score:  0.905
```

---

## 🔍 Preprocessing Pipeline

### Image Resizing
- Standardized to 224×224 pixels
- Reduces computational complexity while preserving diagnostic features

### Normalization
- Pixel intensities scaled to [0, 1] range
- Improves model convergence and learning stability

### Data Augmentation
- Random rotation (up to 15°)
- Horizontal flipping (for patient position variation)
- Random zoom (0.2 scale)
- Width/height shift (0.1 range)

---

## 📝 Ethical Considerations

- **Data Privacy:** All datasets anonymized with personally identifiable information (PII) removed
- **HIPAA Compliance:** Publicly available anonymized datasets were used to preserve patient privacy and confidentiality
- **GDPR Compliance:** Meets European data protection regulations
- **Patient Confidentiality:** Multiple safeguards implemented to protect patient identity

---

## 🚀 Future Work

### Extensions
- **Multi-Disease Detection:** Extend framework for tuberculosis, COVID-19, and other lung diseases
- **Real-Time Implementation:** Optimize for deployment on edge devices and low-bandwidth environments
- **Ensemble Methods:** Combine multiple models for improved robustness
- **Explainability:** Implement interpretability tools (Grad-CAM, LIME) for clinical transparency

### Improvements
- Increase training dataset diversity
- Reduce false negative rate through class weighting
- Multi-modal imaging (CT and MRI integration)
- Clinical trial validation

---

## 💡 Key Contributions

- **Automated Detection System:** Provides an objective, efficient alternative to manual interpretation
- **Clinical Decision Support:** Assists radiologists in high-volume diagnostic settings
- **Healthcare Equity:** Enables diagnosis in resource-limited settings with limited specialist availability
- **Efficiency Improvement:** Reduces diagnostic time and workload burden on radiologists

---

## 📚 Literature References

- Rajpurkar et al. (2017): CheXNet — Radiologist-Level Pneumonia Detection
- Zhang et al. (2020): Impact of Data Augmentation on Deep Learning Models
- Baltruschat et al. (2019): Deep Learning for Multi-Disease Diagnosis
- WHO (2023): Global Pneumonia Statistics

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## ✉️ Contact & Support

- **Author:** Sai Dumpala
- **Email:** dumpalasairamkrishnareddy@gmail.com
- **GitHub:** [https://github.com/SAIDUMPALA01/medical-image-classification-system1](https://github.com/SAIDUMPALA01/medical-image-classification-system1)
- **LinkedIn:** [www.linkedin.com/in/sai-dumpala](https://www.linkedin.com/in/sai-dumpala)

---

## 🙏 Acknowledgments

- Kaggle for providing the Chest X-ray Pneumonia Dataset
- TensorFlow and Keras communities for excellent deep learning frameworks
- All radiologists and medical professionals who contributed to dataset annotations
- Research institutions and universities supporting this work

## Citation

```bibtex
@software{pneumonia_detection_2025,
  author = {Sai Dumpala},
  title  = {Pneumonia Detection using Convolutional Neural Networks},
  year   = {2025},
  url    = {https://github.com/SAIDUMPALA01/medical-image-classification-system1}
}
```
