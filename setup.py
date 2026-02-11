from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="medical-image-classification-system",
    version="1.0.0",
    author="Sai Ram",
    author_email="your.email@example.com",
    description="Deep learning-based medical image classification system for automated pneumonia detection from chest X-ray images using CNNs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SAIDUMPALA01/medical-image-classification-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.12.0",
        "keras>=2.12.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "opencv-python>=4.8.0",
        "scikit-image>=0.21.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.66.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "pylint>=2.17.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pneumonia-predict=src.predict:main",
            "pneumonia-train=src.train:main",
        ],
    },
    keywords=[
        "pneumonia detection",
        "deep learning",
        "CNN",
        "medical imaging",
        "chest x-ray",
        "healthcare AI",
        "machine learning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/pneumonia-detection-cnn/issues",
        "Source": "https://github.com/yourusername/pneumonia-detection-cnn",
        "Documentation": "https://github.com/yourusername/pneumonia-detection-cnn/wiki",
    },
)
