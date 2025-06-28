# 🫁 Lung Disease Prediction from Chest X-ray Images

This repository contains the final project for the MSc course **Human Data Analytics** at the University of Padua.  
The goal is to develop, train, and evaluate deep learning models for lung disease prediction using chest X-ray images from the [ChestMNIST dataset](https://zenodo.org/records/10519652).

---

## 📁 Project Structure

```
HDA_lung_disease_pred/
├── HDA_project_C3.ipynb          # Main notebook for exploration and development
├── .gitignore                   # Git ignore rules
├── data/                          # Data directory (datasets stored locally)
│   └── chestmnist_64.npz         # ChestMNIST 64x64 resolution dataset
├── HDA_lung_disease_pred/        # Main package directory
│   ├── __init__.py
│   ├── models/                   # Custom model architectures
│   │   ├── __init__.py
│   │   └── cnn_baseline.py       # Baseline CNN implementation
│   ├── scripts/                  # Training and prediction scripts
│   │   ├── __init__.py
│   │   ├── train.py             # Training script with utilities
│   │   └── demo.py              # Demo script
│   └── utils/                    # Utility functions and preprocessing
│       ├── __init__.py
│       ├── data_preparation.py   # Dataset loading and preprocessing
│       └── plot_results.py     # Visualization and plotting functions
├── tests/                        # Unit tests directory (cleaned up)
│   ├── __init__.py
│   └── conftest.py              # Test fixtures and configuration
├── models/                       # Saved trained models
│   ├── cnn_baseline_64_best.keras    # Best model checkpoint
│   └── cnn_baseline_64_final.keras   # Final trained model
├── logs/                         # Training logs and history
│   └── cnn_baseline_64_history.json  # Training history
├── plots/                        # Saved plots and visualizations
├── pyproject.toml               # Project configuration (PEP 621)
├── poetry.lock                  # Dependency lock file
└── README.md                    # Project documentation
```

---

## 📊 Project Overview

### Task
Multi-label classification of **14 lung diseases** from **frontal chest X-ray images**, based on the ChestMNIST dataset (MedMNIST v2).

### Dataset
- **112,120 X-ray images** across multiple resolutions (64×64, 128×128, 224×224)
- **30,805 unique patients**  
- **14 disease categories**: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural, Hernia
- **Multi-label classification** (some images have multiple disease labels)
- Source: [MedMNIST v2 - ChestMNIST](https://zenodo.org/records/10519652)

### Current Implementation Status ✅
- **Data Pipeline**: Complete dataset download and loading functionality with configurable paths
- **Exploratory Data Analysis**: Label distribution analysis, image visualization, multi-label statistics
- **Baseline Model**: CNN architecture with configurable parameters trained on 64×64 images
- **Training Framework**: Complete training pipeline with early stopping, callbacks, and live plotting
- **Environment**: Poetry-based dependency management with PEP 621 configuration
- **Testing Suite**: Clean test infrastructure with configurable paths
- **Visualization**: Comprehensive plotting functions with save capabilities for reports

### Objectives
- Design original deep learning architectures in **TensorFlow**
- Compare multiple approaches (e.g., baseline CNN, attention)
- Evaluate classification performance using proper metrics (AUC, F1-score, etc.)
- Analyze computational complexity and training time
- Write a clear and structured scientific report using LaTeX

---

## ⚙️ Quick Setup

### Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd HDA_lung_disease_pred

# Install dependencies
poetry install

# Activate the environment
poetry shell
```

### Configuration Options
The project uses environment variables with sensible defaults:
```python
# Data directories
DATA_DIR = os.getenv("DATA_DIR", "./data")
BASE_URL = os.getenv("BASE_URL", "https://zenodo.org/records/10519652/files")

# Figure settings
FIGURE_WIDTH = int(os.getenv("FIGURE_WIDTH", "10"))
FIGURE_HEIGHT = int(os.getenv("FIGURE_HEIGHT", "6"))
```

---

## 🧪 Testing

### Test Infrastructure
- **Clean test environment** with configurable paths
- **Isolated directories** using temporary files  
- **CI/CD ready** with proper mocking
- **Environment-based configuration**

### Running Tests
```bash
# Run all tests
poetry run pytest tests/ -v

# Run with custom configuration
TEST_RANDOM_SEED=123 poetry run pytest tests/ -v
```

### Test Configuration
Tests use environment variables with defaults:
```python
TEST_DATA_DIR = os.getenv('TEST_DATA_DIR', 'test_data')
TEST_RANDOM_SEED = int(os.getenv('TEST_RANDOM_SEED', '42'))
```

---

## 🛠️ Technologies & Dependencies

- **Python 3.10+** (managed with Poetry)
- **TensorFlow 2.19+** (deep learning framework)
- **MedMNIST 3.0+** (medical dataset library)
- **NumPy, Pandas** (data manipulation)
- **Matplotlib, Seaborn** (visualization)
- **scikit-learn** (evaluation metrics)
- **Jupyter** (interactive development)
- **pytest** (testing framework)

---

## 🧪 Evaluation Metrics

For multi-label classification, the following metrics are implemented:

- **Precision & Recall** (per-class and macro-averaged)
- **F1-Score** (macro and micro, handling class imbalance)  
- **AUC** (area under ROC curve, appropriate for binary relevance)
- **Subset Accuracy** (exact match for complete label sets)
- **Hamming Score** (label-wise accuracy)
- **Training Efficiency** (time, memory usage, convergence)

### Important Note on Metrics
❌ **Binary Accuracy** is NOT appropriate for multi-label classification  
✅ **Use Precision, Recall, F1, AUC** for meaningful evaluation

---

## 🚀 Quick Start

### Data Loading and EDA
```python
from HDA_lung_disease_pred.utils.data_preparation import download_MNIST_dataset, create_tf_dataset
from HDA_lung_disease_pred.utils.plot_results import plot_label_distribution

# Load dataset
dataset = download_MNIST_dataset("chestmnist_64")
train_data, val_data, test_data = create_tf_dataset(dataset, batch_size=32)

# EDA and visualization
plot_label_distribution(dataset, label_names, save=True)
```

### Model Training
```python
from HDA_lung_disease_pred.models.cnn_baseline import cnn_baseline_model
from HDA_lung_disease_pred.scripts.train import train_model

# Create and train model
model = cnn_baseline_model(
    input_shape=(64, 64, 1), 
    num_classes=14,
    metrics=['precision', 'recall']  # Multi-label appropriate metrics
)

# Train with live plotting
trained_model, history = train_model(
    model=model,
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    epochs=10,
    patience=5,
    save_history=True
)
```

---

## 📊 Results Summary

### Baseline CNN (64×64 images)
- **Architecture**: 3 Conv2D layers + 2 Dense layers, configurable parameters
- **Multi-label Classification**: Sigmoid activation, binary crossentropy loss
- **Training Features**: Early stopping, learning rate scheduling, live plotting
- **Metrics**: Precision, Recall, F1-Score, AUC (appropriate for multi-label tasks)
- **Status**: ✅ Successfully implemented with comprehensive training pipeline

### Key Features
- **Live Training Plots**: Real-time loss and metrics visualization
- **Plot Saving**: Automatic plot saving for reports in `plots/` directory
- **Flexible Architecture**: Configurable filters, dropout, learning rate
- **Multi-label Ready**: Proper metrics and loss functions for medical diagnosis

---

## � Documentation

### Key Files
- **`HDA_project_C3.ipynb`**: Complete workflow from data exploration to model training
- **`HDA_lung_disease_pred/models/cnn_baseline.py`**: Configurable CNN architecture implementation  
- **`HDA_lung_disease_pred/utils/data_preparation.py`**: Data loading with tf.data.Dataset support
- **`HDA_lung_disease_pred/utils/plot_results.py`**: Visualization functions with save capabilities
- **`HDA_lung_disease_pred/scripts/train.py`**: Complete training pipeline with live plotting

### Project Features
- 🔍 **Comprehensive EDA**: Label distribution, multi-label analysis, sample visualization
- 🏗️ **Modular Architecture**: Clean separation of models, utilities, and training scripts
- 📦 **Modern Python Setup**: Poetry dependency management, PEP 621 configuration
- 🧪 **Clean Testing**: Test infrastructure with configurable paths and isolated environments
- ⚙️ **Environment Configuration**: Flexible configuration via environment variables
- 🔒 **CI/CD Ready**: Tests work in clean environments without side effects
- 📊 **Complete Pipeline**: From raw data to trained model with live monitoring
- 💾 **Report Ready**: Automatic plot saving for academic reports
- 🎯 **Multi-label Focus**: Proper metrics and evaluation for medical diagnosis

---

## 🎯 Future Work

- [ ] Implement attention-based architectures
- [ ] Experiment with higher resolution images (128×128, 224×224)
- [ ] Add data augmentation techniques
- [ ] Implement ensemble methods
- [ ] Create comprehensive evaluation report
- [ ] Deploy model for inference

---

## 📚 References

- Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification." Scientific Data 10.1 (2023): 41.
- Wang, Xiaosong, et al. "ChestX-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases." CVPR 2017.
- [ChestMNIST Dataset](https://zenodo.org/records/10519652) - Zenodo Repository

---

## 👨‍� Author

**Mattia Piazza** (ID: 2073223)  
MSc Data Science, University of Padua  
📧 mattiapiazza@gmail.com

---

*Developed for the Human Data Analytics course – MSc in Data Science @ University of Padua*