# 🫁 Lung Disease Prediction from Chest X-ray Images

This repository contains the final project for the MSc course **Human Data Analytics** at the University of Padua.  
The goal is to develop, train, and evaluate deep learning models for lung disease prediction using chest X-ray images from the [ChestMNIST dataset](https://zenodo.org/records/10519652).

---

## 📁 Project Structure

```
HDA_lung_disease_pred/
├── HDA_project_C3.ipynb          # Main notebook for exploration and development
├── data/                          # Data directory (datasets stored locally)
│   ├── chestmnist_64.npz         # ChestMNIST 64x64 resolution dataset
│   ├── chestmnist_128.npz        # ChestMNIST 128x128 resolution dataset
│   └── chestmnist_224.npz        # ChestMNIST 224x224 resolution dataset
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
│       ├── eda_utils.py         # Exploratory data analysis utilities
│       ├── plot_results.py     # Visualization and plotting functions
│       └── train_utils.py       # Training utilities and helpers
├── tests/                        # Unit tests
│   ├── __init__.py
│   └── test_eda_utils.py        # Tests for EDA functions
├── models/                       # Saved trained models
│   ├── cnn_baseline_64_best.keras    # Best model checkpoint
│   └── cnn_baseline_64_final.keras   # Final trained model
├── logs/                         # Training logs and history
│   └── cnn_baseline_64_history.json  # Training history
├── pyproject.toml               # Project configuration (Poetry)
├── poetry.lock                  # Dependency lock file
├── .gitignore                   # Git ignore rules
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
- **Data Pipeline**: Complete dataset download and loading functionality
- **Exploratory Data Analysis**: Label distribution analysis, image visualization, multi-label statistics
- **Baseline Model**: CNN architecture with 684K parameters trained on 64×64 images
- **Training Framework**: Complete training pipeline with early stopping, class weights, and validation
- **Environment**: Poetry-based dependency management with PEP 621 configuration

### Objectives
- Design original deep learning architectures in **TensorFlow**
- Compare multiple approaches (e.g., baseline CNN, attention)
- Evaluate classification performance using proper metrics (AUC, F1-score, etc.)
- Analyze computational complexity and training time
- Write a clear and structured scientific report using LaTeX

---

## 🛠️ Technologies & Dependencies

- **Python 3.10+** (managed with Poetry)
- **TensorFlow 2.19+** (deep learning framework)
- **MedMNIST 3.0+** (medical dataset library)
- **NumPy, Pandas** (data manipulation)
- **Matplotlib, Seaborn** (visualization)
- **scikit-learn** (evaluation metrics)
- **Jupyter** (interactive development)

---

## 🧪 Evaluation Metrics

- **Accuracy** (overall classification performance)
- **F1-score** (macro & per-class, handling class imbalance)  
- **AUC** (area under ROC curve, macro & per-class)
- **Precision & Recall** (per-class performance)
- **Confusion Matrix** (detailed classification results)
- **Training Efficiency** (time, memory usage, convergence)

---

## � Quick Start

To run the complete pipeline:

```python
# In the Jupyter notebook or Python script
from HDA_lung_disease_pred.utils.eda_utils import download_MNIST_dataset
from HDA_lung_disease_pred.models.cnn_baseline import cnn_baseline_model
from HDA_lung_disease_pred.scripts.train import train_model

# Download and load data
dataset = download_MNIST_dataset("chestmnist_64")

# Create and train model
model = cnn_baseline_model(input_shape=(64, 64, 1), num_classes=14)
trained_model, history = train_model(model, train_data, val_data, test_data)
```

---

## 📊 Results Summary

### Baseline CNN (64×64 images)
- **Architecture**: 3 Conv2D layers + 2 Dense layers
- **Parameters**: 684,430 (2.61 MB)
- **Training**: 10 epochs with early stopping
- **Dataset**: 1,000 training / 200 validation / 200 test samples
- **Status**: ✅ Successfully trained and evaluated

---

## � Documentation

### Key Files
- **`HDA_project_C3.ipynb`**: Complete workflow from data exploration to model training
- **`HDA_lung_disease_pred/models/cnn_baseline.py`**: Baseline CNN architecture implementation  
- **`HDA_lung_disease_pred/utils/eda_utils.py`**: Data analysis and visualization functions
- **`HDA_lung_disease_pred/scripts/train.py`**: Training pipeline with validation and callbacks

### Project Features
- 🔍 **Comprehensive EDA**: Label distribution, multi-label analysis, sample visualization
- 🏗️ **Modular Architecture**: Clean separation of models, utilities, and training scripts
- 📦 **Modern Python Setup**: Poetry dependency management, PEP 621 configuration
- 🧪 **Robust Testing**: Unit tests for core functionality
- 📊 **Complete Pipeline**: From raw data to trained model evaluation

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