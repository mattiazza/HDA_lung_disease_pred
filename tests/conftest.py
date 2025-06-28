"""
Test configuration and fixtures for EDA utilities testing.
"""
import pytest
import tempfile
import shutil
import os
import numpy as np
from pathlib import Path
from unittest.mock import Mock


@pytest.fixture
def test_data_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_models_dir():
    """Create a temporary directory for test models"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_logs_dir():
    """Create a temporary directory for test logs"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config():
    """Test configuration with environment variables or defaults"""
    return {
        'data_dir': os.getenv('TEST_DATA_DIR', 'test_data'),
        'models_dir': os.getenv('TEST_MODELS_DIR', 'test_models'),
        'logs_dir': os.getenv('TEST_LOGS_DIR', 'test_logs'),
        'base_url': os.getenv('TEST_BASE_URL', 'https://zenodo.org/records/10519652/files'),
        'test_dataset_name': os.getenv('TEST_DATASET_NAME', 'chestmnist_64'),
        'figure_size': (
            int(os.getenv('TEST_FIGURE_WIDTH', '10')),
            int(os.getenv('TEST_FIGURE_HEIGHT', '6'))
        ),
        'random_seed': int(os.getenv('TEST_RANDOM_SEED', '42'))
    }


@pytest.fixture
def sample_label_names():
    """Sample label names for testing"""
    return {
        0: 'Atelectasis',
        1: 'Cardiomegaly', 
        2: 'Effusion'
    }


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing"""
    np.random.seed(42)  # For reproducible tests
    
    n_train, n_val, n_test = 10, 5, 3
    n_labels = 3
    img_size = 64
    
    # Create realistic binary label data
    train_labels = np.random.binomial(1, 0.2, (n_train, n_labels))
    val_labels = np.random.binomial(1, 0.2, (n_val, n_labels))
    test_labels = np.random.binomial(1, 0.2, (n_test, n_labels))
    
    # Ensure at least one sample per label for testing
    for i in range(n_labels):
        if i < n_train:
            train_labels[i, i] = 1
    
    # Add some healthy samples (all zeros)
    train_labels[-2:] = 0
    
    return {
        'train_images': np.random.randint(0, 255, (n_train, img_size, img_size), dtype=np.uint8),
        'train_labels': train_labels,
        'val_images': np.random.randint(0, 255, (n_val, img_size, img_size), dtype=np.uint8),
        'val_labels': val_labels,
        'test_images': np.random.randint(0, 255, (n_test, img_size, img_size), dtype=np.uint8),
        'test_labels': test_labels
    }


@pytest.fixture
def mock_npz_file(mock_dataset):
    """Create a mock NpzFile object"""
    mock_file = Mock()
    mock_file.__getitem__ = lambda self, key: mock_dataset[key]
    mock_file.keys = lambda: mock_dataset.keys()
    return mock_file
