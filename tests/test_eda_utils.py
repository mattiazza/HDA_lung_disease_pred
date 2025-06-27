import pytest
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, mock_open, MagicMock
import matplotlib.pyplot as plt
from numpy.lib.npyio import NpzFile

from HDA_lung_disease_pred.utils.eda_utils import (
    download_MNIST_dataset,
    get_label_names,
    get_all_labels,
    plot_label_distribution,
    plot_n_label_per_sample,
    plot_image_per_label
)


class TestDownloadMNISTDataset:
    """Tests for download_MNIST_dataset function"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    @patch('src.project_c3.utils.eda_utils.requests.get')
    @patch('src.project_c3.utils.eda_utils.np.load')
    def test_download_success(self, mock_load, mock_get):
        """Test successful download of dataset"""
        # Mock response
        mock_response = Mock()
        mock_response.iter_content.return_value = [b'test_data_chunk']
        mock_get.return_value = mock_response
        
        # Mock numpy load
        mock_data = Mock()
        mock_load.return_value = mock_data
        
        result = download_MNIST_dataset("chestmnist_64")
        
        # Verify directory creation
        assert os.path.exists("data")
        
        # Verify correct URL was called
        expected_url = "https://zenodo.org/records/10519652/files/chestmnist_64.npz?download=1"
        mock_get.assert_called_once_with(expected_url, stream=True)
        
        # Verify file was loaded
        mock_load.assert_called_once_with("data/chestmnist_64.npz")
        assert result == mock_data
    
    @patch('src.project_c3.utils.eda_utils.np.load')
    def test_file_already_exists(self, mock_load):
        """Test behavior when file already exists"""
        # Create data directory and file
        os.makedirs("data", exist_ok=True)
        with open("data/chestmnist_64.npz", "w") as f:
            f.write("test")
        
        mock_data = Mock()
        mock_load.return_value = mock_data
        
        result = download_MNIST_dataset("chestmnist_64")
        
        # Verify file was loaded without download
        mock_load.assert_called_once_with("data/chestmnist_64.npz")
        assert result == mock_data
    
    @patch('src.project_c3.utils.eda_utils.requests.get')
    def test_download_http_error(self, mock_get):
        """Test handling of HTTP errors"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_get.return_value = mock_response
        
        with pytest.raises(Exception, match="HTTP Error"):
            download_MNIST_dataset("chestmnist_64")


class TestGetLabelNames:
    """Tests for get_label_names function"""
    
    @patch('src.project_c3.utils.eda_utils.INFO')
    def test_get_label_names_success(self, mock_info):
        """Test successful extraction of label names"""
        # Mock INFO data
        mock_info.__getitem__.return_value = {
            'label': {
                '0': 'Atelectasis',
                '1': 'Cardiomegaly',
                '2': 'Effusion'
            }
        }
        
        result = get_label_names("chestmnist")
        
        expected = {0: 'Atelectasis', 1: 'Cardiomegaly', 2: 'Effusion'}
        assert result == expected
    
    @patch('src.project_c3.utils.eda_utils.INFO')
    def test_get_label_names_empty(self, mock_info):
        """Test with empty label dictionary"""
        mock_info.__getitem__.return_value = {'label': {}}
        
        result = get_label_names("chestmnist")
        
        assert result == {}


class TestGetAllLabels:
    """Tests for get_all_labels function"""
    
    def test_get_all_labels_success(self):
        """Test successful concatenation of all labels"""
        # Create mock dataset
        mock_dataset = {
            'train_labels': np.array([[1, 0, 1], [0, 1, 0]]),
            'val_labels': np.array([[1, 1, 0], [0, 0, 1]]),
            'test_labels': np.array([[0, 1, 1]])
        }
        
        result = get_all_labels(mock_dataset)
        
        expected = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 1]
        ])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_get_all_labels_empty(self):
        """Test with empty datasets"""
        mock_dataset = {
            'train_labels': np.array([]).reshape(0, 3),
            'val_labels': np.array([]).reshape(0, 3),
            'test_labels': np.array([]).reshape(0, 3)
        }
        
        result = get_all_labels(mock_dataset)
        
        assert result.shape == (0, 3)


class TestPlotLabelDistribution:
    """Tests for plot_label_distribution function"""
    
    @patch('src.project_c3.utils.eda_utils.plt.show')
    @patch('src.project_c3.utils.eda_utils.sns.barplot')
    @patch('src.project_c3.utils.eda_utils.plt.figure')
    @patch('src.project_c3.utils.eda_utils.plt.title')
    @patch('src.project_c3.utils.eda_utils.plt.xlabel')
    @patch('src.project_c3.utils.eda_utils.plt.ylabel')
    @patch('src.project_c3.utils.eda_utils.plt.xticks')
    @patch('src.project_c3.utils.eda_utils.plt.tight_layout')
    def test_plot_label_distribution(self, mock_tight_layout, mock_xticks, mock_ylabel, 
                                   mock_xlabel, mock_title, mock_figure, mock_barplot, mock_show):
        """Test label distribution plotting"""
        # Create mock dataset
        mock_dataset = {
            'train_labels': np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]]),
            'val_labels': np.array([[1, 1, 0], [0, 0, 1]]),
            'test_labels': np.array([[0, 1, 1]])
        }
        
        label_names = {0: 'Label1', 1: 'Label2', 2: 'Label3'}
        
        # Call function
        plot_label_distribution(mock_dataset, label_names)
        
        # Verify plotting functions were called
        mock_figure.assert_called_once_with(figsize=(10, 6))
        mock_barplot.assert_called_once()
        mock_title.assert_called_once_with("Distribution of Labels in ChestMNIST Datasets")
        mock_xlabel.assert_called_once_with("Labels")
        mock_ylabel.assert_called_once_with("Count")
        
        # Check that barplot was called with correct data
        call_args = mock_barplot.call_args
        # Calculate expected counts manually: sum across all labels for each position
        # Label 0: appears in positions [0,2] in first sample, [0] in third sample, [0] in fourth sample = 3 total
        # Label 1: appears in position [1] in second sample, [1] in fourth sample, [1] in last sample = 3 total  
        # Label 2: appears in positions [0,2] in first sample, [2] in fifth sample, [1] in last sample = 3 total
        expected_counts = np.array([3, 3, 3])  # All labels appear 3 times each
        
        # Verify the y values (counts) are correct
        np.testing.assert_array_equal(call_args[1]['y'], expected_counts)


class TestPlotNLabelPerSample:
    """Tests for plot_n_label_per_sample function"""
    
    @patch('src.project_c3.utils.eda_utils.plt.show')
    @patch('src.project_c3.utils.eda_utils.sns.histplot')
    @patch('src.project_c3.utils.eda_utils.plt.figure')
    @patch('src.project_c3.utils.eda_utils.plt.gca')
    def test_plot_n_label_per_sample(self, mock_gca, mock_figure, mock_histplot, mock_show):
        """Test number of labels per sample plotting"""
        # Create mock dataset
        mock_dataset = {
            'train_labels': np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]]),  # 2, 1, 3 labels
            'val_labels': np.array([[1, 1, 0], [0, 0, 0]]),  # 2, 0 labels
            'test_labels': np.array([[0, 1, 1]])  # 2 labels
        }
        
        # Mock axes and patches
        mock_ax = Mock()
        mock_patches = [Mock() for _ in range(4)]  # 4 different label counts (0, 1, 2, 3)
        
        for i, patch in enumerate(mock_patches):
            patch.get_x.return_value = i - 0.4
            patch.get_width.return_value = 0.8
            patch.get_height.return_value = [1, 1, 3, 1][i]  # Mock heights
        
        mock_ax.patches = mock_patches
        mock_gca.return_value = mock_ax
        
        # Call function
        plot_n_label_per_sample(mock_dataset)
        
        # Verify plotting functions were called
        mock_figure.assert_called_once_with(figsize=(10, 6))
        mock_histplot.assert_called_once()
        
        # Verify text was added to patches
        assert mock_ax.text.call_count == 4


class TestPlotImagePerLabel:
    """Tests for plot_image_per_label function"""
    
    @patch('src.project_c3.utils.eda_utils.plt.show')
    @patch('src.project_c3.utils.eda_utils.plt.subplot')
    @patch('src.project_c3.utils.eda_utils.plt.imshow')
    @patch('src.project_c3.utils.eda_utils.plt.figure')
    @patch('src.project_c3.utils.eda_utils.plt.title')
    @patch('src.project_c3.utils.eda_utils.plt.axis')
    def test_plot_image_per_label(self, mock_axis, mock_title, mock_figure, mock_imshow, mock_subplot, mock_show):
        """Test image per label plotting"""
        # Create mock dataset with images
        train_images = np.random.randint(0, 255, (10, 64, 64), dtype=np.uint8)
        train_labels = np.array([
            [1, 0, 0],  # Only label 0
            [0, 1, 0],  # Only label 1
            [0, 0, 1],  # Only label 2
            [0, 0, 0],  # No labels (healthy)
            [1, 1, 0],  # Multiple labels
            [0, 0, 0],  # Another healthy
            [1, 0, 0],  # Another label 0
            [0, 1, 0],  # Another label 1
            [0, 0, 1],  # Another label 2
            [0, 0, 0],  # Another healthy
        ])
        
        mock_dataset = {
            'train_images': train_images,
            'train_labels': train_labels
        }
        
        label_names = {0: 'Label0', 1: 'Label1', 2: 'Label2'}
        
        # Call function
        plot_image_per_label(mock_dataset, label_names)
        
        # Verify plotting functions were called
        mock_figure.assert_called_once_with(figsize=(15, 10))
        
        # Should be called 4 times: 3 labels + 1 healthy
        assert mock_subplot.call_count == 4
        assert mock_imshow.call_count == 4
        mock_show.assert_called_once()
    
    @patch('src.project_c3.utils.eda_utils.plt.show')
    @patch('src.project_c3.utils.eda_utils.plt.subplot')
    @patch('src.project_c3.utils.eda_utils.plt.imshow')
    @patch('src.project_c3.utils.eda_utils.plt.figure')
    @patch('src.project_c3.utils.eda_utils.plt.title')
    @patch('src.project_c3.utils.eda_utils.plt.axis')
    def test_plot_image_per_label_no_healthy(self, mock_axis, mock_title, mock_figure, mock_imshow, mock_subplot, mock_show):
        """Test plotting when no healthy samples exist"""
        # Create mock dataset without healthy samples
        train_images = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
        train_labels = np.array([
            [1, 0, 0],  # Only label 0
            [0, 1, 0],  # Only label 1
            [0, 0, 1],  # Only label 2
        ])
        
        mock_dataset = {
            'train_images': train_images,
            'train_labels': train_labels
        }
        
        label_names = {0: 'Label0', 1: 'Label1', 2: 'Label2'}
        
        # Call function - should handle case where no healthy samples exist
        with pytest.raises(IndexError):
            plot_image_per_label(mock_dataset, label_names)


class TestIntegration:
    """Integration tests combining multiple functions"""
    
    def test_full_workflow_simulation(self):
        """Test a complete workflow with mock data"""
        # Create realistic mock dataset
        n_samples = 100
        n_labels = 14
        
        # Generate realistic label distribution (some labels more common)
        np.random.seed(42)
        train_labels = np.random.binomial(1, 0.1, (n_samples, n_labels))
        val_labels = np.random.binomial(1, 0.1, (20, n_labels))
        test_labels = np.random.binomial(1, 0.1, (30, n_labels))
        
        # Ensure at least some samples have each label
        for i in range(n_labels):
            train_labels[i, i] = 1
        
        # Add some healthy samples (all zeros)
        train_labels[-5:] = 0
        
        mock_dataset = {
            'train_images': np.random.randint(0, 255, (n_samples, 64, 64), dtype=np.uint8),
            'train_labels': train_labels,
            'val_labels': val_labels,
            'test_labels': test_labels
        }
        
        # Test get_all_labels
        all_labels = get_all_labels(mock_dataset)
        assert all_labels.shape == (150, n_labels)  # 100 + 20 + 30
        
        # Verify concatenation is correct
        expected = np.concatenate([train_labels, val_labels, test_labels])
        np.testing.assert_array_equal(all_labels, expected)
        
        # Test with real label names structure
        label_names = {i: f'Disease_{i}' for i in range(n_labels)}
        
        # These should run without error (plotting functions)
        with patch('src.project_c3.utils.eda_utils.plt.show'):
            with patch('src.project_c3.utils.eda_utils.plt.figure'):
                with patch('src.project_c3.utils.eda_utils.sns.barplot'):
                    plot_label_distribution(mock_dataset, label_names)
                
                with patch('src.project_c3.utils.eda_utils.sns.histplot'):
                    with patch('src.project_c3.utils.eda_utils.plt.gca') as mock_gca:
                        mock_ax = Mock()
                        mock_ax.patches = []
                        mock_gca.return_value = mock_ax
                        plot_n_label_per_sample(mock_dataset)


if __name__ == "__main__":
    pytest.main([__file__])