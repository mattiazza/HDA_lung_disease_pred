import os
import requests
from medmnist import INFO
import numpy as np
from numpy.lib.npyio import NpzFile
import tensorflow as tf
from typing import Tuple, Dict



# Configuration from environment variables with defaults
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "./data")
DEFAULT_BASE_URL = os.getenv("BASE_URL", "https://zenodo.org/records/10519652/files")

DEFAULT_BATCH_SIZE = 32
DEFAULT_SHUFFLE = False
DEFAULT_BUFFER_SIZE = 10000  # Default buffer size for shuffling
DEFAULT_USE_CLASS_WEIGHTS = False




def download_MNIST_dataset(dataset_name: str, output_dir: str = None, base_url: str = None) -> NpzFile:
    """
    Downloads the MNIST dataset images and saves them to the specified directory
    """
    if output_dir is None:
        output_dir = DEFAULT_DATA_DIR
    if base_url is None:
        base_url = DEFAULT_BASE_URL

    # Create data directory if needed
    os.makedirs(output_dir, exist_ok=True)

    url = f"{base_url}/{dataset_name}.npz?download=1"
    save_path = os.path.join(output_dir, f"{dataset_name}.npz")
    
    # Check if the file already exists
    if os.path.exists(save_path):
        print(f"File {save_path} already exists. Skipping download.")
        return np.load(save_path)

    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise exception for HTTP errors

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded file to {save_path}")
    return np.load(save_path)



def get_label_names(dataset_name: str) -> dict:
    """
    Extracts and prints the labels from the medmnist INFO.
    """
    
    #extract dataset information
    info = INFO[dataset_name]

    label_names = dict()

    for idx, label_name in info['label'].items():
        label_names[int(idx)] = label_name

    # # Print all the labels
    # print("idx: label_name \n")
    # for idx, label_name in lbls_dict.items():
    #     print(f"{idx}: {label_name}")

    return label_names



def get_all_labels(dataset: NpzFile) -> np.ndarray:
    """
    Returns all the labels from the ChestMNIST dataset.
    """

    all_labels = np.concatenate([
        dataset["train_labels"],
        dataset["val_labels"],
        dataset["test_labels"]
    ])
    return all_labels



def load_set(dataset: NpzFile, set_name: str = "train", n_sample: int = None, buffer_size: int = None) -> tf.data.Dataset:
    """
    Load the specified set (train, val, or test) from the dataset and normalize the images.
    """

    X = dataset[f"{set_name}_images"].astype(np.float32) / 255.0
    y = dataset[f"{set_name}_labels"].astype(np.int32)


    if len(X.shape[1:]) < 3:
        X = np.expand_dims(X, axis=-1)
    
    if n_sample is not None:
        X = X[:n_sample]
        y = y[:n_sample]

    # Create a TensorFlow dataset with appropriate shuffle buffer
    tf_dataset = tf.data.Dataset.from_tensor_slices((X, y))

    print(f"Loaded {set_name} set with shape: {X.shape}, labels shape: {y.shape}")

    return tf_dataset


def get_input_shape(dataset : tf.data.Dataset) -> Tuple[int, int, int]:
    """
    Extract input shape from a tf.data.Dataset
    """
    for X, _ in dataset.take(1):
        # Get the shape of the first sample
        input_shape = X.shape  # Exclude batch dimension
        return tuple(input_shape)
    
    raise ValueError("Dataset is empty, cannot determine input shape.")
    


def calculate_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced multi-label classification
    """
    class_weights = {}
    n_classes = y_train.shape[1]
    
    for i in range(n_classes):
        # For each class, calculate weight based on frequency
        positive_samples = np.sum(y_train[:, i])
        negative_samples = len(y_train) - positive_samples
        
        if positive_samples > 0:
            weight = negative_samples / positive_samples
        else:
            weight = 1.0
            
        class_weights[i] = weight
    
    return class_weights


def calculate_sample_weights(y_train: np.ndarray) -> np.ndarray:
    """
    Calculte sample weights for multi-label data.
    Each sample gets weighted based on its active classes.
    """
    
    class_weights = calculate_class_weights(y_train)

    sample_weights = np.ones(len(y_train))
    
    for i, sample in enumerate(y_train):
        # Average the weights of active classes
        active_classes = np.where(sample == 1)[0]
        
        if len(active_classes) > 0:
            weights_for_sample = [class_weights[cls] for cls in active_classes]
            sample_weights[i] = np.mean(weights_for_sample)
    
    return sample_weights


def prepare_dataset(
    dataset: tf.data.Dataset,
    batch_size: int = None,
    shuffle: bool = None,
    buffer_size: int = None,
    use_class_weights: bool = None
) -> tf.data.Dataset:
    """
    Prepare the dataset for training, validation, or testing.
    """
    
    batch_size = batch_size or DEFAULT_BATCH_SIZE
    shuffle = shuffle or DEFAULT_SHUFFLE
    buffer_size = buffer_size or DEFAULT_BUFFER_SIZE
    use_class_weights = use_class_weights or DEFAULT_USE_CLASS_WEIGHTS



    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)


    if use_class_weights:
        # Extract all labels to calculate weights
        all_labels = []
        for _, labels in dataset:
            all_labels.append(labels.numpy())
        all_labels = np.vstack(all_labels)
        
        sample_weights_array = calculate_sample_weights(all_labels)
        
        # Create sample weights dataset and zip
        sample_weight_dataset = tf.data.Dataset.from_tensor_slices(sample_weights_array)
        dataset = tf.data.Dataset.zip((dataset, sample_weight_dataset))
        
        # Use lambda to unpack the nested structure
        dataset = dataset.map(lambda data_sample_weight: (
            data_sample_weight[0][0],  # X
            data_sample_weight[0][1],  # y  
            data_sample_weight[1]      # sample_weight
        ))
        
        print("\nUsing sample weights for class imbalance")


    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset


def prepare_data(dataset_name: str = "chestmnist_64", 
                batch_size: int = 32,
                n_sample_train: int = None,
                n_sample_val: int = None,
                n_sample_test: int = None,
                use_class_weights: bool = False) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Tuple[int, int, int]]:
    """
    Prepare training, validation, and test datasets all at once.
    
    Args:
        dataset_name: Name of the dataset to download (e.g., "chestmnist_64", "chestmnist_128")
        batch_size: Batch size for all datasets
        n_sample_train: Number of training samples to use (None for all)
        n_sample_val: Number of validation samples to use (None for all)
        n_sample_test: Number of test samples to use (None for all)
        use_class_weights: Whether to use class weights for training data
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, input_shape)
    """
    print(f"📁 Preparing datasets for {dataset_name}...")
    
    # Download dataset
    dataset = download_MNIST_dataset(dataset_name)
    
    # Load datasets
    train_dataset = load_set(dataset, "train", n_sample=n_sample_train)
    val_dataset = load_set(dataset, "val", n_sample=n_sample_val) 
    test_dataset = load_set(dataset, "test", n_sample=n_sample_test)
    
    # Get input shape from first batch
    for X, _ in train_dataset.take(1):
        input_shape = tuple(X.shape)
        break
    
    # Prepare datasets with batching
    train_dataset = prepare_dataset(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        use_class_weights=use_class_weights
    )
    val_dataset = prepare_dataset(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    test_dataset = prepare_dataset(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print(f"✅ Datasets prepared - Input shape: {input_shape}")
    print(f"   • Training: batched and shuffled")
    print(f"   • Validation: batched")
    print(f"   • Test: batched")
    if use_class_weights:
        print(f"   • Class weights applied to training data")
    
    return train_dataset, val_dataset, test_dataset, input_shape

