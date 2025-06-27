import numpy as np
from numpy.lib.npyio import NpzFile



def load_set(dataset: NpzFile, set: str = "train", n_sample: int = None)-> tuple[np.ndarray, np.ndarray]:
    """
    Load the specified set (train, val, or test) from the dataset and normalize the images.

    Args:
        dataset (NpzFile): The dataset loaded from a .npz file.
        set (str): The set to load, can be "train", "val", or "test".
    """

    X = dataset[f"{set}_images"].astype(np.float32) / 255.0
    y = dataset[f"{set}_labels"].astype(np.float32)

    if len(X.shape[1:]) < 3:
        X = np.expand_dims(X, axis=-1)
    
    if n_sample is not None:
        X = X[:n_sample]
        y = y[:n_sample]

    print(f"Loaded {set} set with shape: {X.shape}, labels shape: {y.shape}")


    return X, y