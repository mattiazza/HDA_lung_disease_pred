import numpy as np
from numpy.lib.npyio import NpzFile



def load_set(dataset: NpzFile, set: str = "train"):
    """
    Load the specified set (train, val, or test) from the dataset and normalize the images.

    Args:
        dataset (NpzFile): The dataset loaded from a .npz file.
        set (str): The set to load, can be "train", "val", or "test".
    """

    X = dataset[f"{set}_images"].astype(np.float32) / 255.0
    y = dataset[f"{set}_labels"].astype(np.float32)

    return X, y