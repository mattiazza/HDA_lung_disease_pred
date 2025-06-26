import os
import requests
from medmnist import INFO
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import NpzFile
import seaborn as sns



def download_MNIST_dataset(dataset_name: str)-> None:
    """
    Downloads the MNIST dataset images and saves them to the 'data' directory.
    """

    # Create data directory if needed
    os.makedirs("data", exist_ok=True)

    url = f"https://zenodo.org/records/10519652/files/{dataset_name}.npz?download=1"
    save_path = f"data/{dataset_name}.npz"
    
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



def get_percentage():
    """
    Returns a dataframe with the percentage
    """



def plot_label_distribution(dataset: NpzFile, label_names: dict):
    """    
    Plot the distribution of labels in the ChestMNIST datasets.      
    """

    all_labels = get_all_labels(dataset)
    n_labels = np.sum(all_labels, axis=0)

    # Plot the distribution of labels in ChestMNIST datasets
    plt.figure(figsize=(10, 6))
    sns.barplot(x=label_names.values(), y=n_labels, color="mediumseagreen")
    plt.title("Distribution of Labels in ChestMNIST Datasets")
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()



def plot_n_label_per_sample(dataset: NpzFile):
    """
    Plot the number of labels per sample in the ChestMNIST datasets.
    """
    all_labels = get_all_labels(dataset)

    n_labels_per_sample = np.sum(all_labels, axis=1)

    plt.figure(figsize=(10, 6))
    ax = sns.histplot(n_labels_per_sample, bins=np.arange(0, 5) - 0.5, kde=False, color="mediumseagreen")

    # Get the current axis
    ax = plt.gca()

    # Loop through the patches (bars)
    for i, p in enumerate(ax.patches):
        # Calculate the center position of each bar
        x_position = p.get_x() + p.get_width() / 2
        y_position = p.get_height() / 2
        
        # Height of the bar
        height = p.get_height()

        # Add the label (0, 1, 2, 3) at the center
        ax.text(
            x_position,      # x position (center of bar)
            y_position,      # y position (middle of bar)
            f'{height}',          # text (just the number 0, 1, 2, 3)
            ha='center',     # horizontal alignment
            va='center',     # vertical alignment
            fontsize=14,     # larger font size
            color='white',   # white text for visibility
            fontweight='bold'
        )
    plt.xticks(ticks=[0, 1, 2, 3], labels=['0', '1', '2', '3'])
    plt.title("Number of Labels per Sample in ChestMNIST Datasets")
    plt.xlabel("Number of Labels")
    plt.ylabel("Count")


def plot_image_per_label(dataset: NpzFile, label_names: dict):
    """
    Plot one image per label from the dataset.
    
    Args:
        dataset: The dataset containing images and labels.
        label_names: A dictionary mapping label indices to label names.
    """

    plt.figure(figsize=(15, 10))

    for i in label_names.keys():
        target_label = [1 if i == n else 0 for n in range(len(label_names.values()))]

        # Get the indices where the condition is True
        img_index = np.where(np.all(dataset['train_labels'] == target_label, axis=1))[0]

        plt.subplot(3, 5, i + 1)
        img = dataset['train_images'][img_index[0]]
        plt.imshow(img / 255.0, cmap='gray')
        plt.title(label_names[i])
        plt.axis('off')

        # Plot healthy lung image
        if i == len(label_names.keys()) - 1:
            plt.subplot(3, 5, 15)

            target_label = np.zeros(len(label_names.values()), dtype=int)

            img_index = np.where(np.all(dataset['train_labels'] == target_label, axis=1))[0]

            img = dataset['train_images'][img_index[0]]
            plt.imshow(img / 255.0, cmap='gray')
            plt.title('healthy')
            plt.axis('off')
    plt.show()