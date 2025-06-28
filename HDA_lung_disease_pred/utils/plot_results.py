import os
from medmnist import INFO
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import NpzFile
from typing import Dict, Any
import seaborn as sns
from IPython.display import clear_output
import tensorflow as tf

from HDA_lung_disease_pred.utils.data_preparation import get_all_labels


DEFAULT_FIGURE_WIDTH = int(os.getenv("FIGURE_WIDTH", "10"))
DEFAULT_FIGURE_HEIGHT = int(os.getenv("FIGURE_HEIGHT", "6"))


def save_plot(plot_name: str, model_name: str = None, dataset_name: str = None, 
              plots_dir: str = "plots", dpi: int = 300, format: str = "png"):
    """
    Save the current plot with a standardized naming convention.
    
    Args:
        plot_name: Name of the plotting function (e.g., 'label_distribution', 'loss')
        model_name: Name of the model (for model-related plots)
        dataset_name: Name of the dataset (for EDA plots)
        plots_dir: Directory to save plots in
        dpi: Resolution for saved image
        format: Image format ('png', 'pdf', 'svg', etc.)
    
    Naming Convention:
        - Model plots: {model_name}_plot_{plot_name}.{format}
        - EDA plots: {dataset_name}_plot_{plot_name}.{format}
    """
    # Create plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    # Determine filename based on naming convention
    if model_name:
        filename = f"{model_name}_plot_{plot_name}.{format}"
    elif dataset_name:
        filename = f"{dataset_name}_plot_{plot_name}.{format}"
    else:
        filename = f"plot_{plot_name}.{format}"
    
    # Full path
    filepath = os.path.join(plots_dir, filename)
    
    # Save the plot
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', format=format)
    print(f"Plot saved: {filepath}")


def plot_label_distribution(dataset: NpzFile, label_names: dict, figsize: tuple = None, 
                          dataset_name: str = "chestmnist", save: bool = False):
    """    
    Plot the distribution of labels in the ChestMNIST datasets.
    
    Args:
        dataset: The dataset containing images and labels
        label_names: Dictionary mapping label indices to label names
        figsize: Figure size as (width, height). Defaults to environment settings or (10, 6)
        dataset_name: Name of the dataset for saving plots
        save: Whether to save the plot
    """
    if figsize is None:
        figsize = (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_HEIGHT)

    all_labels = get_all_labels(dataset)
    n_labels = np.sum(all_labels, axis=0)

    # Plot the distribution of labels in ChestMNIST datasets
    plt.figure(figsize=figsize)
    sns.barplot(x=label_names.values(), y=n_labels, color="mediumseagreen")
    plt.title("Distribution of Labels in ChestMNIST Datasets")
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save:
        save_plot("label_distribution", dataset_name=dataset_name)


def plot_n_label_per_sample(dataset: NpzFile, figsize: tuple = None, 
                           dataset_name: str = "chestmnist", save: bool = False):
    """
    Plot the number of labels per sample in the ChestMNIST datasets.
    
    Args:
        dataset: The dataset containing images and labels
        figsize: Figure size as (width, height). Defaults to environment settings or (10, 6)
        dataset_name: Name of the dataset for saving plots
        save: Whether to save the plot
    """
    if figsize is None:
        figsize = (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_HEIGHT)

    all_labels = get_all_labels(dataset)
    n_labels_per_sample = np.sum(all_labels, axis=1)

    plt.figure(figsize=figsize)
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
    
    if save:
        save_plot("n_label_per_sample", dataset_name=dataset_name)


def plot_image_per_label(dataset: NpzFile, label_names: dict, figsize: tuple = None,
                        dataset_name: str = "chestmnist", save: bool = False):
    """
    Plot one image per label from the dataset.
    
    Args:
        dataset: The dataset containing images and labels
        label_names: A dictionary mapping label indices to label names
        figsize: Figure size as (width, height). Defaults to (15, 10)
        dataset_name: Name of the dataset for saving plots
        save: Whether to save the plot
    """
    if figsize is None:
        figsize = (15, 10)

    plt.figure(figsize=figsize)

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
    
    if save:
        save_plot("image_per_label", dataset_name=dataset_name)
    
    plt.show()


class LossOnlyLivePlotCallback(tf.keras.callbacks.Callback):
    """Simple callback that plots only loss during training"""
    
    def __init__(self, figsize=(10, 6), model_name=None, save_final=False):
        super().__init__()
        self.figsize = figsize
        self.model_name = model_name
        self.save_final = save_final
        self.epoch = 0
        self.history = {
            "loss": [], 
            "val_loss": []
        }
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch += 1
        
        # Update history - only track loss
        self.history["loss"].append(logs.get("loss", 0))
        self.history["val_loss"].append(logs.get("val_loss", 0))
        
        clear_output(wait=True)
        
        # Create single plot for loss only
        plt.figure(figsize=self.figsize)
        
        if self.history["loss"]:
            epochs = range(1, len(self.history["loss"]) + 1)
            plt.plot(epochs, self.history["loss"], 'b-', label='Training Loss', linewidth=2)
            plt.plot(epochs, self.history["val_loss"], 'r-', label='Validation Loss', linewidth=2)
            
        plt.title('Model Loss', fontsize=16)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Print current loss values
        if self.epoch > 0:
            print(f"Epoch {self.epoch}: Loss: {self.history['loss'][-1]:.4f} | Val Loss: {self.history['val_loss'][-1]:.4f}")
    
    def on_train_end(self, logs=None):
        """Save final plot when training ends"""
        if self.save_final and self.model_name:
            # Recreate the plot for saving
            plt.figure(figsize=self.figsize)
            
            if self.history["loss"]:
                epochs = range(1, len(self.history["loss"]) + 1)
                plt.plot(epochs, self.history["loss"], 'b-', label='Training Loss', linewidth=2)
                plt.plot(epochs, self.history["val_loss"], 'r-', label='Validation Loss', linewidth=2)
                
            plt.title('Model Loss', fontsize=16)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            save_plot("loss_curve", model_name=self.model_name)
            plt.close()  # Close the figure to free memory


class MultiLabelLivePlotCallback(tf.keras.callbacks.Callback):
    """Specialized callback for multi-label classification metrics"""
    
    def __init__(self, figsize=(18, 6), model_name=None, save_final=False):
        super().__init__()
        self.figsize = figsize
        self.model_name = model_name
        self.save_final = save_final
        self.epoch = 0
        self.history = {
            "loss": [], "val_loss": [],
            "precision": [], "val_precision": [],
            "recall": [], "val_recall": [],
            "f1_score": [], "val_f1_score": [],
            "auc": [], "val_auc": []
        }
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch += 1
        
        # Update history
        for key in self.history.keys():
            value = logs.get(key, None)
            if value is not None:
                self.history[key].append(value)
        
        clear_output(wait=True)
        
        # Create 3 subplots for multi-label metrics
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        
        # Plot 1: Loss
        self._plot_loss(axes[0])
        
        # Plot 2: Precision & Recall
        self._plot_precision_recall(axes[1])
        
        # Plot 3: F1 & AUC
        self._plot_f1_auc(axes[2])
        
        plt.tight_layout()
        plt.show()
        
        # Print current metrics
        self._print_metrics()

    def _plot_loss(self, ax):
        if self.history["loss"]:
            epochs = range(1, len(self.history["loss"]) + 1)
            ax.plot(epochs, self.history["loss"], 'b-', label='Train Loss', linewidth=2)
            ax.plot(epochs, self.history["val_loss"], 'r-', label='Val Loss', linewidth=2)
            ax.set_title('Model Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_precision_recall(self, ax):
        if self.history["precision"]:
            epochs = range(1, len(self.history["precision"]) + 1)
            ax.plot(epochs, self.history["precision"], 'g-', label='Train Precision', linewidth=2)
            ax.plot(epochs, self.history["val_precision"], 'orange', label='Val Precision', linewidth=2)
            
        if self.history["recall"]:
            epochs = range(1, len(self.history["recall"]) + 1)
            ax.plot(epochs, self.history["recall"], 'purple', label='Train Recall', linewidth=2)
            ax.plot(epochs, self.history["val_recall"], 'brown', label='Val Recall', linewidth=2)
            
        ax.set_title('Precision & Recall')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1.1])
        ax.legend()
        ax.grid(True, alpha=0.3)
    def _plot_f1_auc(self, ax):
        if self.history["f1_score"]:
            epochs = range(1, len(self.history["f1_score"]) + 1)
            ax.plot(epochs, self.history["f1_score"], 'cyan', label='Train F1', linewidth=2)
            ax.plot(epochs, self.history["val_f1_score"], 'magenta', label='Val F1', linewidth=2)
            
        if self.history["auc"]:
            epochs = range(1, len(self.history["auc"]) + 1)
            ax.plot(epochs, self.history["auc"], 'lime', label='Train AUC', linewidth=2)
            ax.plot(epochs, self.history["val_auc"], 'red', label='Val AUC', linewidth=2)
            
        ax.set_title('F1-Score & AUC')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1.1])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _print_metrics(self):
        if self.epoch > 0:
            print(f"\nEpoch {self.epoch} Summary:")
            if self.history["loss"]:
                print(f"  Loss: {self.history['loss'][-1]:.4f} | Val Loss: {self.history['val_loss'][-1]:.4f}")
            if self.history["precision"]:
                print(f"  Precision: {self.history['precision'][-1]:.4f} | Val Precision: {self.history['val_precision'][-1]:.4f}")
            if self.history["recall"]:
                print(f"  Recall: {self.history['recall'][-1]:.4f} | Val Recall: {self.history['val_recall'][-1]:.4f}")
            if self.history["f1_score"]:
                print(f"  F1-Score: {self.history['f1_score'][-1]:.4f} | Val F1-Score: {self.history['val_f1_score'][-1]:.4f}")
            if self.history["auc"]:
                print(f"  AUC: {self.history['auc'][-1]:.4f} | Val AUC: {self.history['val_auc'][-1]:.4f}")
    
    def on_train_end(self, logs=None):
        """Save final plots when training ends"""
        if self.save_final and self.model_name:
            # Create final plots for saving
            fig, axes = plt.subplots(1, 3, figsize=self.figsize)
            
            # Plot 1: Loss
            self._plot_loss(axes[0])
            
            # Plot 2: Precision & Recall
            self._plot_precision_recall(axes[1])
            
            # Plot 3: F1 & AUC
            self._plot_f1_auc(axes[2])
            
            plt.tight_layout()
            
            # Save the complete metrics plot
            save_plot("training_metrics", model_name=self.model_name)
            plt.close()  # Close the figure to free memory


def plot_training_history(history: Dict[str, Any], model_name: str = None, 
                         figsize: tuple = (15, 5), save: bool = False):
    """
    Plot training history from a completed training session.
    
    Args:
        history: Training history dictionary from model.fit()
        model_name: Name of the model for saving plots
        figsize: Figure size as (width, height)
        save: Whether to save the plot
    """
    # Determine available metrics
    available_metrics = [key for key in history.keys() if not key.startswith('val_') and key != 'loss']
    n_plots = min(len(available_metrics) + 1, 4)  # Loss + up to 3 other metrics
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Plot 1: Loss
    axes[0].plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot other metrics
    for i, metric in enumerate(available_metrics[:n_plots-1]):
        ax = axes[i+1]
        val_metric = f'val_{metric}'
        
        ax.plot(epochs, history[metric], 'b-', label=f'Training {metric.title()}', linewidth=2)
        if val_metric in history:
            ax.plot(epochs, history[val_metric], 'r-', label=f'Validation {metric.title()}', linewidth=2)
        
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        if metric != 'loss':
            ax.set_ylim([0, 1.1])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save and model_name:
        save_plot("training_history", model_name=model_name)
    
    plt.show()


def plot_model_comparison(histories: Dict[str, Dict], metric: str = 'val_loss', 
                         figsize: tuple = (12, 6), save: bool = False):
    """
    Compare multiple model training histories.
    
    Args:
        histories: Dictionary of {model_name: history_dict}
        metric: Metric to compare ('val_loss', 'val_accuracy', etc.)
        figsize: Figure size as (width, height)
        save: Whether to save the plot
    """
    plt.figure(figsize=figsize)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, (model_name, history) in enumerate(histories.items()):
        if metric in history:
            epochs = range(1, len(history[metric]) + 1)
            color = colors[i % len(colors)]
            plt.plot(epochs, history[metric], color=color, label=model_name, linewidth=2)
    
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace("_", " ").title())
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save:
        save_plot(f"model_comparison_{metric}")
    
    plt.show()
