"""
Training utilities for easy model training from notebooks.
"""

from project_c3.scripts.train import train_model


def quick_train(dataset_size=64, epochs=20, batch_size=32):
    """
    Quick training function for notebook use.
    
    Args:
        dataset_size: Size of images (64, 128, or 224)
        epochs: Number of training epochs
        batch_size: Training batch size
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    return train_model(
        model_name=f"cnn_baseline_quick",
        dataset_size=dataset_size,
        epochs=epochs,
        batch_size=batch_size,
        lr=0.001,
        use_class_weights=True,
        patience=5,  # Shorter patience for quick training
        save_history=True
    )


def full_train(dataset_size=64, epochs=50):
    """
    Full training function with optimal settings.
    
    Args:
        dataset_size: Size of images (64, 128, or 224)
        epochs: Number of training epochs
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    return train_model(
        model_name=f"cnn_baseline_full",
        dataset_size=dataset_size,
        epochs=epochs,
        batch_size=32,
        lr=0.001,
        use_class_weights=True,
        patience=10,
        save_history=True
    )
