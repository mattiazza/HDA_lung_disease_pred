"""
Training script for ChestMNIST classification models.

This module provides functionality to train CNN models on the ChestMNIST dataset
with configurable parameters and automatic model saving.
"""

import os
import argparse
import json
import time
from datetime import datetime
from typing import Tuple, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

from HDA_lung_disease_pred.utils.eda_utils import download_MNIST_dataset, get_label_names
from HDA_lung_disease_pred.utils.data_preparation import load_set
from HDA_lung_disease_pred.models.cnn_baseline import cnn_baseline_model


def calculate_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced multi-label classification.
    
    Args:
        y_train: Training labels array of shape (n_samples, n_classes)
        
    Returns:
        Dictionary mapping class indices to their weights
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


def create_callbacks(model_name: str, patience: int = 10) -> list:
    """
    Create training callbacks for model optimization.
    
    Args:
        model_name: Name for saving the best model
        patience: Patience for early stopping
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=f'models/{model_name}_best.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks


# def train_model_1(
#     model_name: str = "cnn_baseline",
#     dataset_size: int = 64,
#     epochs: int = 50,
#     batch_size: int = 32,
#     learning_rate: float = 0.001,
#     use_class_weights: bool = True,
#     patience: int = 10,
#     save_history: bool = True
# ) -> Tuple[tf.keras.Model, Dict[str, Any]]:
#     """
#     Train a CNN model on ChestMNIST dataset.
    
#     Args:
#         model_name: Name of the model for saving
#         dataset_size: Size of images (64, 128, or 224)
#         epochs: Number of training epochs
#         batch_size: Training batch size
#         learning_rate: Learning rate for optimizer
#         use_class_weights: Whether to use class weights for imbalanced data
#         patience: Early stopping patience
#         save_history: Whether to save training history
        
#     Returns:
#         Tuple of (trained_model, training_history)
#     """
#     print(f"Starting training for {model_name} with {dataset_size}x{dataset_size} images")
    
#     # Create directories
#     os.makedirs('models', exist_ok=True)
#     os.makedirs('logs', exist_ok=True)
    
#     # Load dataset
#     dataset_name = f"chestmnist_{dataset_size}"
#     print(f"Loading {dataset_name} dataset...")
#     chest_dataset = download_MNIST_dataset(dataset_name)
    
#     # Prepare data
#     X_train, y_train = load_set(chest_dataset, "train")
#     X_val, y_val = load_set(chest_dataset, "val")
#     X_test, y_test = load_set(chest_dataset, "test")
    
#     print(f"Data shapes:")
#     print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
#     print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
#     print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
#     # Get label information
#     label_names = get_label_names("chestmnist")
#     num_classes = len(label_names)
    
#     # Create model
#     input_shape = X_train.shape[1:]
#     model = cnn_baseline_model(input_shape, num_classes)
    
#     # Compile model
#     model.compile(
#         optimizer=Adam(learning_rate=learning_rate),
#         loss='binary_crossentropy',  # Multi-label classification
#         metrics=['accuracy', 'precision', 'recall']
#     )
    
#     print("\nModel architecture:")
#     model.summary()
    
#     # Calculate class weights if requested
#     class_weight = None
#     if use_class_weights:
#         class_weights = calculate_class_weights(y_train)
#         print(f"\nUsing class weights: {class_weights}")
#         # For multi-label, we need to handle this differently in training
    
#     # Create callbacks
#     callbacks = create_callbacks(f"{model_name}_{dataset_size}", patience)

#     # Train model
#     print(f"\nStarting training for {epochs} epochs...")
#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=epochs,
#         batch_size=batch_size,
#         callbacks=callbacks,
#         verbose=1
#     )
    
#     # Evaluate on test set
#     print("\nEvaluating on test set...")
#     test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
#         X_test, y_test, verbose=0
#     )
    
#     print(f"Test Results:")
#     print(f"  Loss: {test_loss:.4f}")
#     print(f"  Accuracy: {test_accuracy:.4f}")
#     print(f"  Precision: {test_precision:.4f}")
#     print(f"  Recall: {test_recall:.4f}")
    
#     # Save training history
#     if save_history:
#         history_data = {
#             'history': history.history,
#             'test_metrics': {
#                 'loss': float(test_loss),
#                 'accuracy': float(test_accuracy),
#                 'precision': float(test_precision),
#                 'recall': float(test_recall)
#             },
#             'config': {
#                 'model_name': model_name,
#                 'dataset_size': dataset_size,
#                 'epochs': epochs,
#                 'batch_size': batch_size,
#                 'learning_rate': learning_rate,
#                 'use_class_weights': use_class_weights,
#                 'patience': patience
#             },
#             'timestamp': datetime.now().isoformat()
#         }
        
#         history_filename = f"logs/{model_name}_{dataset_size}_history.json"
#         with open(history_filename, 'w') as f:
#             json.dump(history_data, f, indent=2)
#         print(f"Training history saved to {history_filename}")
    
#     # Save final model
#     final_model_path = f"models/{model_name}_{dataset_size}_final.keras"
#     model.save(final_model_path)
#     print(f"Final model saved to {final_model_path}")
    
#     return model, history.history


##############################   
#    Train Model Function    #
##############################


def train_model(
    model: tf.keras.Model,

    train_data: tuple,
    val_data: tuple,
    test_data: tuple,

    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,

    use_class_weights: bool = True,
    patience: int = 10,
    save_history: bool = True
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Train a CNN model on ChestMNIST dataset.
    
    Args:
        model: Keras model instance to train
        train_data: Tuple of (X_train, y_train)
        val_data: Tuple of (X_val, y_val)
        test_data: Tuple of (X_test, y_test)
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        use_class_weights: Whether to use class weights for imbalanced data
        patience: Early stopping patience
        save_history: Whether to save training history
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    
    # Extract training, validation, and test data
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data


    dataset_size = X_train.shape[1]  

    print(f"Starting training for {model.name} with {dataset_size}x{dataset_size} images")
    

    # Compile model if not compiled yet
    if not model.compiled:
        print("\nCompiling model...")
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='binary_crossentropy',  # Multi-label classification
            metrics=['accuracy', 'precision', 'recall']
        )
        print(f"used optimizer: {model.optimizer.name} with learning rate {lr}\n",
              "used loss function: {model.loss}\n",
              "used metrics: {model.metrics_names}")
    
    
    # Calculate class weights if requested
    class_weights = None
    if use_class_weights:
        class_weights = calculate_sample_weights(y_train)
        print(f"\nUsing class weights")
    
    # Create callbacks
    callbacks = create_callbacks(f"{model.name}", patience)
    

    
    ########## Train model ##########
    
    print(f"\nStarting training for {epochs} epochs...")
    
    start_training_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        sample_weight=class_weights,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    end_training_time = time.time()
    train_time = end_training_time - start_training_time
    

    ########## Evaluate on test set ##########
    
    print("\nEvaluating on test set...")

    start_test_time = time.time()
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
        X_test, y_test, verbose=0
    )
    end_test_time = time.time()
    test_time = end_test_time - start_test_time

    print(f"Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    
    # Save training history
    if save_history:
        history_data = {
            'history': history.history,
            'test_metrics': {
                'loss': float(test_loss),
                'accuracy': float(test_accuracy),
                'precision': float(test_precision),
                'recall': float(test_recall)
            },
            'config': {
                'model_name': model.name,
                'dataset_size': dataset_size,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': lr,
                'use_class_weights': use_class_weights,
                'patience': patience
            },
            'testing_time': test_time,
            'training_time': train_time
        }
        
        history_filename = f"logs/{model.name}_history.json"
        with open(history_filename, 'w') as f:
            json.dump(history_data, f, indent=2)
        print(f"Training history saved to {history_filename}")
    
    # Save final model
    final_model_path = f"models/{model.name}_final.keras"
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return model, history.history
