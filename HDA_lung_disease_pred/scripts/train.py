import os
import json
import time
from typing import Tuple, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from HDA_lung_disease_pred.utils.plot_results import LossOnlyLivePlotCallback



# Default parameters
DEFAULT_BATCH_SIZE=32
DEFAULT_EPOCHS=10
DEFAULT_PATIENCE=10
DEFAULT_USE_CLASS_WEIGHTS=False
DEFAULT_SAVE_HISTORY=True



def create_callbacks(model_name: str, patience: int = 10) -> list:
    """
    Create training callbacks for model optimization
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
        ),
        LossOnlyLivePlotCallback()
    ]
    
    return callbacks



##############################   
#    Train Model Function    #
##############################


def train_model(
    model: tf.keras.Model,

    train_data: tf.data.Dataset,
    val_data: tf.data.Dataset,
    test_data: tf.data.Dataset,

    epochs: int = None,
    batch_size: int = None,

    patience: int = None,
    use_class_weights: bool = None,
    save_history: bool = None
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Train, validate and testing a model on the given dataset
    """

    epochs = epochs or DEFAULT_EPOCHS
    batch_size = batch_size or DEFAULT_BATCH_SIZE
    patience = patience or DEFAULT_PATIENCE
    use_class_weights = use_class_weights or DEFAULT_USE_CLASS_WEIGHTS
    save_history = save_history or DEFAULT_SAVE_HISTORY
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Extract dataset properties
    sample_batch = next(iter(train_data))
    sample_images, _ = sample_batch

    img_size = sample_images.shape[1]  # Height of images


    print(f"Starting training for {model.name} with {img_size}x{img_size} images")
    print(f"Batch size: {batch_size}")
    
    # Create callbacks
    callbacks = create_callbacks(f"{model.name}", patience)

    
    ########## Train model ##########
    
    print(f"\nStarting training for {epochs} epochs...")
    
    start_training_time = time.time()
    history = model.fit(
        train_data,
        validation_data=(val_data),
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

    test_results = model.evaluate(test_data, verbose=0)

    end_test_time = time.time()
    test_time = end_test_time - start_test_time

    test_loss = test_results[0]

    metric_names = [m.name if hasattr(m, 'name') else str(m) for m in model.metrics]

    # Create metrics dictionary dynamically
    test_metrics_dict = {'loss': float(test_loss)}

    # Parse additional metrics
    for i, metric_name in enumerate(metric_names):
        if i + 1 < len(test_results):  # Ensure we don't go out of bounds
            test_metrics_dict[metric_name] = float(test_results[i + 1])

    print(f"Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    for metric_name, value in test_metrics_dict.items():
        if metric_name != 'loss':
            print(f"  {metric_name.title()}: {value:.4f}")
    
    # Save training history
    if save_history:
        history_data = {
            'history': history.history,
            'test_metrics': test_metrics_dict,
            'config': {
                'model_name': model.name,
                'dataset_size': img_size,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': float(model.optimizer.learning_rate.numpy()),
                'use_class_weights': use_class_weights,
                'patience': patience,
                'metrics': metric_names
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




