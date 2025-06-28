
import optuna
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple
import json
import os
from datetime import datetime

from HDA_lung_disease_pred.models.cnn_baseline import cnn_baseline_model
from HDA_lung_disease_pred.utils.data_preparation import prepare_data


def create_callbacks(model_name: str, patience: int = 10) -> list:
    """Create callbacks for model training."""
    callbacks = []
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )
    callbacks.append(early_stopping)
    
    # Reduce learning rate on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=patience//2,
        min_lr=1e-7,
        verbose=0
    )
    callbacks.append(reduce_lr)
    
    return callbacks


def objective(trial: optuna.Trial, 
              train_dataset: tf.data.Dataset,
              val_dataset: tf.data.Dataset,
              input_shape: Tuple,
              num_classes: int = 14,
              epochs: int = 50,
              patience: int = 10) -> float:
    """
    Objective function for Optuna optimization
    """
    
    # Suggest hyperparameters
    hyperparams = {
        # Architecture hyperparameters
        'filters': [
            trial.suggest_int(f'filters_layer_{i}', 16, 256, step=16) 
            for i in range(trial.suggest_int('num_conv_layers', 2, 4))
        ],
        'filter_size': (3, 3),  # Keep fixed for simplicity
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.7, step=0.1),
        'dense_units': trial.suggest_int('dense_units', 64, 512, step=64),
        
        # Training hyperparameters
        'lr': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'optimizer': trial.suggest_categorical('optimizer', ['adam']),  # Can extend later
        
        # Fixed parameters
        'input_shape': input_shape,
        'num_classes': num_classes,
        'loss': 'binary_crossentropy',
        'metrics': ['precision', 'recall', 'auc'],
        'model_name': f'optuna_trial_{trial.number}'
    }
    
    # Create model with suggested hyperparameters
    try:
        model = cnn_baseline_model(**hyperparams)
        
        # Create callbacks
        callbacks = create_callbacks(hyperparams['model_name'], patience=patience)
        
        # Train model
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0
        )
        
        # Get best validation AUC (assuming AUC is in metrics)
        val_auc_history = history.history.get('val_auc', [0])
        best_val_auc = max(val_auc_history) if val_auc_history else 0
        
        # Clean up model to free memory
        del model
        tf.keras.backend.clear_session()
        
        return best_val_auc
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {str(e)}")
        return 0.0  # Return poor score for failed trials


def optimize_hyperparameters(n_trials: int = 50,
                           dataset_name: str = "chestmnist_64",
                           epochs: int = 50,
                           patience: int = 10,
                           study_name: str = None) -> optuna.Study:
    """
    Run hyperparameter optimization using Optuna
    """
    
    print("Starting hyperparameter optimization...")
    print(f"Dataset: {dataset_name}")
    print(f"Trials: {n_trials}")
    print(f"Max epochs per trial: {epochs}")
    
    # Prepare data once
    train_dataset, val_dataset, test_dataset, input_shape = prepare_data(
        dataset_name=dataset_name,
        batch_size=32
    )
    
    # Create study
    if study_name is None:
        study_name = f"cnn_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    study = optuna.create_study(
        direction='maximize',  # Maximize validation AUC
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Define objective with fixed parameters
    def objective_wrapper(trial):
        return objective(
            trial=trial,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            input_shape=input_shape,
            epochs=epochs,
            patience=patience
        )
    
    # Run optimization
    study.optimize(objective_wrapper, n_trials=n_trials, timeout=None)
    
    # Print results
    print("🎯 Optimization completed!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation AUC: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_dir = "optimization_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save study
    study_path = os.path.join(results_dir, f"{study_name}_study.pkl")
    with open(study_path, 'wb') as f:
        import pickle
        pickle.dump(study, f)
    
    # Save best parameters as JSON
    best_params_path = os.path.join(results_dir, f"{study_name}_best_params.json")
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    print(f"📁 Results saved to {results_dir}/")
    
    return study


def train_best_model(study: optuna.Study,
                    dataset_name: str = "chestmnist_64",
                    epochs: int = 100,
                    save_path: str = None) -> tf.keras.Model:
    """
    Train the final model with best hyperparameters.
    
    Args:
        study: Completed Optuna study
        dataset_name: Dataset to use for training
        epochs: Number of epochs for final training
        save_path: Path to save the final model
        
    Returns:
        Trained model
    """
    
    print("🏆 Training best model with optimal hyperparameters...")
    
    # Prepare data
    train_dataset, val_dataset, test_dataset, input_shape = prepare_data(
        dataset_name=dataset_name,
        batch_size=32
    )
    
    # Get best hyperparameters
    best_params = study.best_params.copy()
    
    # Reconstruct filters list
    num_layers = len([k for k in best_params.keys() if k.startswith('filters_layer_')])
    filters = [best_params.pop(f'filters_layer_{i}') for i in range(num_layers)]
    best_params.pop('num_conv_layers', None)  # Remove if exists
    
    # Model parameters
    model_params = {
        'input_shape': input_shape,
        'num_classes': 14,
        'filters': filters,
        'filter_size': (3, 3),
        'dropout_rate': best_params['dropout_rate'],
        'dense_units': best_params['dense_units'],
        'lr': best_params['learning_rate'],
        'optimizer': best_params['optimizer'],
        'loss': 'binary_crossentropy',
        'metrics': ['precision', 'recall', 'auc'],
        'model_name': 'cnn_baseline_optimized'
    }
    
    # Create and train model
    model = cnn_baseline_model(**model_params)
    
    # Create callbacks with model checkpointing
    callbacks = create_callbacks('cnn_baseline_optimized', patience=15)
    
    if save_path:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
    
    # Train model
    print("Training final model...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = model.evaluate(test_dataset, verbose=0)
    
    print("🎊 Final model training completed!")
    print("Test set results:")
    for metric_name, value in zip(model.metrics_names, test_results):
        print(f"  {metric_name}: {value:.4f}")
    
    return model


if __name__ == "__main__":
    # Run optimization
    study = optimize_hyperparameters(
        n_trials=30,  # Adjust based on computational resources
        dataset_name="chestmnist_64",
        epochs=50,
        patience=10
    )
    
    # Train best model
    best_model = train_best_model(
        study=study,
        epochs=100,
        save_path="models/cnn_baseline_optimized.keras"
    )
