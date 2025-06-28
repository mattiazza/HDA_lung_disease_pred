import tensorflow as tf
from tensorflow.keras import layers, models

from typing import Tuple


PARAMETER_REGISTRY = {
    "input_shape": (64, 64, 1),  
    "num_classes": 14,           
    "filters": [32, 64, 128],    
    "filter_size": (3, 3),       
    "dropout_rate": 0.3,
    "dense_units": 128,
    "optimizer": "adam",
    "lr": 0.001,
    "loss": "binary_crossentropy",
    "metrics": ["accuracy", "precision", "recall"],
    "model_name": "cnn_baseline"         
}

OPTIMIZER_REGISTRY = {
    "adam": tf.keras.optimizers.Adam
}
METRICS_REGISTRY = {
    "precision": tf.keras.metrics.Precision,
    "recall": tf.keras.metrics.Recall,
    "auc": tf.keras.metrics.AUC,
    "f1_score": tf.keras.metrics.F1Score
}

def cnn_baseline_model(
        input_shape: Tuple[int, int, int] = None,
        num_classes: int = None,
        filters: list = None,
        filter_size: Tuple[int, int] = None,
        dropout_rate: float = None,
        dense_units: int = None,
        optimizer: str = None,
        lr: float = None,
        loss: str = None,
        metrics: list = None,
        model_name: str = None) -> tf.keras.Model:
    """
    Builds a baseline CNN model for multi-label classification.
    """

    # Use PARAMETER_REGISTRY as fallback defaults
    input_shape = input_shape or PARAMETER_REGISTRY["input_shape"]
    num_classes = num_classes or PARAMETER_REGISTRY["num_classes"]
    filters = filters or PARAMETER_REGISTRY["filters"]
    filter_size = filter_size or PARAMETER_REGISTRY["filter_size"]
    dropout_rate = dropout_rate or PARAMETER_REGISTRY["dropout_rate"]
    dense_units = dense_units or PARAMETER_REGISTRY["dense_units"]
    optimizer = optimizer or PARAMETER_REGISTRY["optimizer"]
    lr = lr or PARAMETER_REGISTRY["lr"]
    loss = loss or PARAMETER_REGISTRY["loss"]
    metrics = metrics or PARAMETER_REGISTRY["metrics"]
    model_name = model_name or PARAMETER_REGISTRY["model_name"]


    model = models.Sequential(name=model_name)
    model.add(layers.Input(shape=input_shape)) # Input layer

    for i, f in enumerate(filters):
        model.add(layers.Conv2D(f, filter_size, activation='relu', name=f'conv_{i+1}'))
        model.add(layers.MaxPooling2D(name=f'maxpool_{i+1}'))

    # Flatten the output   
    model.add(layers.Flatten())
    # Add dropout layer to reduce overfitting
    model.add(layers.Dropout(dropout_rate, name='dropout_flatten'))
    # Add dense layer
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dense(num_classes, activation="sigmoid"))

    metrics = [METRICS_REGISTRY[m]() for m in metrics if m in METRICS_REGISTRY]

    # Compile model
    model.compile(
        optimizer=OPTIMIZER_REGISTRY[optimizer](learning_rate=lr),
        loss=loss,  # Multi-label classification
        metrics=metrics   # 'recall'
    )
    print(f"Model {model_name} compiled with:",
          f"• optimizer: {optimizer}",
          f"• loss: {loss}",
          f"• metrics: {metrics}")

    return model