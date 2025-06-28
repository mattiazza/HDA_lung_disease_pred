import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from typing import Tuple

# Default config registry
PARAMETER_REGISTRY = {
    "input_shape": (64, 64, 1),
    "num_classes": 14,
    "filters": [32, 64, 128],
    "dropout_rate": 0.3,
    "dense_units": 128,
    "optimizer": "adam",
    "lr": 0.001,
    "loss": "binary_crossentropy",
    "metrics": ["accuracy", "precision", "recall"],
    "model_name": "cnn_unet"
}

OPTIMIZER_REGISTRY = {
    "adam": Adam
}

METRICS_REGISTRY = {
    "accuracy": tf.keras.metrics.BinaryAccuracy,
    "precision": tf.keras.metrics.Precision,
    "recall": tf.keras.metrics.Recall,
    "auc": tf.keras.metrics.AUC,
    "f1_score": tf.keras.metrics.AUC  # F1 is not directly supported; AUC is a common substitute for multi-label
}

def cnn_unet_model(
    input_shape: Tuple[int, int, int] = None,
    num_classes: int = None,
    filters: list = None,
    dropout_rate: float = None,
    dense_units: int = None,
    optimizer: str = None,
    lr: float = None,
    loss: str = None,
    metrics: list = None,
    model_name: str = None
) -> tf.keras.Model:
    """
    U-Net inspired CNN: Downsampling encoder with Global Average Pooling and dense classifier.
    Suitable for multi-label classification.
    """
    input_shape = input_shape or PARAMETER_REGISTRY["input_shape"]
    num_classes = num_classes or PARAMETER_REGISTRY["num_classes"]
    filters = filters or PARAMETER_REGISTRY["filters"]
    dropout_rate = dropout_rate or PARAMETER_REGISTRY["dropout_rate"]
    dense_units = dense_units or PARAMETER_REGISTRY["dense_units"]
    optimizer = optimizer or PARAMETER_REGISTRY["optimizer"]
    lr = lr or PARAMETER_REGISTRY["lr"]
    loss = loss or PARAMETER_REGISTRY["loss"]
    metrics = metrics or PARAMETER_REGISTRY["metrics"]
    model_name = model_name or PARAMETER_REGISTRY["model_name"]

    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Downsampling (encoder-like) blocks
    for i, f in enumerate(filters):
        x = layers.Conv2D(f, (3, 3), activation='relu', padding='same', name=f"conv_{i+1}_1")(x)
        x = layers.Conv2D(f, (3, 3), activation='relu', padding='same', name=f"conv_{i+1}_2")(x)
        x = layers.MaxPooling2D(name=f"pool_{i+1}")(x)

    # Global average pooling instead of flattening
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    x = layers.Dropout(dropout_rate, name="dropout")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_1")(x)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    compiled_metrics = [METRICS_REGISTRY[m]() for m in metrics if m in METRICS_REGISTRY]

    model.compile(
        optimizer=OPTIMIZER_REGISTRY[optimizer](learning_rate=lr),
        loss=loss,
        metrics=compiled_metrics
    )

    print(f"Model {model_name} compiled with:",
          f"\n • optimizer: {optimizer} (lr={lr})",
          f"\n • loss: {loss}",
          f"\n • metrics: {[m.name for m in compiled_metrics]}")

    return model
