import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from typing import Tuple
   

def  cnn_baseline_model(input_shape: Tuple[int, int, int], 
                          num_classes: int, 
                          optimizer: tf.keras.optimizers.Optimizer = Adam(learning_rate=0.001),
                          loss: str = 'binary_crossentropy',
                          metrics: list = ['accuracy', 'precision', 'recall'],
                          model_name:str = "cnn_baseline"
                          )-> tf.keras.Model:
    """    
    Builds a baseline CNN model for multi-label classification.
    """

    model = models.Sequential([
        layers.Input(shape=input_shape),

        # First convolutional layers
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        # Second convolutional layers
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        # Third convolutional layers
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="sigmoid")  # Multi-label
    ], name=model_name)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,  # Multi-label classification
        metrics=metrics   # 'recall'
    )

    return model