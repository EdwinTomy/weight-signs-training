from __future__ import print_function
import tensorflow.keras.layers as layers
import keras

def create_model(model_type):
    num_classes = 10
    if model_type == "cifar10_with_conv":
        return keras.Sequential(
            [
                layers.Input(shape=(32, 32, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
        

    if model_type == "cifar10_no_conv":
        return keras.Sequential(
            [
                layers.Input(shape=(32, 32, 3)),
                layers.Flatten(),
                layers.Dense(256, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

    if model_type == "mnist_with_conv":
        return keras.Sequential(
            [
                layers.Input(shape=(28, 28, 1)),
                layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(4, activation="relu"),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )


    if model_type == "mnist_no_conv":
        return keras.Sequential(
            [
                layers.Input(shape=(28, 28, 1)),
                layers.Flatten(),
                layers.Dense(8, activation="relu"),
                layers.Dense(8, activation="relu"),
                layers.Dense(8, activation="relu"),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    
    if model_type == "xor":
        return keras.Sequential(
            [
                layers.Input(shape=(2, 1)),
                layers.Flatten(),
                layers.Dense(4, activation="relu"),
                layers.Dense(4, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

    


