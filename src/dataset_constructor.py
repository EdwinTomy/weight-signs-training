from tensorflow.keras import datasets
import keras
import numpy as np

def mnist_dataset():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, x_test, y_train, y_test

def cifar_dataset():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, x_test, y_train, y_test

def xor_dataset():
    x_train = np.random.uniform(-1, 1, size=(1000, 2))
    y_train = np.logical_xor(x_train[:, 0] > 0, x_train[:, 1] > 0).astype(int)

    x_test = np.random.uniform(-1, 1, size=(1000, 2))
    y_test = np.logical_xor(x_test[:, 0] > 0, x_test[:, 1] > 0).astype(int)
    
    return x_train, x_test, y_train, y_test

