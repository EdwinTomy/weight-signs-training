import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from collections import Counter
import pandas as pd

import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Convolution2D, Convolution1D, Convolution3D, Conv1D, Conv2D, Conv3D
from callback_constructor import *
from dataset_constructor import *
from neuralnet_constructor import *
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# x_train, x_test, y_train, y_test = cifar_dataset()
# model = create_model("cifar10_with_conv")
# tracker = SignChangeTracker(model)
# opt = keras.optimizers.SGD(learning_rate=0.01)
# #model.compile(loss="mse", optimizer=opt, metrics=["accuracy"])
# opt = keras.optimizers.Adam(learning_rate=0.01)
# model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# model.fit(x_train, y_train, epochs=10, )#callbacks = [tracker])
# sign_changes = tracker.get_sign_changes()
# print(sign_changes)


# (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# # Normalize pixel values to be between 0 and 1
# x_train, x_test = x_train / 255.0, x_test / 255.0

x_train, x_test, y_train, y_test = mnist_dataset()

# Run the model 100 times
all_runs_history = []
all_sign_changes = []
for run in range(100):
    model = create_model("mnist_no_conv")
    tracker = SignChangeTracker(model)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    #model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=0.01), metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test,y_test), callbacks=tracker)
    sign_changes = tracker.get_sign_changes()

    for i, sign_change in enumerate(sign_changes):
        count = Counter(sign_change.flatten())
        if len(all_sign_changes) == i:
            all_sign_changes.append(count)
        else:
            all_sign_changes[i] += count


combined_df = pd.concat([pd.DataFrame(counter.items(), columns=['Key', 'Value']) for counter in all_sign_changes], ignore_index=True)

# Convert to DataFrame for easier handling (optional)
df_list = []
for idx, history in enumerate(all_runs_history):
    for key in history.keys():
        for epoch, value in enumerate(history[key]):
            df_list.append({'run': idx, 'epoch': epoch, 'metric': key, 'value': value})

df = pd.DataFrame(df_list)
df.to_csv('data/mnist_no_conv.csv', index=False)
combined_df.to_csv('data/mnist_no_conv_signs.csv', index=False)