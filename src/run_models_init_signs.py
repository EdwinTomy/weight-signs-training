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


x_train, x_test, y_train, y_test = mnist_dataset()


# Run the model 100 times
all_runs_history = []
all_sign_history = []
all_sign_changes = []
for run in range(100):
    model = create_model("mnist_no_conv")
    tracker = SignChangeTracker(model)
    sign_configs = tracker.get_sign_configurations()
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    #model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=0.01), metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test,y_test), callbacks=tracker)
    all_runs_history.append(model.history.history)
    sign_changes = tracker.get_sign_changes()

    for i, sign_change in enumerate(sign_changes):
        count = Counter(sign_change.flatten())
        if len(all_sign_changes) == i:
            all_sign_changes.append(count)
        else:
            all_sign_changes[i] += count

    model_s = create_model("mnist_no_conv")

    in_layer = 0
    for layer in model_s.layers:
        if isinstance(layer, (Dense, Conv2D)):
            weights, biases = layer.get_weights()
            # Preserve the magnitude from the default initialization, adjust only the signs
            #print(weights.shape, )
            adjusted_weights = np.abs(weights) * sign_configs[in_layer]
            layer.set_weights([adjusted_weights, biases])
            in_layer +=1


    model_s.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    #model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=0.01), metrics=["accuracy"])
    model_s.fit(x_train, y_train, epochs=10, validation_data=(x_test,y_test))
    all_sign_history.append(model_s.history.history)


combined_df = pd.concat([pd.DataFrame(counter.items(), columns=['Key', 'Value']) for counter in all_sign_changes], ignore_index=True)


# Convert to DataFrame for easier handling (optional)
df_list = []
for idx, history in enumerate(all_runs_history):
    for key in history.keys():
        for epoch, value in enumerate(history[key]):
            df_list.append({'run': idx, 'epoch': epoch, 'metric': key, 'value': value})

df = pd.DataFrame(df_list)
df.to_csv('data/mnist_no_conv.csv', index=False)


dfs_list = []
for idx, history in enumerate(all_sign_history):
    for key in history.keys():
        for epoch, value in enumerate(history[key]):
            dfs_list.append({'run': idx, 'epoch': epoch, 'metric': key, 'value': value})

df = pd.DataFrame(dfs_list)
df.to_csv('data/mnist_no_conv_init.csv', index=False)
combined_df.to_csv('data/mnist_no_conv_signs.csv', index=False)