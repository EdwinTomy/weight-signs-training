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
import time

x_train, x_test, y_train, y_test = mnist_dataset()

start = time.time()
for id_t, threshold in enumerate([0.1, 0.05]):
    for id_m, multiplier in enumerate([2, 5, 10, 20]):
        for id_l, limit in enumerate([2, 3, 4]):

            print("threshold {}, multiplier {}, limit{}".format(threshold, multiplier, limit))

            all_runs_history = []
            for run in range(250):
                model = create_model("mnist_no_conv")
                tracker = SignChanger(model, threshold=threshold, multiplier=multiplier, change_limit=limit)
                model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
                #model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=0.01), metrics=["accuracy"])
                model.fit(x_train, y_train, epochs=10, validation_data=(x_test,y_test), callbacks=tracker)
                all_runs_history.append(model.history.history)


            # Convert to DataFrame for easier handling (optional)
            df_list = []
            for idx, history in enumerate(all_runs_history):
                for key in history.keys():
                    for epoch, value in enumerate(history[key]):
                        df_list.append({'run': idx, 'epoch': epoch, 'metric': key, 'value': value})

            df = pd.DataFrame(df_list)
            df.to_csv('weight-signs-training/data/data_mnist_no_conv/multiplier{}_{}_{}.csv'.format(id_t, id_m, id_l), index=False)

diff = time.time()-start
print(diff)