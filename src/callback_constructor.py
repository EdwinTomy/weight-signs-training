import numpy as np
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.callbacks import Callback

class SignChangeTracker(Callback):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.prev_weights = []
        self.sign_changes = []

        for layer in self.model.layers: 
            if isinstance(layer, (Dense, Conv2D)):
                self.prev_weights.append(layer.get_weights()[0])
                self.sign_changes.append(np.zeros_like(layer.get_weights()[0]))


    def on_epoch_end(self, epoch, logs=None):
        i = 0
        for layer in self.model.layers: 
            if isinstance(layer, (Dense, Conv2D)):
                current_weights = layer.get_weights()[0]
                changes = np.sign(current_weights) != np.sign(self.prev_weights[i])
                self.sign_changes[i] += changes.astype(int)
                self.prev_weights[i] = current_weights
                i+=1

    def get_sign_changes(self):
        return self.sign_changes
    
    def get_sign_configurations(self):
        """Returns the sign configurations of the model's weights."""
        sign_configs = []
        for layer in self.model.layers:
            if isinstance(layer, (Dense, Conv2D)):
                weights = layer.get_weights()[0]  # Only get the weights, not biases
                sign_configs.append(np.sign(weights))
        return sign_configs


class SignChanger(SignChangeTracker):
    def __init__(self, model, threshold=0.1, multiplier=2, change_limit=2):
        super().__init__(model)
        self.threshold = threshold
        self.multiplier = multiplier
        self.change_limit = change_limit
        self.change_limit_tracker = []

        for layer in self.model.layers: 
            if isinstance(layer, (Dense, Conv2D)):
                self.change_limit_tracker.append(np.zeros_like(layer.get_weights()[0]))



    def on_epoch_begin(self, epoch, logs=None):
        if self.prev_weights is None:
            i = 0
            for layer in self.model.layers:
                if isinstance(layer, (Dense, Conv2D)):
                    weights, biases = layer.get_weights()
                    # Multiply weights by -2 where they are below the threshold x
                    mask = (weights < self.threshold) & (weights > -self.threshold)
                    mask_limit = self.change_limit_tracker < self.change_limit
                    mask &= mask_limit
                    weights[mask] *= (-self.multiplier)
                    self.change_limit_tracker[mask] += 1
                    layer.set_weights([weights, biases])
                    self.prev_weights[i] = weights  # Update prev_weig
                    i+=1



        