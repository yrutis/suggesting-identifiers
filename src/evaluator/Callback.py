import keras
import numpy as np


class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.currentPredictions = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.validation_data[0])
        current_dict = {}
        current_dict["X"] = self.validation_data[0]
        current_dict["Y_hat"] = np.argmax(y_pred, axis=1)
        current_dict["Y"] = self.validation_data[1]
        self.currentPredictions.append(current_dict)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return