import keras
import numpy as np
import tensorflow as tf

class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.currentPredictions = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.validation_data[0])
        max_lenght = min(len(y_pred)-1, 1000)
        current_dict = {}
        current_dict["X"] = self.validation_data[0]
        current_dict["X"] = current_dict["X"][0:max_lenght]
        current_dict["Y_hat"] = np.argmax(y_pred, axis=1)
        current_dict["Y_hat"] = current_dict["Y_hat"][0:max_lenght]
        current_dict["Y"] = self.validation_data[1]
        current_dict["Y"] = current_dict["Y"][0:max_lenght]
        top_k = tf.nn.top_k(y_pred[:max_lenght], k=5, sorted=True, name=None)
        sess = tf.Session()
        top_k = sess.run(top_k)
        current_dict["top_k"] = top_k
        self.currentPredictions.append(current_dict)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return