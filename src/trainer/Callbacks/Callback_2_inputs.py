import os

import keras
import pandas as pd
import numpy as np


class Histories(keras.callbacks.Callback):
    def __init__(self, report_folder, tokenizer):
        super(Histories, self).__init__()
        self.report_folder = report_folder
        self.tokenizer = tokenizer


    def on_train_begin(self, logs={}):
        self.currentPredictions = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):

        y_pred = self.model.predict([self.validation_data[0], self.validation_data[1]])
        current_dict = {}
        current_dict["method_body"] = self.validation_data[0]
        current_dict["type"] = self.validation_data[1]
        current_dict["Y_hat"] = np.argmax(y_pred, axis=1)
        current_dict["Y"] = self.validation_data[2]
        self.currentPredictions.append(current_dict)


        # Creating a reverse dictionary
        reverse_word_map = dict(map(reversed, self.tokenizer.word_index.items()))

        # Function takes a tokenized sentence and returns the words
        def sequence_to_text(list_of_indices):
            # Looking up words in dictionary
            words = [reverse_word_map.get(letter) for letter in list_of_indices]
            return (words)


        method_body = current_dict['method_body'].tolist()
        type = current_dict['type'].tolist()
        first_y = current_dict['Y'].tolist()
        first_y_hat = current_dict['Y_hat'].tolist()

        first_y_hat = list(map(lambda x: [x], first_y_hat))

        # Creating texts
        method_body_reversed = list(map(sequence_to_text, method_body))
        type_reversed = list(map(sequence_to_text, type))
        first_y_reversed = list(map(sequence_to_text, first_y))
        first_y_hat_reversed = list(map(sequence_to_text, first_y_hat))

        df = pd.DataFrame(
            {"method_body": method_body_reversed,
             "type": type_reversed,
             "Y": first_y_reversed,
             "Y_hat": first_y_hat_reversed})

        prediction_file = os.path.join(self.report_folder, 'myPred_epoch-' + str(epoch) + '.csv')
        df.to_csv(prediction_file)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
