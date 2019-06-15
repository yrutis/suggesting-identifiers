import logging

import keras
import numpy as np
import tensorflow as tf
import os
import pandas as pd

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
        '''

        :param epoch: epoch
        :param logs: logs
        :return: make 1000 predictions from validation data every 5 epochs
        '''

        logger = logging.getLogger(__name__)
        if epoch % 5 == 0:
            max_lenght = min(len(self.validation_data[0]) - 1, 1000)
            logger.info("for")
            y_pred = self.model.predict(self.validation_data[0][0:max_lenght])

            current_dict = {}
            current_dict["X"] = self.validation_data[0][0:max_lenght]
            current_dict["Y_hat"] = np.argmax(y_pred, axis=1)
            current_dict["Y"] = self.validation_data[1][0:max_lenght]

            #get top k predictions
            #top_k = tf.nn.top_k(y_pred[:max_lenght], k=5, sorted=True, name=None)
            #sess = tf.Session()
            #top_k = sess.run(top_k)
            #current_dict["top_k"] = top_k
            self.currentPredictions.append(current_dict)


            # Creating a reverse dictionary
            reverse_word_map = dict(map(reversed, self.tokenizer.word_index.items()))

            # Function takes a tokenized sentence and returns the words
            def sequence_to_text(list_of_indices):
                # Looking up words in dictionary
                words = [reverse_word_map.get(letter) for letter in list_of_indices]
                return (words)


            first_x = current_dict['X'].tolist()
            first_y = current_dict['Y'].tolist()
            first_y_hat = current_dict['Y_hat'].tolist()
            #first_k_y_hat = current_dict['top_k'].indices.tolist()
            #self_k_y_probs = current_dict['top_k'].values.tolist()

            first_y_hat = list(map(lambda x: [x], first_y_hat))
            # Creating texts
            first_x_reversed = list(map(sequence_to_text, first_x))
            first_y_reversed = list(map(sequence_to_text, first_y))
            first_y_hat_reversed = list(map(sequence_to_text, first_y_hat))
            #first_k_y_hat_reversed = list(map(sequence_to_text, first_k_y_hat))

            df = pd.DataFrame(
                {"X": first_x_reversed,
                 "Y": first_y_reversed,
                 "Y_hat": first_y_hat_reversed
                 #"top_k": first_k_y_hat_reversed,
                 #"top_k_probs": self_k_y_probs
                 })

            prediction_file = os.path.join(self.report_folder, 'myPred_epoch-' + str(epoch) + '.csv')
            df.to_csv(prediction_file)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return