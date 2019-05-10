import matplotlib.pyplot as plt
import os
import logging
import numpy as np
import src.utils.path as path_file
import pandas as pd


class AbstractTrain(object):

    def __init__(self, model, data, tokenizer, config, callbacks, report_folder):
        self.model = model
        self.config = config
        self.history = None
        self.type = None
        self.tokenizer = tokenizer
        self.trainX = data[0]
        self.trainY = data[1]
        self.valX = data[2]
        self.valY = data[3]
        self.callbacks = callbacks
        self.report_folder = report_folder


    def train(self):
        self.history = self.model.fit(self.trainX, self.trainY,
                            validation_data=[self.valX, self.valY],
                            batch_size=self.config.trainer.batch_size,
                            epochs=self.config.trainer.num_epochs,
                            callbacks=[self.callbacks])

        self.save()

    def save_callback_predictions(self):
        # Creating a reverse dictionary
        reverse_word_map = dict(map(reversed, self.tokenizer.word_index.items()))

        # Function takes a tokenized sentence and returns the words
        def sequence_to_text(list_of_indices):
            # Looking up words in dictionary
            words = [reverse_word_map.get(letter) for letter in list_of_indices]
            return (words)

        i = 0
        while i < len(self.callbacks.currentPredictions):
            # every 5 epochs thats why i + 1
            if (i + 1) % 5 == 0:

                first_x = self.callbacks.currentPredictions[i]['X'].tolist()
                first_y = self.callbacks.currentPredictions[i]['Y'].tolist()
                first_y_hat = self.callbacks.currentPredictions[i]['Y_hat'].tolist()
                first_k_y_hat = self.callbacks.currentPredictions[i]['top_k'].indices.tolist()
                self_k_y_probs = self.callbacks.currentPredictions[i]['top_k'].values.tolist()


                first_y_hat = list(map(lambda x: [x], first_y_hat))
                # Creating texts
                first_x_reversed = list(map(sequence_to_text, first_x))
                first_y_reversed = list(map(sequence_to_text, first_y))
                first_y_hat_reversed = list(map(sequence_to_text, first_y_hat))
                first_k_y_hat_reversed = list(map(sequence_to_text, first_k_y_hat))

                df = pd.DataFrame(
                    {"X": first_x_reversed,
                     "Y": first_y_reversed,
                     "Y_hat": first_y_hat_reversed,
                     "top_k": first_k_y_hat_reversed,
                     "top_k_probs": self_k_y_probs})

                prediction_file = os.path.join(self.report_folder, 'myPred_epoch-' + str(i+1) + '.csv')
                df.to_csv(prediction_file)
            i += 1

    def save(self):
        if not self.history:
            raise Exception("You have to train the model first before saving")


        # get logger
        logger = logging.getLogger(__name__)

        model_weights = os.path.join(self.report_folder, "model.h5")
        model_serialized = os.path.join(self.report_folder, "model.json")

        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model_serialized, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(model_weights)
        logger.info("Saved model to disk")


'''
    def predict(self, x):
        if not self.history:
            raise Exception("You have to train the model first before making a prediction")
    
        # get logger
        logger = logging.getLogger(__name__)
    
    
        prediction1 = self.model.predict(x)  # predict for 1 pair
        # sorting the predictions in descending order
        sorting = (-prediction1).argsort()  # sorts by index
        #print("after sorting it is {}".format(sorting))
    
    
        # getting the top 5 predictions
        sorted_ = sorting[0][:5]
        sorted_ = sorted_.tolist()
        #print("sorted is {}".format(sorted_))
        predicted_label = self.encoder.inverse_transform(sorted_)
        predicted_label = predicted_label.tolist()  # convert numpy to list
    
        predictions = []
    
        for idx, value in enumerate(sorted_):
            current_dict = {}
            # just some rounding steps
            prob = (prediction1[0][value]) * 100
            prob = "%.2f" % round(prob, 2)
            current_dict['prob'] = prob
            current_dict['name'] = predicted_label[idx]
            predictions.append(current_dict)
            #logger.info("Number {} prob is {} % for {}".format(idx + 1, prob, predicted_label[idx]))
        return predictions

'''

