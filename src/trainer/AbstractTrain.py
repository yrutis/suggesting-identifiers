import matplotlib.pyplot as plt
import os
import logging
import numpy as np
import src.utils.path as path_file


class AbstractTrain(object):

    def __init__(self, model, data, encoder, config):
        self.model = model
        self.config = config
        self.history = None
        self.type = None
        self.encoder = encoder
        self.trainX = data[0]
        self.trainY = data[1]
        self.valX = data[2]
        self.valY = data[3]


    def train(self):
        raise NotImplementedError

    def predict(self, x):
        if not self.history:
            raise Exception("You have to train the model first before making a prediction")

        # get logger
        logger = logging.getLogger(__name__)


        prediction1 = self.model.predict(x)  # predict for 1 pair
        # sorting the predictions in descending order
        sorting = (-prediction1).argsort()  # sorts by index
        #print("after sorting it is {}".format(sorting))

        #just top5 suggestion
        idx = (-prediction1).argsort()[0][0:4]
        idx2 = idx.tolist()
        pr2 = prediction1[0]
        probs = np.take(pr2, idx2)
        decoded = self.encoder.inverse_transform(idx)


        # getting the top 5 predictions
        sorted_ = sorting[0][:5]
        sorted_ = sorted_.tolist()
        #print("sorted is {}".format(sorted_))
        predicted_label = self.encoder.inverse_transform(sorted_)
        predicted_label = predicted_label.tolist()  # convert numpy to list

        for idx, value in enumerate(sorted_):
            dict = {}
            # just some rounding steps
            prob = (prediction1[0][value]) * 100
            prob = "%.2f" % round(prob, 2)
            logger.info("Number {} prob is {} % for {}".format(idx + 1, prob, predicted_label[idx]))



    def save(self):
        if not self.history:
            raise Exception("You have to train the model first before saving")

        if not self.type:
            raise Exception("You need to assign a type to the model")

        # get logger
        logger = logging.getLogger(__name__)

        model_folder = path_file.model_folder

        model_weights = os.path.join(model_folder, "model-"+self.type+".h5")
        model_serialized = os.path.join(model_folder, "model-"+self.type+".json")

        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model_serialized, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(model_weights)
        logger.info("Saved model to disk")