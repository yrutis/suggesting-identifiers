import os
from keras.utils import plot_model
import logging

class AbstractModel(object):

    def __init__(self):
        self.model = None
        self.type = None

    def build_model(self):
        raise NotImplementedError

    def save_model_architecture(self):
        if not self.model:
            raise Exception("You have to build the model first")

        if not self.type:
            raise Exception("The model needs to have a type")


        # path to model folder
        model_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                    'models')

        # check if model folder exists
        if not os.path.exists(model_folder):
            logging.info("creating models folder...")
            os.mkdir(model_folder)

        # save model architecture to disk
        plotted_model = os.path.join(model_folder, 'model-' + self.type +'.png')
        plot_model(self.model, to_file=plotted_model)
        logging.info("Saved model architecture to disk")