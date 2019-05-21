import os
from keras.utils import plot_model
import logging
import src.utils.path as path_file

class AbstractModel(object):

    def __init__(self, config, report_folder):
        self.model = None
        self.type = None
        self.config = config
        self.report_folder = report_folder
        self.input_name = "1-type-" + str(config.data_loader.window_size_params) \
                          + "-params-" + str(config.data_loader.window_size_body) + "-body"

    def build_model(self):
        raise NotImplementedError

    def save_model_architecture(self):
        if not self.model:
            raise Exception("You have to build the model first")

        if not self.type:
            raise Exception("The model needs to have a type")

        logger = logging.getLogger(__name__)


        # save model architecture to disk
        plotted_model = os.path.join(self.report_folder, 'model-' + self.type +'.png')
        plot_model(self.model, to_file=plotted_model)
        logger.info("Saved model architecture to disk")