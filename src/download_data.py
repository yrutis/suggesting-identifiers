# -*- coding: utf-8 -*-
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


import logging
import os

import src.utils.config as config_loader

import src.data.make_dataset as make_dataset
import src.data.prepare_data as prepare_data


from src.data.Preprocessor import Preprocessor
from src.models.SimpleNN import SimpleNN
from src.models.LSTMModel import LSTMModel

from src.trainer.SimpleNNTrainer import SimpleNNTrainer
from src.trainer.LSTMTrainer import LSTMTrainer

from src.evaluator.Evaluator import Evaluator


def main():
    """ downloads data
    """



    # get logger
    logger = logging.getLogger(__name__)

    simpleNN_config, LSTM_config = config_loader.load_configs()

    filename = simpleNN_config.data_loader.name
    window_size = simpleNN_config.data_loader.window_size

    #create decoded version of dataset
    prepare_data.main(filename, window_size)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

