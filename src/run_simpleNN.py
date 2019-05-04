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
    """ runs model
    """

    def runSimpleNN():

        logger.info('Create the model...')
        model1 = SimpleNN(context_vocab_size=preprocessor.context_vocab_size,
                          length_Y=preprocessor.trainY.shape[1],
                          windows_size=window_size,
                          config=simpleNN_config)

        logger.info("create trainer...")
        trainer1 = SimpleNNTrainer(model=model1.model, data=data, encoder=preprocessor.encoder, config=simpleNN_config)

        logger.info("start training...")
        trainer1.train()

        reversed_seq = preprocessor.reverse_tokenize(preprocessor.valX[1:2])
        logger.info("correct Y is {}".format(preprocessor.encoder.inverse_transform(preprocessor.valY[1:2])))
        logger.info("make a prediction for {}" .format(reversed_seq))


        trainer1.predict(preprocessor.valX[1:2])

        logger.info("save evaluation to file")
        evaluator1 = Evaluator(trainer1)
        evaluator1.visualize()
        evaluator1.evaluate()



    # get logger
    logger = logging.getLogger(__name__)

    simpleNN_config, LSTM_config = config_loader.load_configs()

    filename = simpleNN_config.data_loader.name
    window_size = simpleNN_config.data_loader.window_size

    #create decoded version of dataset
    #prepare_data.main(filename, window_size)

    #encode inputs, outputs to make ready for model
    preprocessor = Preprocessor(filename=filename)
    preprocessor.tokenize()
    data = [preprocessor.trainX, preprocessor.trainY, preprocessor.valX, preprocessor.valY]

    #run model
    runSimpleNN()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

