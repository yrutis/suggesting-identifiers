# -*- coding: utf-8 -*-
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import logging

import src.utils.config as config_loader

import src.data.make_dataset as make_dataset
import src.data.prepare_data as prepare_data
from src.data.Preprocessor import Preprocessor
from src.models.LSTMModel import LSTMModel
from src.trainer.LSTMTrainer import LSTMTrainer
from src.evaluator.Evaluator import Evaluator
import tensorflow as tf





def main():
    """ runs model
    """


    def runLSTM():
        logger.info("create LSTM Model...")
        model2 = LSTMModel(context_vocab_size=preprocessor.context_vocab_size,
                           length_Y=preprocessor.trainY.shape[1],
                           windows_size=window_size,
                           config=LSTM_config)

        logger.info("create trainer...")
        trainer2 = LSTMTrainer(model=model2.model, data=data, encoder=preprocessor.encoder, config=LSTM_config)

        logger.info("start LSTM training...")
        trainer2.train()


        logger.info("make a prediction...")
        logger.info("prediction for {}" .format(preprocessor.reverse_tokenize(preprocessor.valX[1:2])))
        logger.info("correct Y is {}".format(preprocessor.encoder.inverse_transform(preprocessor.valY[1:2])))
        trainer2.predict(preprocessor.valX[1:2])

        logger.info("save evaluation to file")
        evaluator2 = Evaluator(trainer2)
        evaluator2.visualize()
        evaluator2.evaluate()



    # get logger
    logger = logging.getLogger(__name__)

    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer('window_size', 3, 'must be between 2 - 8')

    logger.info("window size is {}".format(FLAGS.window_size))

    simpleNN_config, LSTM_config = config_loader.load_configs()
    filename = LSTM_config.data_loader.name
    #window_size = LSTM_config.data_loader.window_size
    window_size = FLAGS.window_size

    filename = filename + '-' + str(window_size)

    #create decoded version of dataset
    #make_dataset.main(filename, window_size)
    #prepare_data.main(filename, window_size)


    #encode inputs, outputs to make ready for model
    preprocessor = Preprocessor(filename=filename)
    preprocessor.tokenize()
    data = [preprocessor.trainX, preprocessor.trainY, preprocessor.valX, preprocessor.valY]


    #run model
    runLSTM()
    logger.info("window size is {}".format(FLAGS.window_size))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

