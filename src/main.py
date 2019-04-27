# -*- coding: utf-8 -*-
import logging
import os

import src.utils.config as config_loader

import src.data.make_dataset as make_dataset
from src.data.Preprocessor import Preprocessor
from src.models.SimpleNN import SimpleNN
from src.models.LSTMModel import LSTMModel

from src.trainer.SimpleNNTrainer import SimpleNNTrainer
from src.trainer.LSTMTrainer import LSTMTrainer

from src.evaluator.Evaluator import Evaluator


def main():
    """ runs model
    """

    # TODO make logging work


    def runSimpleNN():

        print('Create the model...')
        model1 = SimpleNN(context_vocab_size=preprocessor.context_vocab_size,
                          length_Y=preprocessor.trainY.shape[1],
                          windows_size=window_size,
                          config=simpleNN_config)

        print("create trainer...")
        trainer1 = SimpleNNTrainer(model=model1.model, data=data, encoder=preprocessor.encoder, config=simpleNN_config)

        print("start training...")
        trainer1.train()

        reversed_seq = preprocessor.reverse_tokenize(preprocessor.valX[1:2])
        print("make a prediction for {}" .format(reversed_seq))
        trainer1.predict(preprocessor.valX[1:2])

        print("save evaluation to file")
        evaluator1 = Evaluator(trainer1)
        evaluator1.visualize()
        evaluator1.evaluate()


    def runLSTM():
        print("create LSTM Model...")
        model2 = LSTMModel(context_vocab_size=preprocessor.context_vocab_size,
                           length_Y=preprocessor.trainY.shape[1],
                           windows_size=window_size,
                           config=LSTM_config)

        print("create trainer...")
        trainer2 = LSTMTrainer(model=model2.model, data=data, encoder=preprocessor.encoder, config=LSTM_config)

        print("start LSTM training...")
        trainer2.train()

        print("make a prediction...")
        trainer2.predict(preprocessor.valX[1:2])

        print("save evaluation to file")
        evaluator2 = Evaluator(trainer2)
        evaluator2.visualize()
        evaluator2.evaluate()



    # get logger
    logger = logging.getLogger(__name__)

    simpleNN_config, LSTM_config = config_loader.load_configs()
    filename = LSTM_config.data_loader.name
    print(filename)
    window_size = 8

    #create decoded version of dataset
    make_dataset.main(filename, window_size)

    #encode inputs, outputs to make ready for model
    preprocessor = Preprocessor(filename=filename)
    preprocessor.tokenize()
    data = [preprocessor.trainX, preprocessor.trainY, preprocessor.valX, preprocessor.valY]

    #run model
    runLSTM()
    runSimpleNN()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

