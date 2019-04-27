# -*- coding: utf-8 -*-
import logging
import os

import src.utils.config as config_loader
from src.data.Preprocessor import Preprocessor
from src.models.SimpleNN import SimpleNN
from src.models.LSTMModel import LSTMModel

from src.trainer.SimpleNNTrainer import SimpleNNTrainer
from src.trainer.LSTMTrainer import LSTMTrainer

from src.evaluator.Evaluator import Evaluator


def main(filename):
    """ runs model
    """


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

        #TODO save models weights, json in specific place
        #ToDO save accs loss in specific place
        #TODO save sklearn report in specific place

        #TODO make logging work



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

    window_size = 8
    preprocessor = Preprocessor(filename=filename)
    preprocessor.tokenize()
    preprocessor.reverse_tokenize(preprocessor.valX[1:2])

    data = [preprocessor.trainX, preprocessor.trainY, preprocessor.valX, preprocessor.valY]

    simpleNN_config, LSTM_config = config_loader.load_configs()
    runLSTM()
    #runSimpleNN()





if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    filename = 'bigbluebutton_methoddeclarations_train'
    main(filename)

