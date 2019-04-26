# -*- coding: utf-8 -*-
import logging
import os

from src.models.SimpleNN import SimpleNN
from src.models.LSTMModel import LSTMModel

from src.trainer.SimpleNNTrainer import SimpleNNTrainer
from src.trainer.LSTMTrainer import LSTMTrainer

from src.evaluator.Evaluator import Evaluator

from src.data.Preprocessor import Preprocessor

def main(filename):
    """ runs model
    """

    def runSimpleNN():
        print('Create the model...')
        model1 = SimpleNN(context_vocab_size=preprocessor.context_vocab_size,
                          length_Y=preprocessor.trainY.shape[1],
                          windows_size=window_size)

        print("create trainer...")
        trainer1 = SimpleNNTrainer(model=model1.model, data=data, encoder=preprocessor.encoder)

        print("start training...")
        trainer1.train()

        print("make a prediction...")
        preprocessor.reverse_tokenize(preprocessor.valX[1:2]) #TODO change method

        trainer1.predict(preprocessor.valX[1:2])

        #TODO build trainX, trainY, valX, valY according to specific model



    def runLSTM():
        print("create LSTM Model...")
        model2 = LSTMModel(context_vocab_size=preprocessor.context_vocab_size,
                           length_Y=preprocessor.trainY.shape[1],
                           windows_size=window_size)

        print("create trainer...")
        trainer2 = LSTMTrainer(model=model2.model, data=data, encoder=preprocessor.encoder)

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

    runSimpleNN()




    
    

   


    


    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    filename = 'bigbluebutton_methoddeclarations_train'
    main(filename)

