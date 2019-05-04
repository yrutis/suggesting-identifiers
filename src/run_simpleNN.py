# -*- coding: utf-8 -*-
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


import logging
import os

import src.utils.config as config_loader
import src.utils.path as path_file
import json

import src.data.make_dataset as make_dataset
import src.data.prepare_data as prepare_data


from src.data.Preprocessor import Preprocessor
from src.models.SimpleNN import SimpleNN
from src.models.LSTMModel import LSTMModel

from src.trainer.SimpleNNTrainer import SimpleNNTrainer
from src.trainer.LSTMTrainer import LSTMTrainer

from src.evaluator.Evaluator import Evaluator
import tensorflow as tf
import pandas as pd

def main():
    """ runs model
    """

    def runSimpleNN():

        logger.info('Create the model...')
        model1 = SimpleNN(context_vocab_size=preprocessor.max_context_vocab_size,
                          length_Y=preprocessor.trainY.shape[1],
                          windows_size=window_size,
                          config=simpleNN_config)

        logger.info("create trainer...")
        trainer1 = SimpleNNTrainer(model=model1.model, data=data, encoder=preprocessor.encoder, config=simpleNN_config)

        logger.info("start training...")
        trainer1.train()



        # generating some predictions...
        df_full = pd.DataFrame(columns=['X', 'Y', 'Predictions'])
        i = 1
        while i < 100:
            # logger.info("make a prediction...")
            # logger.info("prediction for {}" .format(preprocessor.reverse_tokenize(preprocessor.valX[i:i+1])))
            # logger.info("correct Y is {}".format(preprocessor.encoder.inverse_transform(preprocessor.valY[i:i+1])))
            x = preprocessor.reverse_tokenize(preprocessor.valX[i:i + 1])
            y = preprocessor.encoder.inverse_transform(preprocessor.valY[i:i + 1])
            predictions = trainer1.predict(preprocessor.valX[i:i + 1])
            # logger.info(predictions)
            df = {"X": x,
                  "Y": y,
                  "Predictions": [predictions]}

            df_full = df_full.append(df, ignore_index=True)
            i += 1

        logger.info(df_full.head())

        predictions_report = os.path.join(report_folder_simpleNN, filename + "-predictions.csv")
        df_full.to_csv(predictions_report)

        logger.info("save evaluation to file")
        evaluator1 = Evaluator(trainer1)
        evaluator1.visualize()
        evaluator1.evaluate()



    # get logger
    logger = logging.getLogger(__name__)

    simpleNN_config, LSTM_config = config_loader.load_configs()

    filename = simpleNN_config.data_loader.name


    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer('window_size', simpleNN_config.data_loader.window_size, 'must be between 2 - 8')

    simpleNN_config.data_loader.window_size = FLAGS.window_size
    window_size = FLAGS.window_size

    filename = filename + '-' + str(window_size)

    # create unique report folder
    report_folder = path_file.report_folder
    report_folder_simpleNN = os.path.join(report_folder,
                                      'reports-' + simpleNN_config.name + '-' + str(simpleNN_config.data_loader.counter))
    os.mkdir(report_folder_simpleNN)

    # write in report folder
    with open(os.path.join(report_folder_simpleNN, 'simpleNN.json'), 'w') as outfile:
        json.dump(simpleNN_config, outfile, indent=4)

    #create decoded version of dataset
    #prepare_data.main(filename, window_size)

    #encode inputs, outputs to make ready for model
    preprocessor = Preprocessor(filename=filename, max_words=10000)
    preprocessor.tokenize()
    data = [preprocessor.trainX, preprocessor.trainY, preprocessor.valX, preprocessor.valY]

    #run model
    runSimpleNN()

    simpleNN_config.data_loader.counter += 1
    # overwrite
    with open(path_file.simpleNN_config_path, 'w') as outfile:
        json.dump(simpleNN_config, outfile, indent=4)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

