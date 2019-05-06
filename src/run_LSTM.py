# -*- coding: utf-8 -*-
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import logging

import src.utils.config as config_loader
import src.utils.path as path_file

import src.data.make_dataset as make_dataset
import src.data.prepare_data as prepare_data
from src.data.Preprocessor import Preprocessor
from src.models.LSTMModel import LSTMModel
from src.trainer.LSTMTrainer import LSTMTrainer
from src.evaluator.Evaluator import Evaluator
import tensorflow as tf
import pandas as pd
import os
import json
from datetime import datetime
from random import randint

from src.evaluator.Callback import Histories



def main():
    """ runs model
    """

    def runLSTM():
        logger.info("create LSTM Model...")
        model2 = LSTMModel(context_vocab_size=preprocessor.max_context_vocab_size,
                           length_Y=preprocessor.trainY.shape[1],
                           windows_size=window_size,
                           config=LSTM_config, report_folder=report_folder_LSTM)

        logger.info("getting callback object...")
        histories = Histories()


        logger.info("create trainer...")
        trainer2 = LSTMTrainer(model=model2.model, data=data, encoder=preprocessor.encoder, config=LSTM_config)

        logger.info("start LSTM training...")
        trainer2.train()


        #generating some predictions...
        df_full = pd.DataFrame(columns=['X', 'Y', 'Predictions'])
        i = 1
        while i < len(preprocessor.valX):
            x = preprocessor.reverse_tokenize(preprocessor.valX[i:i+1])
            y = preprocessor.encoder.inverse_transform(preprocessor.valY[i:i+1])
            predictions = trainer2.predict(preprocessor.valX[i:i+1])
            df = {"X": x,
                  "Y": y,
                  "Predictions": [predictions]}

            df_full = df_full.append(df, ignore_index = True)
            i += 1

        logger.info(df_full.head())

        predictions_report = os.path.join(report_folder_LSTM, filename +"-predictions.csv")
        df_full.to_csv(predictions_report)

        logger.info("save evaluation to file")
        evaluator2 = Evaluator(trainer2, report_folder_LSTM)
        evaluator2.visualize()
        evaluator2.evaluate()



    # get logger
    logger = logging.getLogger(__name__)
    simpleNN_config, LSTM_config = config_loader.load_configs()
    filename = LSTM_config.data_loader.name



    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer('window_size', LSTM_config.data_loader.window_size, 'must be between 2 - 8')

    logger.info("window size is {}".format(FLAGS.window_size))

    LSTM_config.data_loader.window_size = FLAGS.window_size
    window_size = FLAGS.window_size
    filename = filename + "-" + str(window_size)



    #create unique report folder
    random_nr = randint(0, 10000)
    unique_folder_key = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S') + "-" + str(random_nr)
    report_folder = path_file.report_folder
    report_folder_LSTM = os.path.join(report_folder, 'reports-'+LSTM_config.name+'-'+unique_folder_key)

    os.mkdir(report_folder_LSTM)

    # write in report folder
    with open(os.path.join(report_folder_LSTM, 'LSTM.json'), 'w') as outfile:
        json.dump(LSTM_config, outfile, indent=4)



    #encode inputs, outputs to make ready for model
    preprocessor = Preprocessor(filename=filename, max_words=LSTM_config.data_loader.max_words)
    preprocessor.tokenize()
    data = [preprocessor.trainX, preprocessor.trainY, preprocessor.valX, preprocessor.valY]


    #run model
    runLSTM()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

