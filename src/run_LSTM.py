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



def main():
    """ runs model
    """


    def runLSTM():
        logger.info("create LSTM Model...")
        model2 = LSTMModel(context_vocab_size=preprocessor.max_context_vocab_size,
                           length_Y=preprocessor.trainY.shape[1],
                           windows_size=window_size,
                           config=LSTM_config)

        logger.info("create trainer...")
        trainer2 = LSTMTrainer(model=model2.model, data=data, encoder=preprocessor.encoder, config=LSTM_config)

        logger.info("start LSTM training...")
        trainer2.train()


        #generating some predictions...
        df_full = pd.DataFrame(columns=['X', 'Y', 'Predictions'])
        i = 1
        while i < 100:

            #logger.info("make a prediction...")
            #logger.info("prediction for {}" .format(preprocessor.reverse_tokenize(preprocessor.valX[i:i+1])))
            #logger.info("correct Y is {}".format(preprocessor.encoder.inverse_transform(preprocessor.valY[i:i+1])))
            x = preprocessor.reverse_tokenize(preprocessor.valX[i:i+1])
            y = preprocessor.encoder.inverse_transform(preprocessor.valY[i:i+1])
            predictions = trainer2.predict(preprocessor.valX[i:i+1])
            #logger.info(predictions)
            df = {"X": x,
                  "Y": y,
                  "Predictions": [predictions]}

            df_full = df_full.append(df, ignore_index = True)
            i += 1

        logger.info(df_full.head())

        predictions_report = os.path.join(report_folder_LSTM, filename +"-predictions.csv")
        df_full.to_csv(predictions_report)

        logger.info("save evaluation to file")
        evaluator2 = Evaluator(trainer2)
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
    report_folder = path_file.report_folder
    report_folder_LSTM = os.path.join(report_folder, 'reports-'+LSTM_config.name+'-'+str(LSTM_config.data_loader.counter))
    os.mkdir(report_folder_LSTM)

    # write in report folder
    with open(os.path.join(report_folder_LSTM, 'LSTM.json'), 'w') as outfile:
        json.dump(LSTM_config, outfile, indent=4)


    #create decoded version of dataset
    #make_dataset.main(filename, window_size)
    #prepare_data.main(filename, window_size)


    #encode inputs, outputs to make ready for model
    preprocessor = Preprocessor(filename=filename, max_words=10000)
    preprocessor.tokenize()
    data = [preprocessor.trainX, preprocessor.trainY, preprocessor.valX, preprocessor.valY]


    #run model
    runLSTM()
    logger.info("window size is {}".format(window_size))

    LSTM_config.data_loader.counter += 1
    # overwrite
    with open(path_file.LSTM_config_path, 'w') as outfile:
        json.dump(LSTM_config, outfile, indent=4)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

