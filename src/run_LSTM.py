from datetime import datetime
from random import randint

from pickle import dump

import logging

import src.data.prepare_data_new as prepare_data_new
from src.evaluator.Callback import Histories
from src.evaluator.Evaluator import Evaluator
from src.models.LSTMModel import LSTMModel
import src.utils.config as config_loader
import src.utils.path as path_file
import os
import json

from src.trainer.AbstractTrain import AbstractTrain

def main():
    # get logger
    logger = logging.getLogger(__name__)

    LSTM_config_path = path_file.LSTM_config_path
    LSTM_config = config_loader.get_config_from_json(LSTM_config_path)

    trainX, trainY, valX, valY, tokenizer, always_unknown_train, always_unknown_test = \
        prepare_data_new.main(LSTM_config.data_loader.name, LSTM_config.data_loader.window_size)

    word_index = tokenizer.word_index
    print('Found {} unique tokens.'.format(len(word_index) + 1))

    vocab_size = len(word_index) + 1
    histories = Histories()

    #create unique report folder
    random_nr = randint(0, 10000)
    unique_folder_key = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S') + "-" + str(random_nr)
    report_folder = path_file.report_folder
    report_folder_LSTM = os.path.join(report_folder, 'reports-' + LSTM_config.name + '-' + unique_folder_key)

    os.mkdir(report_folder_LSTM)

    # write in report folder
    with open(os.path.join(report_folder_LSTM, 'LSTM.json'), 'w') as outfile:
        json.dump(LSTM_config, outfile, indent=4)

    logger.info("create LSTM Model...")
    model2 = LSTMModel(context_vocab_size=vocab_size,
                       windows_size=8,
                       config=LSTM_config, report_folder=report_folder_LSTM)

    data = [trainX, trainY, valX, valY]

    logger.info("create trainer...")
    trainer2 = AbstractTrain(model=model2.model, data=data,
                             tokenizer=tokenizer, config=LSTM_config,
                             callbacks=histories, report_folder=report_folder_LSTM)

    logger.info("start LSTM training...")
    trainer2.train()
    trainer2.save_callback_predictions()

    logger.info("save evaluation to file")
    evaluator2 = Evaluator(trainer2, report_folder_LSTM)
    evaluator2.visualize(always_unknown_train, always_unknown_test)
    evaluator2.evaluate()

    tokenizer_path = os.path.join(report_folder_LSTM, 'tokenizer.pkl')
    dump(tokenizer, open(tokenizer_path, 'wb'))

if __name__ == '__main__':
    main()