from datetime import datetime
from random import randint
from pickle import dump
import logging
import os
import json
import tensorflow as tf


import src.data.prepare_data_new as prepare_data_new
from src.evaluator.Callback import Histories
from src.evaluator.Evaluator import Evaluator
from src.models.LSTMBidModel import LSTMModelBid
import src.utils.config as config_loader
import src.utils.path as path_file

from src.trainer.AbstractTrain import AbstractTrain


def main():
    # get logger
    logger = logging.getLogger(__name__)
    LSTMBid_config_path = path_file.LSTMBid_config_path
    LSTMBid_config = config_loader.get_config_from_json(LSTMBid_config_path)

    FLAGS = tf.app.flags.FLAGS

    #define some tf flags
    tf.app.flags.DEFINE_integer('window_size', LSTMBid_config.data_loader.window_size, 'must be between 2+')
    tf.app.flags.DEFINE_string('data', LSTMBid_config.data_loader.name, 'must be either Android-Universal-Image-Loader or all_methods_train')
    tf.app.flags.DEFINE_integer('epochs', LSTMBid_config.trainer.num_epochs, 'must be between 1-100')
    tf.app.flags.DEFINE_integer('batch_size', LSTMBid_config.trainer.batch_size, 'must be a power of 2 2^1 - 2^6')


    LSTMBid_config.data_loader.window_size = FLAGS.window_size
    logger.info("window size is {}".format(LSTMBid_config.data_loader.window_size))

    LSTMBid_config.data_loader.name = FLAGS.data
    logger.info("data used is {}".format(LSTMBid_config.data_loader.name))

    LSTMBid_config.trainer.num_epochs = FLAGS.epochs
    logger.info("epochs num is {}".format(LSTMBid_config.trainer.num_epochs))

    LSTMBid_config.trainer.batch_size = FLAGS.batch_size
    logger.info("batch size is {}".format(LSTMBid_config.trainer.batch_size))



    trainX, trainY, valX, valY, tokenizer, always_unknown_train, always_unknown_test, statistics = \
        prepare_data_new.main(LSTMBid_config.data_loader.name, LSTMBid_config.data_loader.window_size)

    word_index = tokenizer.word_index
    logger.info('Found {} unique tokens.'.format(len(word_index) + 1))

    vocab_size = len(word_index) + 1

    #create unique report folder
    random_nr = randint(0, 10000)
    unique_folder_key = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S') + "-" + str(random_nr)
    report_folder = path_file.report_folder
    report_folder_LSTMBid = os.path.join(report_folder, 'reports-' + LSTMBid_config.name + '-' + unique_folder_key)

    os.mkdir(report_folder_LSTMBid)
    histories = Histories(report_folder_LSTMBid, tokenizer)


    # write in report folder
    with open(os.path.join(report_folder_LSTMBid, 'LSTMBid.json'), 'w') as outfile:
        json.dump(LSTMBid_config, outfile, indent=4)

    logger.info("create LSTMBid Model...")
    model2 = LSTMModelBid(context_vocab_size=vocab_size,
                       windows_size=LSTMBid_config.data_loader.window_size,
                       config=LSTMBid_config, report_folder=report_folder_LSTMBid)

    data = [trainX, trainY, valX, valY]

    logger.info("create trainer...")
    trainer2 = AbstractTrain(model=model2.model, data=data,
                             tokenizer=tokenizer, config=LSTMBid_config,
                             callbacks=histories, report_folder=report_folder_LSTMBid)

    logger.info("start LSTMBid training...")
    trainer2.train()
    #trainer2.save_callback_predictions()

    logger.info("save evaluation to file")
    evaluator2 = Evaluator(trainer2, report_folder_LSTMBid)
    evaluator2.visualize(always_unknown_train, always_unknown_test)
    evaluator2.evaluate()

    tokenizer_path = os.path.join(report_folder_LSTMBid, 'tokenizer.pkl')
    dump(tokenizer, open(tokenizer_path, 'wb'))

    # safe statistics
    statistics_path = os.path.join(report_folder_LSTMBid, 'statistics.csv')
    statistics.to_csv(statistics_path)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()