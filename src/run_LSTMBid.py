from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from datetime import datetime
from random import randint
from pickle import dump
import logging
import os
import json
import tensorflow as tf


import src.data.prepare_data_token as prepare_data
from src.evaluator.Evaluator import Evaluator
from src.models.LSTMBidModel import LSTMModelBid
import src.utils.config as config_loader
import src.utils.path as path_file

from src.trainer.AbstractTrain import AbstractTrain


def main(config_path):
    # get logger
    logger = logging.getLogger(__name__)

    config = config_loader.get_config_from_json(config_path)

    FLAGS = tf.app.flags.FLAGS
    #define some tf flags

    tf.app.flags.DEFINE_string('data', config.data_loader.name,
                               'must be either Android-Universal-Image-Loader or all_methods_train')
    tf.app.flags.DEFINE_integer('window_size_body', config.data_loader.window_size_body, 'must be between 2+')
    tf.app.flags.DEFINE_integer('window_size_params', config.data_loader.window_size_params, 'must be between 2+')
    tf.app.flags.DEFINE_integer('epochs', config.trainer.num_epochs, 'must be between 1-100')
    tf.app.flags.DEFINE_integer('batch_size', config.trainer.batch_size, 'must be a power of 2 2^1 - 2^6')


    config.data_loader.name = FLAGS.data
    logger.info("data used is {}".format(config.data_loader.name))

    config.data_loader.window_size_body = FLAGS.window_size_body
    logger.info("window size body is {}".format(config.data_loader.window_size_body))

    config.data_loader.window_size_params = FLAGS.window_size_params
    logger.info("window size params is {}".format(config.data_loader.window_size_params))

    config.trainer.num_epochs = FLAGS.epochs
    logger.info("epochs num is {}".format(config.trainer.num_epochs))

    config.trainer.batch_size = FLAGS.batch_size
    logger.info("batch size is {}".format(config.trainer.batch_size))


    #get data, UNK and other statistics
    trainX, trainY, valX, valY, tokenizer, always_unknown_train, always_unknown_test, window_size = \
        prepare_data.main(config.data_loader.name, config.data_loader.window_size_params, config.data_loader.window_size_body)

    word_index = tokenizer.word_index
    logger.info('Found {} unique tokens.'.format(len(word_index) + 1))

    vocab_size = len(word_index) + 1

    print(FLAGS.data)

    #create unique report folder
    random_nr = randint(0, 10000)
    unique_folder_key = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S') + "-" + str(random_nr)
    report_folder = os.path.join(path_file.report_folder, 'reports-' + config.name + '-' + unique_folder_key)

    os.mkdir(report_folder)


    # write in report folder
    with open(os.path.join(report_folder, config.name+'.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    logger.info("create LSTMBid Model...")
    model2 = LSTMModelBid(context_vocab_size=vocab_size,
                       windows_size=window_size,
                       config=config, report_folder=report_folder)

    data = [trainX, trainY, valX, valY]

    logger.info("create trainer...")
    trainer2 = AbstractTrain(model=model2.model, data=data,
                             tokenizer=tokenizer, config=config,
                             report_folder=report_folder)

    logger.info("start LSTMBid training...")
    trainer2.train()

    logger.info("save evaluation to file")
    evaluator2 = Evaluator(trainer2, report_folder)
    evaluator2.visualize(always_unknown_train, always_unknown_test)
    evaluator2.evaluate()

    tokenizer_path = os.path.join(report_folder, 'tokenizer.pkl')
    dump(tokenizer, open(tokenizer_path, 'wb'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    LSTMBid_config_path = path_file.LSTMBid_config_path
    main(LSTMBid_config_path)