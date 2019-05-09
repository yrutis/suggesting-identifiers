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
from src.models.SimpleNN_new import SimpleNNModel
import src.utils.config as config_loader
import src.utils.path as path_file
from src.trainer.AbstractTrain import AbstractTrain


def main():
    # get logger
    logger = logging.getLogger(__name__)
    simpleNN_config_path = path_file.simpleNN_config_path
    simpleNN_config = config_loader.get_config_from_json(simpleNN_config_path)

    FLAGS = tf.app.flags.FLAGS
    #define some tf flags

    tf.app.flags.DEFINE_string('data', simpleNN_config.data_loader.name,
                               'must be either Android-Universal-Image-Loader or all_methods_train')
    simpleNN_config.data_loader.name = FLAGS.data
    logger.info("data used is {}".format(simpleNN_config.data_loader.name))

    tf.app.flags.DEFINE_integer('window_size', simpleNN_config.data_loader.window_size, 'must be between 2+')
    simpleNN_config.data_loader.window_size = FLAGS.window_size
    logger.info("window size is {}".format(simpleNN_config.data_loader.window_size))

    tf.app.flags.DEFINE_integer('epochs', simpleNN_config.trainer.num_epochs, 'must be between 1-100')
    simpleNN_config.trainer.num_epochs = FLAGS.epochs
    logger.info("epochs num is {}".format(simpleNN_config.trainer.num_epochs))

    tf.app.flags.DEFINE_integer('batch_size', simpleNN_config.trainer.batch_size, 'must be a power of 2 2^1 - 2^6')
    simpleNN_config.trainer.batch_size = FLAGS.batch_size
    logger.info("batch size is {}".format(simpleNN_config.trainer.batch_size))

    trainX, trainY, valX, valY, tokenizer, always_unknown_train, always_unknown_test = \
        prepare_data_new.main(simpleNN_config.data_loader.name, simpleNN_config.data_loader.window_size)

    word_index = tokenizer.word_index
    logger.info('Found {} unique tokens.'.format(len(word_index) + 1))

    vocab_size = len(word_index) + 1
    histories = Histories()

    print(FLAGS.data)

    #create unique report folder
    random_nr = randint(0, 10000)
    unique_folder_key = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S') + "-" + str(random_nr)
    report_folder = path_file.report_folder
    report_folder_LSTM = os.path.join(report_folder, 'reports-' + simpleNN_config.name + '-' + unique_folder_key)

    os.mkdir(report_folder_LSTM)

    # write in report folder
    with open(os.path.join(report_folder_LSTM, 'simpleNN.json'), 'w') as outfile:
        json.dump(simpleNN_config, outfile, indent=4)

    logger.info("create LSTM Model...")
    model2 = SimpleNNModel(context_vocab_size=vocab_size,
                       windows_size=simpleNN_config.data_loader.window_size,
                       config=simpleNN_config, report_folder=report_folder_LSTM)

    data = [trainX, trainY, valX, valY]

    logger.info("create trainer...")
    trainer2 = AbstractTrain(model=model2.model, data=data,
                             tokenizer=tokenizer, config=simpleNN_config,
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