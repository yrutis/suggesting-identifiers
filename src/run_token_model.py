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
import numpy as np


import src.data.prepare_data_token as prepare_data
import src.data.prepare_data_test_token as prepare_data_test


from src.evaluator.EvaluatorSubtoken import Evaluator
from src.models.SimpleNN import SimpleNNModel
from src.models.LSTMModel import LSTMModel
from src.models.LSTMBidModel import LSTMModelBid
from src.models.GRUModel import GRUModel

import src.utils.config as config_loader
import src.utils.path as path_file
from src.trainer.AbstractTrain import AbstractTrain


def main(config_path):
    # get logger
    logger = logging.getLogger(__name__)

    config = config_loader.get_config_from_json(config_path)

    FLAGS = tf.app.flags.FLAGS
    # define some tf flags

    tf.app.flags.DEFINE_string('model', config.name,
                               'must be a valid token model simpleNN/ GRU/ LSTM/ LSTMBid')
    tf.app.flags.DEFINE_string('data', config.data_loader.name,
                               'must be either Android-Universal-Image-Loader or all_methods_train')
    tf.app.flags.DEFINE_integer('window_size_body', config.data_loader.window_size_body, 'must be between 2+')
    tf.app.flags.DEFINE_integer('window_size_params', config.data_loader.window_size_params, 'must be between 2+')
    tf.app.flags.DEFINE_integer('epochs', config.trainer.num_epochs, 'must be between 1-100')
    tf.app.flags.DEFINE_integer('batch_size', config.trainer.batch_size, 'must be a power of 2 2^1 - 2^6')
    tf.app.flags.DEFINE_float('remove_train_unk', config.data_loader.remove_train_unk, 'must be between 0 and 1')
    tf.app.flags.DEFINE_float('remove_val_unk', config.data_loader.remove_val_unk, 'must be a between 0 and 1')
    tf.app.flags.DEFINE_float('remove_test_unk', config.data_loader.remove_test_unk, 'must be a between 0 and 1')

    config.name = FLAGS.model
    logger.info("model used is {}".format(config.name))

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

    config.data_loader.remove_train_unk = FLAGS.remove_train_unk
    logger.info("remove_train_unk is {}".format(config.data_loader.remove_train_unk))

    config.data_loader.remove_val_unk = FLAGS.remove_val_unk
    logger.info("remove_val_unk is {}".format(config.data_loader.remove_val_unk))

    config.data_loader.remove_test_unk = FLAGS.remove_test_unk
    logger.info("remove_test_unk is {}".format(config.data_loader.remove_test_unk))


    # get trainX, trainY, valX, valY, tokenizer (dictionary), unknown statistics, window_size of X
    trainX, trainY, valX, valY, tokenizer, perc_unk_train, perc_unk_val, window_size = \
        prepare_data.main(config.data_loader.name,
                          config.data_loader.window_size_params,
                          config.data_loader.window_size_body,
                          remove_train_unk=config.data_loader.remove_train_unk,
                          remove_val_unk=config.data_loader.remove_val_unk)


    #load test data for evaluation
    testX, testY, perc_unk_test = prepare_data_test.main(config.data_loader.name,
                                                     config.data_loader.window_size_params,
                                                     config.data_loader.window_size_body,
                                                     tokenizer,
                                                     remove_test_unk=config.data_loader.remove_test_unk)


    vocab_size = len(tokenizer.word_index) + 1
    logger.info('Found {} unique tokens.'.format(vocab_size))

    #create unique report folder
    random_nr = randint(0, 10000)
    unique_folder_key = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S') + "-" + str(random_nr)
    report_folder = os.path.join(path_file.report_folder, 'reports-' + config.name + '-' + unique_folder_key)

    os.mkdir(report_folder)

    if config.name == "GRU":
        # load specific graph settings for model
        model_config = config_loader.get_config_from_json(path_file.GRU_config_path)
        config.model = model_config.model

        # create the model
        logger.info("create GRU Model...")
        model = GRUModel(context_vocab_size=vocab_size,
                           windows_size=window_size,
                           config=config, report_folder=report_folder)

    elif config.name == "LSTM":
        # load specific graph settings for model
        model_config = config_loader.get_config_from_json(path_file.LSTM_config_path)
        config.model = model_config.model

        # create the model
        logger.info("create LSTM Model...")
        model = LSTMModel(context_vocab_size=vocab_size,
                              windows_size=window_size,
                              config=config, report_folder=report_folder)

    elif config.name == "LSTMBid":
        # load specific graph settings for model
        model_config = config_loader.get_config_from_json(path_file.LSTMBid_config_path)
        config.model = model_config.model

        #create the model
        logger.info("create LSTM Model...")
        model = LSTMModelBid(context_vocab_size=vocab_size,
                              windows_size=window_size,
                              config=config, report_folder=report_folder)

    else:
        #load specific graph settings for model
        model_config = config_loader.get_config_from_json(path_file.simpleNN_config_path)
        config.model = model_config.model

        #create the model
        logger.info("create simpleNN Model...")
        model = SimpleNNModel(context_vocab_size=vocab_size,
                              windows_size=window_size,
                              config=config, report_folder=report_folder)


    logger.info("create trainer...")
    trainer = AbstractTrain(model=model.model, data=[trainX, trainY, valX, valY, testX, testY],
                             tokenizer=tokenizer, config=config,
                             report_folder=report_folder)

    logger.info("start training...")
    trainer.train()
    trainer.visualize_training(perc_unk_train, perc_unk_val)

    logger.info("save evaluation to file")
    evaluator = Evaluator(trainer.model, report_folder)

    # write config in report folder
    with open(os.path.join(report_folder, config.name + '.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    tokenizer_path = os.path.join(report_folder, 'tokenizer.pkl')
    dump(tokenizer, open(tokenizer_path, 'wb'))

    # %% idx2word

    # Creating a reverse dictionary
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

    # Function takes a tokenized sentence and returns the words
    def sequence_to_text(list_of_indices):
        # Looking up words in dictionary
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        return (words)



    predictions = trainer.model.predict(testX, batch_size=8) #get prob dist
    predictions_idx = np.argmax(predictions, axis=1).tolist() #get highest idx for each X
    predictions_idx = [[item] for item in predictions_idx]
    correct = testY.tolist()
    correct = [[item] for item in correct]

    #evaluator.evaluate()
    acc, prec, rec, f1 = evaluator.get_accuracy_precision_recall_f1_score(correct, predictions_idx, 'token')






if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    token_model_config_path = path_file.token_model_config_path
    main(token_model_config_path)