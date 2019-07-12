import shutil
import os
import matplotlib as mpl
from keras.engine.saving import load_model

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import pandas as pd
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from datetime import datetime
from random import randint
from pickle import load
import logging
import os
import tensorflow as tf
import numpy as np
from src.Vocabulary.Vocabulary import Vocabulary


import src.token_approach.data.prepare_data_token as prepare_data
import src.token_approach.data.prepare_data_test_token as prepare_data_test

from src.token_approach.models.SimpleNN import SimpleNNModel
from src.token_approach.models.LSTMModel import LSTMModel
from src.token_approach.models.LSTMBidModel import LSTMModelBid
from src.token_approach.models.GRUModel import GRUModel

import src.utils.config as config_loader
import src.utils.path as path_file
from src.token_approach.trainer.AbstractTrain import AbstractTrain

def train_model(config, report_folder):
    # get logger
    logger = logging.getLogger(__name__)

    # get trainX, trainY, valX, valY, tokenizer (dictionary), unknown statistics, window_size of X
    all_train, all_val, vocab_size, window_size, data_storage, perc_unk_train, perc_unk_val\
        = prepare_data.main(config.data_loader.name,
                          config.data_loader.window_size_params,
                          config.data_loader.window_size_body,
                          partition=config.data_loader.partition,
                          report_folder=report_folder,
                          remove_train_unk=config.data_loader.remove_train_unk,
                          remove_val_unk=config.data_loader.remove_val_unk)

    logger.info('Found {} unique tokens.'.format(vocab_size))

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
        logger.info("create LSTMBid Model...")
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
    trainer = AbstractTrain(model=model.model, config=config,
                             report_folder=report_folder)

    logger.info("start training...")
    trainer.train(all_train=all_train, all_val=all_val, data_storage=data_storage, window_size=window_size)
    trainer.visualize_training(perc_unk_train, perc_unk_val)

    logger.info("saving the model")
    trainer.model.save(os.path.join(report_folder, "best_model.h5"))



def eval_model(config, report_folder):
    # get logger
    logger = logging.getLogger(__name__)

    logger.info("loading tokenizer...")
    try:
        with open(os.path.join(report_folder, 'tokenizer.pkl'), "rb") as input_file:
            tokenizer = load(input_file)
    except:
        raise Exception("Could not load the dictionary, error in preparing data...")


    logger.info("loading best model...")
    try:
        model = load_model(os.path.join(report_folder, 'best_model.h5')) #load the best model before evaluating
    except:
        raise Exception("Could not load the model, error in training...")


    logger.info("load test data for evaluation")
    testX, testY, perc_unk_test = prepare_data_test.main(config.data_loader.name,
                                                         config.data_loader.window_size_params,
                                                         config.data_loader.window_size_body,
                                                         tokenizer,
                                                         remove_test_unk=config.data_loader.remove_test_unk)



    logger.info("evaluating model...")
    loss_acc_list_of_metrics = model.evaluate(testX, testY, verbose=0) #evaluate for acc and top5 acc

    acc = loss_acc_list_of_metrics[1]
    top_5_acc = loss_acc_list_of_metrics[2]

    logger.info("getting all predictions, probabilities...")
    predictions = model.predict(testX, batch_size=8, verbose=0)  # get prob dist
    predictions_idx = np.argmax(predictions, axis=1)  # get highest idx for each X
    predictions_prob = np.amax(predictions, axis=1).tolist()  # get highest prob for each X
    #predictions_prob = np.take(predictions, predictions_idx).tolist()
    predictions_decoded = Vocabulary.revert_back(tokenizer=tokenizer, sequence=predictions_idx)
    testX = testX.tolist()
    testX = [Vocabulary.revert_back(tokenizer=tokenizer, sequence=x) for x in testX]
    testY = Vocabulary.revert_back(tokenizer=tokenizer, sequence=testY)

    logger.info("saving input, corrects, predictions, probabilities to file...")
    # save model data
    model_data = {'Input': testX,
                  'Correct': testY,
                  'Prediction': predictions_decoded,
                  'PredictionProb': predictions_prob}


    df = pd.DataFrame(model_data, columns=['Input', 'Correct', 'Prediction', 'PredictionProb'])
    df.to_csv(os.path.join(report_folder, 'predictions_test.csv'))

    logger.info("saving metrics...")
    # save metrics
    metrics = {'Accuracy': acc,
               'Top-5-Acc': top_5_acc}

    df = pd.DataFrame([metrics], columns=['Accuracy', 'Top-5-Acc'])

    report_file = os.path.join(report_folder, 'metrics.csv')
    df.to_csv(report_file, index=False)

def main(config_path):
    # get logger
    logger = logging.getLogger(__name__)

    config = config_loader.get_config_from_json(config_path)

    FLAGS = tf.app.flags.FLAGS
    # define some tf flags

    tf.app.flags.DEFINE_string('model', config.name,
                               'must be a valid token model simpleNN/ GRU/ LSTM/ LSTMBid')
    tf.app.flags.DEFINE_string('data', config.data_loader.name,
                               'must be ...')
    tf.app.flags.DEFINE_integer('window_size_body', config.data_loader.window_size_body, 'somewhere between 2 and 30')
    tf.app.flags.DEFINE_integer('window_size_params', config.data_loader.window_size_params, 'somewhere between 2 and 10')
    tf.app.flags.DEFINE_integer('epochs', config.trainer.num_epochs, 'somewhere between 1 and 50')
    tf.app.flags.DEFINE_integer('batch_size', config.trainer.batch_size, 'must be a power of 2 2^1 - 2^6')
    tf.app.flags.DEFINE_integer('partition', config.data_loader.partition, 'should be between 10k-100k and dividable by batch_size')
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

    config.data_loader.partition = FLAGS.partition
    logger.info("partition is {}".format(config.data_loader.partition))

    config.data_loader.remove_train_unk = FLAGS.remove_train_unk
    logger.info("remove_train_unk is {}".format(config.data_loader.remove_train_unk))

    config.data_loader.remove_val_unk = FLAGS.remove_val_unk
    logger.info("remove_val_unk is {}".format(config.data_loader.remove_val_unk))

    config.data_loader.remove_test_unk = FLAGS.remove_test_unk
    logger.info("remove_test_unk is {}".format(config.data_loader.remove_test_unk))

    #create unique report folder
    random_nr = randint(0, 10000)
    unique_folder_key = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S') + "-" + str(random_nr)
    report_folder = os.path.join(path_file.report_folder, 'reports-' + config.name + '-' + unique_folder_key)

    os.mkdir(report_folder)

    # 1) train model
    train_model(config=config, report_folder=report_folder)

    # 2) eval model
    eval_model(config=config, report_folder=report_folder)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    token_model_config_path = path_file.token_model_config_path
    main(token_model_config_path)