from __future__ import print_function

import json
import logging
import os
from datetime import datetime
from random import randint

from keras import Input, Model
from keras.optimizers import Adam

import src.utils.path as path_file
from src.data import prepare_data_token
import src.utils.config as config_loader
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping

import tensorflow as tf


def data():

    simpleNN_config_path = path_file.simpleNN_config_path
    simpleNN_config = config_loader.get_config_from_json(simpleNN_config_path)

    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('data', simpleNN_config.data_loader.name,
                               'must be valid data')

    simpleNN_config.data_loader.name = FLAGS.data
    print("data used is {}".format(simpleNN_config.data_loader.name))

    # get data
    trainX, trainY, valX, valY, tokenizer, always_unknown_train, always_unknown_test, window_size = \
        prepare_data_token.main(simpleNN_config.data_loader.name,
                                simpleNN_config.data_loader.window_size_params,
                                simpleNN_config.data_loader.window_size_body,
                                remove_val_unk=0.8)

    vocab_size = len(tokenizer.word_index) + 1
    print('Found {} unique tokens.'.format(vocab_size))

    #create unique report folder
    random_nr = randint(0, 10000)
    unique_folder_key = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S') + "-" + str(random_nr)
    report_folder = path_file.report_folder
    report_folder_simpleNN = os.path.join(report_folder, 'reports-' + simpleNN_config.name + '-' + unique_folder_key)

    os.mkdir(report_folder_simpleNN)


    return trainX, trainY, valX, valY, vocab_size, simpleNN_config, report_folder_simpleNN, window_size




def model(trainX, trainY, valX, valY, vocab_size, simpleNN_config, report_folder_simpleNN, window_size):
    logger = logging.getLogger(__name__)


    contextEmbedding = Embedding(input_dim=vocab_size, output_dim={{choice([64, 128, 256])}},
                                 input_length=window_size)

    tensor = Input(shape=(window_size,))
    c = contextEmbedding(tensor)
    c = Dropout({{uniform(0, 0.5)}})(c)
    c = Flatten()(c)
    c = Dropout({{uniform(0, 0.5)}})(c)
    c = Dense({{choice([50, 70, 100, 200])}}, activation={{choice(['sigmoid', 'relu', 'elu', 'selu'])}})(c)
    c = Dropout({{uniform(0, 0.5)}})(c)

    if {{choice(['three', 'four'])}} == 'four':
        c = Dense({{choice([30, 50, 70, 100, 200])}}, activation={{choice(['sigmoid', 'relu', 'elu', 'selu'])}})(c)
        c = Dropout({{uniform(0, 0.5)}})(c)
    answer = Dense(vocab_size, activation='softmax')(c)

    model = Model(tensor, answer)
    optimizer = Adam(lr={{choice([0.001, 3e-4])}})
    model.compile(optimizer=optimizer, loss=simpleNN_config.model.loss, metrics=simpleNN_config.model.metrics)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   mode= 'min',
                                   patience=5)

    model.fit(trainX, trainY,
              batch_size={{choice([64, 128])}},
              epochs={{choice([10, 15, 20, 30])}},
              verbose=2,
              validation_data=(valX, valY),
              callbacks=[early_stopping])
    score, acc = model.evaluate(valX, valY, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=30,
                                          trials=Trials())
    print(best_run)

    trainX, trainY, valX, valY, vocab_size, simpleNN_config, report_folder_simpleNN, window_size = data()
    print("Evaluation of best performing model:")
    print(best_model.evaluate(valX, valY))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    best_model.save(os.path.join(report_folder_simpleNN, 'best_model.h5'))
    json.dump(best_run, open(os.path.join(report_folder_simpleNN, "best_run.txt"), 'w'))