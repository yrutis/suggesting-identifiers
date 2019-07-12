from __future__ import print_function

import json
import logging
import os

from keras import Input, Model
from keras.optimizers import Adam

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping

import tensorflow as tf
import talos as ta

def data():
    import os
    from datetime import datetime
    from random import randint
    import src.utils.path as path_file

    from src.token_approach.data import prepare_data_token_no_generator
    import src.utils.config as config_loader

    logger = logging.getLogger(__name__)

    token_model_config_path = path_file.token_model_config_path
    token_model_config = config_loader.get_config_from_json(token_model_config_path)

    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('data', token_model_config.data_loader.name,
                               'must be valid data')

    token_model_config.data_loader.name = FLAGS.data
    logger.info("data used is {}".format(token_model_config.data_loader.name))


    # create unique report folder
    random_nr = randint(0, 10000)
    unique_folder_key = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S') + "-" + str(random_nr)
    report_folder = path_file.report_folder
    report_folder_token_model = os.path.join(report_folder, 'reports-hyperopt' + token_model_config.name + '-' + unique_folder_key)

    os.mkdir(report_folder_token_model)

    # get data
    trainX, trainY, valX, valY, tokenizer, always_unknown_train, always_unknown_test, window_size = \
        prepare_data_token_no_generator.main(token_model_config.data_loader.name,
                                token_model_config.data_loader.window_size_params,
                                token_model_config.data_loader.window_size_body,
                                remove_train_unk=0.5,
                                remove_val_unk=0.6,
                                report_folder=report_folder)

    vocab_size = len(tokenizer.word_index) + 1
    print('Found {} unique tokens.'.format(vocab_size))

    return trainX, trainY, valX, valY, vocab_size, token_model_config, report_folder_token_model, window_size



trainX, trainY, valX, valY, vocab_size, token_model_config, report_folder_token_model, window_size = data()


def simpleNN(trainX, trainY, valX, valY, params):
    logger = logging.getLogger(__name__)


    contextEmbedding = Embedding(input_dim=vocab_size, output_dim=params['embedding'],
                                 input_length=window_size)

    tensor = Input(shape=(window_size,))
    c = contextEmbedding(tensor)
    c = Dropout(params['dropout1'])(c)
    c = Flatten()(c)
    c = Dropout(params['dropout2'])(c)
    c = Dense(params['dense_dim'], activation=params['activation_function_1'])(c)
    c = Dropout(params['dropout3'])(c)

    answer = Dense(vocab_size, activation='softmax')(c)

    model = Model(tensor, answer)
    optimizer = Adam(lr=params['learning_rate'])
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['acc'])

    early_stopping = EarlyStopping(monitor='val_loss',
                                   mode= 'min',
                                   patience=5)

    history = model.fit(trainX, trainY,
              batch_size=params['batch_size'],
              epochs=params['epochs'],
              verbose=2,
              validation_data=(valX, valY),
              callbacks=[early_stopping])

    return history, model



# then we can go ahead and set the parameter space
p = {'embedding':[64, 128, 256],
     'activation_function_1':['relu', 'selu', 'elu'],
     'batch_size': [32, 64, 128],
     'epochs': [10, 15, 20, 30, 40],
     'dropout1': [0, 0.2, 0.5],
     'dropout2': [0, 0.2, 0.5],
     'dropout3': [0, 0.2, 0.5],
     'dense_dim': [70, 100, 200],
     'learning_rate': [0.001, 3e-4],
     }



# and run the experiment
t = ta.Scan(x=trainX,
            y=trainY,
            x_val=valX,
            y_val=valY,
            model=simpleNN,
            params=p,
            dataset_name='java-small-proj-split',
            experiment_no='1',
            print_params=True)