import logging

import talos as ta
import wrangle as wr
from keras import Model, Input
from talos.metrics.keras_metrics import fmeasure_acc
from talos import live

import pandas as pd

from keras.models import Sequential
from keras.layers import Dropout, Dense, Embedding, LSTM

# Keras items
from keras.optimizers import Adam, Nadam
from keras.activations import relu, elu
from keras.losses import binary_crossentropy


import json
import logging
import os
from datetime import datetime
from random import randint

from keras import Input, Model
from keras.optimizers import Adam

import src.utils.path as path_file
from src.data import prepare_data_subtoken_no_generator
import src.utils.config as config_loader
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping

import tensorflow as tf

def data():
    logger = logging.getLogger(__name__)


    seq2seq_config_path = path_file.seq2seq_config_path
    seq2seq_config = config_loader.get_config_from_json(seq2seq_config_path)

    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('data', seq2seq_config.data_loader.name,
                               'must be valid data')

    seq2seq_config.data_loader.name = FLAGS.data
    logger.info("data used is {}".format(seq2seq_config.data_loader.name))

    trainX, trainY, valX, valY, tokenizer, vocab_size, max_input_elemts, max_output_elemts = \
        prepare_data_subtoken_no_generator.main(seq2seq_config.data_loader.name,
                                seq2seq_config.data_loader.window_size_params,
                                seq2seq_config.data_loader.window_size_body,
                                   seq2seq_config.data_loader.window_size_name
                                   )

    vocab_size = len(tokenizer.word_index) + 1
    logger.info('Found {} unique tokens.'.format(vocab_size))

    random_nr = randint(0, 10000)
    unique_folder_key = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S') + "-" + str(random_nr)
    report_folder = path_file.report_folder
    report_folder_seq2seq = os.path.join(report_folder, 'reports-' + seq2seq_config.name + '-' + unique_folder_key)

    os.mkdir(report_folder_seq2seq)


    return trainX, trainY, valX, valY, vocab_size, seq2seq_config, report_folder_seq2seq, max_input_elemts



trainX, trainY, valX, valY, vocab_size, seq2seq_config, report_folder_seq2seq, max_input_elemts = data()


# first we have to make sure to input data and params into the function
def seq2seq(x_train, y_train, x_val, y_val, params):

    e = Embedding(vocab_size, params['embedding'])
    encoder_inputs = Input(shape=(None,), name="encoder_input")
    en_x = e(encoder_inputs)
    encoder = LSTM(params['LSTM'], return_state=True, recurrent_dropout=params['recurrent_dropout1'], dropout=params['dropout1'])
    encoder_outputs, state_h, state_c = encoder(en_x)
    encoder_states = [state_h, state_c]
    decoder_inputs = Input(shape=(None,), name='decoder_input')
    dex = e
    final_dex = dex(decoder_inputs)
    decoder_lstm = LSTM(params['LSTM'], return_sequences=True, return_state=True, recurrent_dropout=params['recurrent_dropout2'], dropout=params['dropout2'])
    decoder_outputs, _, _ = decoder_lstm(final_dex, initial_state=encoder_states)

    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    #print(model.summary())

    history = model.fit({"encoder_input": x_train[0], "decoder_input": x_train[1]},
                        y_train,
                        batch_size=32,
                        epochs=params["epochs"],
                        validation_data=({"encoder_input": x_val[0], "decoder_input": x_val[1]}, y_val),
                        verbose=2)


    return history, model

# then we can go ahead and set the parameter space
p = {'embedding':[64, 128, 256],
     'LSTM':[60, 120, 250],
     'epochs': [10, 15, 20],
     'recurrent_dropout1': [0, 0.2, 0.5],
     'dropout1': [0, 0.2, 0.5],
     'recurrent_dropout2': [0, 0.2, 0.5],
     'dropout2': [0, 0.2, 0.5],
     }

# and run the experiment
t = ta.Scan(x=trainX,
            y=trainY,
            x_val=valX,
            y_val=valY,
            model=seq2seq,
            params=p,
            dataset_name='seq2seq_eval',
            experiment_no='1',
            print_params=True)