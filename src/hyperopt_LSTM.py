from __future__ import print_function

import json
import logging
import os
from datetime import datetime
from random import randint

from keras import Input, Model
from keras.optimizers import RMSprop, Adam
from src.models.LSTMModel import LSTMModel

import src.utils.path as path_file
from src.data import prepare_data
import src.utils.config as config_loader
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint


def data():
    maxlen = 100
    max_features = 20000

    # load default settings
    LSTM_config_path = path_file.LSTM_opt_config_path
    LSTM_config = config_loader.get_config_from_json(LSTM_config_path)

    # get data
    trainX, trainY, valX, valY, tokenizer, always_unknown_train, always_unknown_test = \
        prepare_data.main(LSTM_config.data_loader.name, LSTM_config.data_loader.window_size)

    word_index = tokenizer.word_index
    print('Found {} unique tokens.'.format(len(word_index) + 1))

    vocab_size = len(word_index) + 1


    #create unique report folder
    random_nr = randint(0, 10000)
    unique_folder_key = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S') + "-" + str(random_nr)
    report_folder = path_file.report_folder
    report_folder_LSTM = os.path.join(report_folder, 'reports-' + LSTM_config.name + '-' + unique_folder_key)

    os.mkdir(report_folder_LSTM)

    # write in report folder
    with open(os.path.join(report_folder_LSTM, 'LSTM.json'), 'w') as outfile:
        json.dump(LSTM_config, outfile, indent=4)

    return trainX, trainY, valX, valY, vocab_size, LSTM_config


def model(trainX, trainY, valX, valY, vocab_size, LSTM_config):
    logger = logging.getLogger(__name__)


    contextEmbedding = Embedding(input_dim=vocab_size, output_dim=LSTM_config.model.embedding_dim,
                                 input_length=8)

    tensor = Input(shape=(LSTM_config.data_loader.window_size,))
    c = contextEmbedding(tensor)
    c = LSTM(LSTM_config.model.lstm_dim, recurrent_dropout=0.2, dropout=0.2)(c)
    c = Dropout({{uniform(0, 5)}})(c)
    c = Dense(100, activation='sigmoid')(c)
    c = Dropout(0.2)(c)
    answer = Dense(vocab_size, activation='softmax')(c)

    model = Model(tensor, answer)
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss=LSTM_config.model.loss, metrics=LSTM_config.model.metrics)

    model.fit(trainX, trainY,
              batch_size={{choice([64, 128])}},
              epochs=1,
              verbose=2,
              validation_data=(valX, valY))
    score, acc = model.evaluate(valX, valY, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
    print(best_run)