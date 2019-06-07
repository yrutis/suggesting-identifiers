from __future__ import print_function

import json
import logging
import os
from datetime import datetime
from random import randint

from keras import Input, Model
from keras.layers import Bidirectional
from keras.optimizers import Adam

import src.utils.path as path_file
from src.data import prepare_data_token
import src.utils.config as config_loader
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping


def data():

    LSTMBid_config_path = path_file.LSTMBid_config_path
    LSTMBid_config = config_loader.get_config_from_json(LSTMBid_config_path)

    # get data
    trainX, trainY, valX, valY, tokenizer, always_unknown_train, always_unknown_test = \
        prepare_data_token.main(LSTMBid_config.data_loader.name, LSTMBid_config.data_loader.window_size)

    word_index = tokenizer.word_index
    print('Found {} unique tokens.'.format(len(word_index) + 1))

    vocab_size = len(word_index) + 1


    #create unique report folder
    random_nr = randint(0, 10000)
    unique_folder_key = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S') + "-" + str(random_nr)
    report_folder = path_file.report_folder
    report_folder_LSTMBid = os.path.join(report_folder, 'reports-' + LSTMBid_config.name + '-' + unique_folder_key)

    os.mkdir(report_folder_LSTMBid)


    return trainX, trainY, valX, valY, vocab_size, LSTMBid_config, report_folder_LSTMBid




def model(trainX, trainY, valX, valY, vocab_size, LSTMBid_config, report_folder_LSTMBid):
    logger = logging.getLogger(__name__)


    contextEmbedding = Embedding(input_dim=vocab_size, output_dim={{choice([64, 128, 256])}},
                                 input_length=LSTMBid_config.data_loader.window_size)

    tensor = Input(shape=(LSTMBid_config.data_loader.window_size,))
    c = contextEmbedding(tensor)
    c = Dropout({{uniform(0, 0.5)}})(c)
    c = Bidirectional(LSTM({{choice([30, 50, 100])}}, recurrent_dropout={{uniform(0, 0.2)}}, dropout={{uniform(0, 0.2)}}))(c)
    c = Dropout({{uniform(0, 0.5)}})(c)
    c = Dense({{choice([30, 50, 70])}}, activation={{choice(['sigmoid', 'elu', 'selu'])}})(c)
    c = Dropout({{uniform(0, 0.5)}})(c)

    if {{choice(['three', 'four'])}} == 'four':
        c = Dense({{choice([30, 50, 70])}}, activation={{choice(['sigmoid', 'elu', 'selu'])}})(c)
        c = Dropout({{uniform(0, 0.5)}})(c)
    answer = Dense(vocab_size, activation='softmax')(c)

    model = Model(tensor, answer)
    optimizer = Adam(lr={{choice([0.001, 3e-4])}})
    model.compile(optimizer=optimizer, loss=LSTMBid_config.model.loss, metrics=LSTMBid_config.model.metrics)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   mode= 'min',
                                   patience=4)

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

    trainX, trainY, valX, valY, vocab_size, LSTMBid_config, report_folder_LSTM = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(valX, valY))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    best_model.save(os.path.join(report_folder_LSTM, 'best_model.h5'))
    json.dump(best_run, open(os.path.join(report_folder_LSTM, "best_run.txt"), 'w'))