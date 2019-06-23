from __future__ import print_function

import json
import logging
import os
from datetime import datetime
from random import randint

from keras import Input, Model
from keras.optimizers import Adam

import src.utils.path as path_file
from src.data import prepare_data_subtoken
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
        prepare_data_subtoken.main(seq2seq_config.data_loader.name,
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

    trainX = {'x1': trainX[0], 'x2': trainX[1]}
    valX = {'x1': valX[0], 'x2': valY[1]}

    return trainX, trainY, valX, valY, vocab_size, seq2seq_config, report_folder_seq2seq, max_input_elemts




def model(trainX, trainY, valX, valY, vocab_size, seq2seq_config, report_folder_seq2seq, max_input_elemts):
    logger = logging.getLogger(__name__)

    e = Embedding(vocab_size, {{choice([64, 128, 256])}})
    encoder_inputs = Input(shape=(None,), name="encoder_input")
    en_x = e(encoder_inputs)
    encoder = LSTM({{choice([60, 100, 250])}}, return_state=True)
    encoder_outputs, state_h, state_c = encoder(en_x)
    encoder_states = [state_h, state_c]
    decoder_inputs = Input(shape=(None,), name='decoder_input')
    dex = e
    final_dex = dex(decoder_inputs)
    decoder_lstm = LSTM({{choice([50, 100, 200])}}, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(final_dex, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())


    model.fit({'x1': trainX['x1'], 'x2': trainX['x2']}, trainY,
                        batch_size={{choice([50, 100, 200])}},
                        epochs=2,
                        validation_data=[{'x1': valX['x1'], 'x2': valX['x2']}, valY])

    score, acc = model.evaluate({'x1': valX['x1'], 'x2': valX['x2']}, valY, verbose=0)

    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=2,
                                          trials=Trials())
    print(best_run)

    trainX, trainY, valX, valY, vocab_size, seq2seq_config, report_folder_seq2seq, max_input_elemts = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(valX, valY))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    best_model.save(os.path.join(report_folder_seq2seq, 'best_model.h5'))
    json.dump(best_run, open(os.path.join(report_folder_seq2seq, "best_run.txt"), 'w'))