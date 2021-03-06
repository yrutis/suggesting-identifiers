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

from src.token_approach.hyperopt.hyperopt_data import data


def model(trainX, trainY, valX, valY, vocab_size, simpleNN_config, report_folder_simpleNN, window_size):
    logger = logging.getLogger(__name__)


    contextEmbedding = Embedding(input_dim=vocab_size, output_dim={{choice([64, 128, 256])}},
                                 input_length=window_size)

    tensor = Input(shape=(window_size,))
    c = contextEmbedding(tensor)
    c = Dropout({{uniform(0, 0.5)}})(c)
    c = Flatten()(c)
    c = Dropout({{uniform(0, 0.5)}})(c)
    c = Dense({{choice([50, 70, 100, 200])}}, activation={{choice(['relu', 'elu', 'selu'])}})(c)
    c = Dropout({{uniform(0, 0.5)}})(c)

    answer = Dense(vocab_size, activation='softmax')(c)

    model = Model(tensor, answer)
    optimizer = Adam(lr={{choice([0.001, 3e-4])}})
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['acc'])

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
                                          max_evals=10,
                                          trials=Trials())
    print(best_run)

    trainX, trainY, valX, valY, vocab_size, simpleNN_config, report_folder_simpleNN, window_size = data()
    print("Evaluation of best performing model:")
    print(best_model.evaluate(valX, valY))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    best_model.save(os.path.join(report_folder_simpleNN, 'best_model.h5'))
    json.dump(best_run, open(os.path.join(report_folder_simpleNN, "best_run.txt"), 'w'))