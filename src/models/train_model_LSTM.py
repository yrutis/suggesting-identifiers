# -*- coding: utf-8 -*-
import ast
import logging
import os


import keras
from keras import Input
from keras import layers
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model
from keras.utils import to_categorical

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

import matplotlib.pyplot as plt

def main(filename):
    """ runs LSTM model
    """

    #get logger
    logger = logging.getLogger(__name__)


    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    processed_decoded_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'processed'), 'decoded'), filename + '.csv') #get decoded path
    logger.info(processed_decoded_full_path)

    processedDf = pd.read_csv(processed_decoded_full_path)

    context = processedDf['x'].apply(ast.literal_eval) #saves all context x as list in list

    tokenizer = Tokenizer()  # init new tokenizer
    tokenizer.fit_on_texts(context)
    sequences = tokenizer.texts_to_sequences(context)
    padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None,
                                                                  value=0)  # make sure they are all same length

    contextVocabSize = len(tokenizer.word_index) + 1
    print('Found %s unique tokens.' % contextVocabSize)

    lengthTrainX = padded_sequences.shape[0]
    trainX = padded_sequences[0: int(0.9 * lengthTrainX)]

    print("trainW1 shape {}".format(trainX.shape))

    # load Y's
    y = processedDf['y'] #get all Y
    Y = y.values  # convert to numpy
    #Y = y[:, 2]  # get Y values
    print("first load y: {}".format(Y[0]))

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    lenY = len(np.unique(Y))  # amount of unique Y's
    print("this is length of y {} and this of encoded Y {}".format(len(np.unique(Y)), len(np.unique(encoded_Y))))


    #select only 90% for training
    trainYEnc = encoded_Y[0: int(0.9 * Y.shape[0])]
    trainY = to_categorical(trainYEnc, num_classes=lenY)

    print("this is the shape of trainY {}".format(trainY.shape))
    print("this is the first of trainX {}".format(trainX[0:1]))

    contextEmbedding = Embedding(output_dim=50, input_dim=contextVocabSize, input_length=8)


    tensor1 = Input(shape=(8,))
    c1 = contextEmbedding(tensor1)
    c1 = LSTM(50)(c1)
    c1 = keras.layers.Dense(contextVocabSize)(c1)
    answer = layers.Dense(lenY, activation='softmax')(c1)

    model = Model(tensor1, answer)
    optimizer = keras.optimizers.Adam(lr=0.007)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())

    history = model.fit(trainX, trainY,
              validation_split=0.1,
              epochs=3,
              batch_size=100)


    model_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
    if not os.path.exists(model_folder):  # check if path exists
        print("creating models folder...")
        os.mkdir(model_folder)
    plotted_model = os.path.join(model_folder, 'model_LSTM.png')
    plot_model(model, to_file=plotted_model)
    logger.info("Saved model architecture to disk")

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    logger.info("Saved model (JSON + Weights) to disk")

    # saving tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved model tokenizer to disk")



    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('acc.png')
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('loss.png')



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    filename = 'bigbluebutton_methoddeclarations_train'
    main(filename)

