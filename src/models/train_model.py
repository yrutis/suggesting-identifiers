# -*- coding: utf-8 -*-
import ast
import logging
import os
import keras
from keras import Input
from keras import layers
from keras.layers import Embedding
from keras.layers import Flatten
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import pickle


def main(filename):
    """ runs model
    """
    processed_decoded_full_path = '../../data/processed/decoded/' + filename + '.csv'

    processedDf = pd.read_csv(processed_decoded_full_path)

    context = processedDf['x'].apply(ast.literal_eval) #saves all context x as list in list

    tokenizer = Tokenizer()  # init new tokenizer
    tokenizer.fit_on_texts(context)
    sequences = tokenizer.texts_to_sequences(context)
    padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None,
                                                                  value=0)  # make sure they are all same length

    contextVocabSize = len(tokenizer.word_index) + 1
    print('Found %s unique tokens.' % contextVocabSize)


    trainW1 = padded_sequences[0: int(0.9 * padded_sequences.shape[0]), 0]
    trainW2 = padded_sequences[0: int(0.9 * padded_sequences.shape[0]), 1]

    print("trainW1 shape {}".format(trainW1.shape))

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

    contextEmbedding = Embedding(output_dim=50, input_dim=contextVocabSize, input_length=1)

    tensor1 = Input(shape=(1,), dtype='int32', )
    c1 = contextEmbedding(tensor1)
    c1 = Flatten()(c1)
    c1 = keras.layers.Dense(contextVocabSize)(c1)

    tensor2 = Input(shape=(1,), dtype='int32', )
    c2 = contextEmbedding(tensor2)
    c2 = Flatten()(c2)
    c2 = keras.layers.Dense(contextVocabSize)(c2)

    added = keras.layers.Add()([c1, c2])
    answer = layers.Dense(lenY, activation='softmax')(added)

    model = Model([tensor1, tensor2], answer)
    optimizer = keras.optimizers.Adam(lr=0.007)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())

    model.fit([trainW1, trainW2], trainY,
              validation_split=0.1,
              epochs=10,
              batch_size=100)


    if not os.path.exists('../../models'):  # check if path exists
        print("creating models folder...")
        os.mkdir('../../models/')
    plot_model(model, to_file='../../models/model.png')

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    # saving tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    filename = 'Android-Universal-Image-Loader_methoddeclarations_train'
    main(filename)

