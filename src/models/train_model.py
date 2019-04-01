# -*- coding: utf-8 -*-
import logging

from sklearn.preprocessing import LabelEncoder


from keras import Input
from keras import layers

from keras.layers import Embedding
from keras.layers import Flatten
from keras.models import Model
from keras.utils import plot_model
from keras.utils import to_categorical

import pandas as pd
import numpy as np
import ast

import keras
from keras.preprocessing.text import Tokenizer

def main(filename):
    """ runs model
    """
    processed_decoded_full_path = '../../data/processed/decoded/' + filename + '.csv'

    processedDf = pd.read_csv(processed_decoded_full_path)

    context = []
    for index, row in processedDf.iterrows():
        x = ast.literal_eval(row['x'])  # convert string representation back to list
        context.append(x)

    tokenizer = Tokenizer()  # init new tokenizer
    tokenizer.fit_on_texts(context)
    sequences = tokenizer.texts_to_sequences(context)
    padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None,
                                                                  value=0)  # make sure they are all same length

    print(padded_sequences.shape)
    print(padded_sequences[:, 0].shape)
    print("first padded sequence: {}".format(padded_sequences[0]))

    w1 = padded_sequences[:, 0]
    w2 = padded_sequences[:, 1]

    # load Y's
    dataset = processedDf.values
    Y = dataset[:, 3]

    # print("this is y {}" .format(Y))

    contextVocabSize = len(tokenizer.word_index) + 1
    print('Found %s unique tokens.' % contextVocabSize)

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    lenY = len(np.unique(Y))  # amount of unique Y's
    y = to_categorical(encoded_Y, num_classes=lenY)

    print("this is the shape of Y {}".format(y.shape))

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

    history = model.fit([w1, w2], y, epochs=10, batch_size=100, validation_split=0.1)

    plot_model(model, to_file='../../models/model.png')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    filename = 'Android-Universal-Image-Loader_methoddeclarations_train'
    main(filename)

