# -*- coding: utf-8 -*-
import click
import logging
import os
import json
import zipfile
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv

import numpy as np

import keras

from keras import Input
from keras import layers

from keras.callbacks import TensorBoard
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.models import Model
from keras.utils import plot_model
from keras.utils import to_categorical

#import src.source.make_source as make_source
#import src.data.make_dataset as make_data
#import src.features.build_features as build_features

# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main(filename):
    """ runs model
    """
    def createModel(windowSize):
        print("creating model for window size: {}".format(windowSize))

        onlyTensorList = []
        buildModelList = []
        # adding the amount of tensors to the list need and build model
        i = 0
        contextEmbedding = Embedding(output_dim=50, input_dim=vocabSize, input_length=1)
        while i < windowSize:
            tensor = Input(shape=(1,), dtype='int32', )
            onlyTensorList.append(tensor)
            c = contextEmbedding(tensor)
            c = Flatten()(c)
            c = keras.layers.Dense(vocabSize)(c)
            buildModelList.append(c)
            i += 1

        added = keras.layers.Add()(buildModelList)
        added = Dropout(0.2)(added)
        answer = layers.Dense(vocabSize, activation='softmax')(added)
        return onlyTensorList, answer

    def createInputContext(trainX):
        allContextList = []
        windowSize = trainX.shape[1]
        # get all context and save in allContextList
        i = 0
        while i < windowSize:
            allContextList.append(trainX[:, i])
            i += 1
        return allContextList


    with open('../../data/processed/encoded/'+filename+'-tokenizer.json', 'r') as fp:
        tokenIndex = json.load(fp)

    with open('../../data/processed/encoded/'+filename+'.json', 'r') as fp:
        data = json.load(fp)
        trainX = list(data['x'].values()) #retrieve all x values as a list of list
        trainY = list(data['y'].values()) #retrieve all y values as a list of strings

    #converting trainX and trainY to a numpy array
    trainX = np.array(trainX)
    trainY = np.array(trainY)

    # get total length of tokens
    vocabSize = len(tokenIndex) + 1

    allContextList = createInputContext(trainX)
    print(allContextList)
    tryingOnlyTensors, tryingTarget = createModel(trainX.shape[1])

    # one hot encode outputs
    y = to_categorical(trainY, num_classes=vocabSize)

    model = Model(tryingOnlyTensors, tryingTarget)
    optimizer = keras.optimizers.Adam(lr=0.007)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())

    history = model.fit(allContextList, y, epochs=3, batch_size=100, validation_split=0.1)

    plot_model(model, to_file='../../models/model.png')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automatically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    filename = 'bigbluebutton_methoddeclarations_train'

    #download data
    #make_source.main()

    #create dataset
    #make_data.main(filename)

    #create ready to feed into model file
    #build_features.main(filename)

    #run model
    main(filename)

