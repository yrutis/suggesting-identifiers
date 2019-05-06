import os
import logging
import ast
import pickle

import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



class Preprocessor(object):
    def __init__(self, filename, max_words=None):
        self.__filename = filename
        self.trainX = None
        self.trainY = None
        self.valX = None
        self.valY = None
        self.tokenizer = None
        self.max_context_vocab_size = max_words
        self.encoder = None


    def preprocess(self):
        pass


    def tokenize(self):
        logger = logging.getLogger(__name__)

        #move levels up to reach data folder
        data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
        processed_decoded_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'processed'), 'decoded'),
                                                   self.__filename + '.csv')  # get decoded path

        processedDf = pd.read_csv(processed_decoded_full_path)

        context = processedDf['x'].apply(ast.literal_eval)  # saves all context x as list in list


        self.tokenizer = Tokenizer(num_words=self.max_context_vocab_size)  # init new tokenizer
        self.tokenizer.fit_on_texts(context)
        sequences = self.tokenizer.texts_to_sequences(context)
        padded_sequences = pad_sequences(sequences, maxlen=None, value=0)  # make sure they are all same length

        if not self.max_context_vocab_size:
            self.max_context_vocab_size = len(self.tokenizer.word_index) + 1
            logger.info('Found %s unique tokens.' % self.max_context_vocab_size)
        else:
            logger.info("only considering the topmost {} words" .format(self.max_context_vocab_size))


        # load Y's
        y = processedDf['y']  # get all Y
        Y = y.values  # convert to numpy
        logger.info("first load y: {}".format(Y[0]))

        # encode class values as integers
        self.encoder = LabelEncoder()
        self.encoder.fit(Y)
        encoded_Y = self.encoder.transform(Y)

        # amount of unique Y's
        lenY = len(np.unique(Y))


        self.trainX, self.valX, self.trainY, self.valY = train_test_split(padded_sequences, encoded_Y, test_size = 0.2)


        self.trainY = to_categorical(self.trainY, num_classes=lenY)

        logger.info("this is lenY {} unique, this is shape of categorical Y {}" .format(lenY, self.trainY.shape))

        # saving tokenizer
        #with open('tokenizer.pickle', 'wb') as handle:
        #    pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def reverse_tokenize(self, sequence):

        if not self.tokenizer:
            Exception("You need to create a tokenizer first!")

        # Creating a reverse dictionary
        reverse_word_map = dict(map(reversed, self.tokenizer.word_index.items()))

        # Function takes a tokenized sentence and returns the words
        def sequence_to_text(list_of_indices):
            # Looking up words in dictionary
            words = [reverse_word_map.get(letter) for letter in list_of_indices]
            return (words)

        # Creating reversed text
        reversed_text = list(map(sequence_to_text, sequence))

        return reversed_text
