import os

import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, data_folder, type, vocab_size,
                 batch_size=32, input_dim=11, output_dim=4, shuffle=True):
        'Initialization'
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.data_folder = data_folder
        self.vocab_size = vocab_size
        self.type = type
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # print("this are the indexes {}".format(indexes))

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty([self.batch_size, self.input_dim])
        y = np.empty((self.batch_size), dtype=int)

        encoder_input_data = np.empty(
            (self.batch_size, self.input_dim),
            dtype='float32')
        decoder_input_data = np.empty(
            (self.batch_size, self.output_dim),
            dtype='float32')
        decoder_target_data = np.empty(
            (self.batch_size, self.output_dim, self.vocab_size),
            dtype='float32')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
             # Store sample
            encoder_input_data[i,] = np.load(os.path.join(self.data_folder, str(self.type) + 'X1-' + str(ID) + '.npy'))

            decoder_input_data[i,] = np.load(os.path.join(self.data_folder, str(self.type) + 'X2-' + str(ID) + '.npy'))

            decoder_target_data[i,] = np.load(os.path.join(self.data_folder, str(self.type) + 'Y-' + str(ID) + '.npy'))

            # Store class
            #y[i] = np.load(os.path.join(self.data_folder, str(self.type) + 'Y-' + str(ID) + '.npy'))

        X = [encoder_input_data, decoder_input_data]
        y = decoder_target_data

        return X, y