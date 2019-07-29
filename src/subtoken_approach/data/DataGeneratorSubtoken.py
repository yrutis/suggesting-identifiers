import os

import numpy as np
import keras
import logging


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, data_folder, type, vocab_size, partition,
                 batch_size=32, input_dim=11, output_dim=4, shuffle=True):
        'Initialization'
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.data_folder = data_folder
        self.partition = partition
        self.vocab_size = vocab_size
        self.type = type
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.index = 0
        self.counter = 0
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        #print("number of batches per epoch {}".format(int(np.floor(len(self.list_IDs) / self.batch_size))))
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[self.index*self.batch_size:(self.index+1)*self.batch_size]
        self.index += 1
        #print("this are the indexes {}".format(indexes))

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.index = 0
        self.counter = 0
        self.indexes = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # get logger
        logger = logging.getLogger(__name__)

        #logger.info("creating the batch tensors...")

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

            if ID % self.partition == 0:
                self.current_partion_x1 = None
                self.current_partion_x2 = None
                self.current_partion_y = None

                current_partion_x1_name = os.path.join(self.data_folder, str(self.type) + 'X1-' + str(self.counter) + '.npy')
                current_partion_x2_name = os.path.join(self.data_folder, str(self.type) + 'X2-' + str(self.counter) + '.npy')
                current_partion_y_name = os.path.join(self.data_folder, str(self.type) + 'Y-' + str(self.counter) + '.npy')

                #logger.info("loading current_partion_x1 {}".format(current_partion_x1_name))
                self.current_partion_x1 = np.load(
                    os.path.join(self.data_folder, str(self.type) + 'X1-' + str(self.counter) + '.npy'))
                #logger.info("{} bytes" .format(self.current_partion_x1.size * self.current_partion_x1.itemsize))


                #logger.info("loading current_partion_x2 {}".format(current_partion_x2_name))
                self.current_partion_x2 = np.load(
                    os.path.join(self.data_folder, str(self.type) + 'X2-' + str(self.counter) + '.npy'))
                #logger.info("{} bytes" .format(self.current_partion_x2.size * self.current_partion_x2.itemsize))


                #logger.info("loading current_partion_y {}".format(current_partion_y_name))
                self.current_partion_y = np.load(
                    os.path.join(self.data_folder, str(self.type) + 'Y-' + str(self.counter) + '.npy'))
                 #logger.info("{} bytes" .format(self.current_partion_y.size * self.current_partion_y.itemsize))



                self.counter += 1

            assert (self.current_partion_x1.shape[1] == self.input_dim)
            assert (self.current_partion_x2.shape[1] == self.output_dim)

            # print("i: {}, ID: {}".format(i, ID))
            # print("numpy shape x {}, y {}".format(self.current_partion_x.shape[0], self.current_partion_y.shape[0]))
            # print("trying to access position {} to {} from {}".format((ID % partition), ((ID % partition) + 1), self.current_partion_x.shape[0]))

             # Store sample
            encoder_input_data[i,] = self.current_partion_x1[(ID % self.partition):(ID % self.partition) + 1]
            #encoder_input_data[i,] = np.load(os.path.join(self.data_folder, str(self.type) + 'X1-' + str(ID) + '.npy'))

            decoder_input_data[i,] = self.current_partion_x2[(ID % self.partition):(ID % self.partition) + 1]
            #decoder_input_data[i,] = np.load(os.path.join(self.data_folder, str(self.type) + 'X2-' + str(ID) + '.npy'))

            decoder_target_data[i,] = self.current_partion_y[(ID % self.partition):(ID % self.partition) + 1]
            #decoder_target_data[i,] = np.load(os.path.join(self.data_folder, str(self.type) + 'Y-' + str(ID) + '.npy'))

        #print("AFTER idx {} type {}, list of Ids: {}".format(self.index, self.type, list_IDs_temp))
        #print("first X: {}".format(X[0]))
        X = [encoder_input_data, decoder_input_data]
        y = decoder_target_data

        return X, y