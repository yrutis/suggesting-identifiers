import os

import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data_folder, type, partition, batch_size=32, dim=11, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.data_folder = data_folder
        self.partition = partition
        self.type = type
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.index = 0
        self.counter = 0
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        print(int(np.floor(len(self.list_IDs) / self.batch_size)))
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
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros([self.batch_size, self.dim])
        y = np.zeros((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            if ID % self.partition == 0:
                self.current_partion_x = np.load(
                    os.path.join(self.data_folder, str(self.type) + 'X-' + str(self.counter) + '.npy'))
                self.current_partion_y = np.load(
                    os.path.join(self.data_folder, str(self.type) + 'Y-' + str(self.counter) + '.npy'))
                self.counter += 1

            assert(self.current_partion_x.shape[1] == self.dim)

            #print("i: {}, ID: {}".format(i, ID))
            #print("numpy shape x {}, y {}".format(self.current_partion_x.shape[0], self.current_partion_y.shape[0]))
            #print("trying to access position {} to {} from {}".format((ID % partition), ((ID % partition) + 1), self.current_partion_x.shape[0]))

            # Store sample

            X[i,] = self.current_partion_x[(ID % self.partition):(ID % self.partition) + 1]

            # Store class
            y[i] = self.current_partion_y[(ID % self.partition):(ID % self.partition) + 1]

        #print("AFTER idx {} type {}, list of Ids: {}".format(self.index, self.type, list_IDs_temp))
        #print("first X: {}".format(X[0]))
        #print("y: {}".format(y))


        return X, y