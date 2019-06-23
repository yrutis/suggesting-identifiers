import os

import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data_folder, type, batch_size=32, dim=11, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.data_folder = data_folder
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
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #print("this are the indexes {}".format(indexes))

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
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty([self.batch_size, self.dim])
        y = np.empty((self.batch_size), dtype=int)

        smallest_idx = min(list_IDs_temp)
        max_idx = max(list_IDs_temp)



        #print("first element of list id temp {}, last element {}".format(list_IDs_temp[0], list_IDs_temp[-1]))
        #print("amount of steps per epoch {}".format(int(np.floor(len(self.list_IDs) / self.batch_size))))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            name = os.path.join(self.data_folder, str(self.type) + 'X-' + str(ID) + '.npy')
            current = np.load(os.path.join(self.data_folder, str(self.type) + 'X-' + str(ID) + '.npy'))
            # Store sample
            X[i,] = np.load(os.path.join(self.data_folder, str(self.type) + 'X-' + str(ID) + '.npy'))

            # Store class
            y[i] = np.load(os.path.join(self.data_folder, str(self.type) + 'Y-' + str(ID) + '.npy'))

        return X, y