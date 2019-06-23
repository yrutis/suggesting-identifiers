import numpy as np
from keras import Model, Input
from keras.layers import Embedding, Dense, Flatten

from keras.models import Sequential
from keras.optimizers import Adam

# Parameters
from src.data.Datagenerator import DataGenerator

from src.data.prepare_data_token import main as prepare_data

all_trainX, all_trainY, all_valX, all_valY, vocab_size, window_size, \
training_processed_decoded, validation_processed_decoded = prepare_data("Android-Universal-Image-Loader", 8, 2, using_generator=True)

params = {'dim': window_size,
          'batch_size': 64,
          'shuffle': False}

# Datasets
partition_train = {'train': all_trainX}
partition_val = {'validation': all_valX}
print(all_valX[-1])


# Generators
training_generator = DataGenerator(all_trainX, training_processed_decoded, 'train', **params)
validation_generator = DataGenerator(all_valX, validation_processed_decoded, 'val', **params)

contextEmbedding = Embedding(input_dim=vocab_size, output_dim=20, input_length=window_size)

tensor = Input(shape=(window_size,))
c = contextEmbedding(tensor)
c = Flatten()(c)

#c = Dropout(self.config.model.dropout_1)(c)
c = Dense(20)(c)
#c = Dropout(self.config.model.dropout_2)(c)
answer = Dense(vocab_size, activation='softmax')(c)

model = Model(tensor, answer)
optimizer = Adam()
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['acc'])
print(model.summary())

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6,
                    epochs=2,
                    verbose=2)
