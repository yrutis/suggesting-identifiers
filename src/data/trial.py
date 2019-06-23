import os

import numpy as np
from keras import Model, Input
from keras.layers import Embedding, Dense, Flatten, LSTM

from keras.models import Sequential
from keras.optimizers import Adam

# Parameters
from src.data.DataGeneratorSubtoken import DataGenerator

from src.data.prepare_data_subtoken import main as prepare_data

report_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')

all_train, all_val, vocab_size, max_input_elemts, max_output_elemts, data_storage = \
    prepare_data("Android-Universal-Image-Loader", 8, 2, 3, report_folder=report_folder, using_generator=True)

params = {'input_dim': max_input_elemts,
          'output_dim': max_output_elemts,
          'batch_size': 64,
          'shuffle': False}

# Generators
training_generator = DataGenerator(all_train, data_storage, 'train', vocab_size, **params)
validation_generator = DataGenerator(all_val, data_storage, 'val', vocab_size, **params)

e = Embedding(vocab_size, 20)
encoder_inputs = Input(shape=(None,), name="encoder_input")
en_x = e(encoder_inputs)
encoder = LSTM(20, return_state=True)
encoder_outputs, state_h, state_c = encoder(en_x)
encoder_states = [state_h, state_c]
decoder_inputs = Input(shape=(None,), name='decoder_input')
dex = e
final_dex = dex(decoder_inputs)
decoder_lstm = LSTM(20, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(final_dex, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())


# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6,
                    epochs=2,
                    verbose=2)
