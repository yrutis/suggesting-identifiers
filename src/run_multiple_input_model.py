#%%
import keras
from keras import layers
from keras.optimizers import Adam
from keras.utils import plot_model
from src.data.prepare_data_multiple_input import main as prepare_data

#%% init some variables

filename = 'all_methods_train_without_platform'
window_size = 8

#%%
train_method_body_padded, train_parameters_padded, train_type_padded, trainY, \
val_method_body_padded, val_parameters_padded, val_type_padded, valY, \
tokenizer, perc_unk_train, perc_unk_val = prepare_data(filename, window_size)

#%%

word_index = tokenizer.word_index
print('Found {} unique tokens.'.format(len(word_index) + 1))
vocab_size = len(word_index) + 1

#%% construct model

method_body_input = keras.Input(shape=(None,), name='method_body')  # Variable-length sequence of ints
parameter_input = keras.Input(shape=(None,), name='parameters')  # Variable-length sequence of ints
return_type_input = keras.Input(shape=(None,), name='return_type_input')  # Variable-length sequence of ints

shared_embedding = layers.Embedding(vocab_size, 64, name='shared_embedding')

method_body_features = shared_embedding(method_body_input)
parameter_features = shared_embedding(parameter_input)
return_type_features = shared_embedding(return_type_input)


method_body_features = layers.LSTM(128)(method_body_features)
parameter_features = layers.LSTM(32)(parameter_features)
return_type_features = layers.LSTM(32)(return_type_features)


x = layers.concatenate([method_body_features, parameter_features, return_type_features])

method_name_pred = layers.Dense(vocab_size, activation='softmax', name='method_name')(x)


model = keras.Model(inputs=[method_body_input, parameter_input, return_type_input],
                    outputs=[method_name_pred])
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['acc'])
print(model.summary())

plot_model(model, to_file='model.png')

#%% train model

model.fit([train_method_body_padded, train_parameters_padded, train_type_padded], trainY,
          validation_data=([val_method_body_padded, val_parameters_padded, val_type_padded], valY),
          epochs=20,
          batch_size=128)
