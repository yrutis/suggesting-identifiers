#%%
from datetime import datetime
from random import randint

from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import plot_model
from src.data.prepare_data_multiple_input import main as prepare_data
import src.utils.path as path_file

from matplotlib import pyplot as plt
from sklearn import metrics

import keras
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from src.trainer.Callbacks.Callback_3_inputs import Histories



#%% init some variables

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data', 'Android-Universal-Image-Loader',
                           'must be either Android-Universal-Image-Loader or all_methods_train')
tf.app.flags.DEFINE_integer('window_size', 8, 'must be between 2+')

filename = FLAGS.data
window_size = FLAGS.window_size

#%%
train_method_body_padded, train_parameters_padded, train_type_padded, trainY, \
val_method_body_padded, val_parameters_padded, val_type_padded, valY, \
tokenizer, perc_unk_train, perc_unk_val = prepare_data(filename, window_size)

#%%

word_index = tokenizer.word_index
print('Found {} unique tokens.'.format(len(word_index) + 1))
vocab_size = len(word_index) + 1


#%%
#create unique report folder
random_nr = randint(0, 10000)
unique_folder_key = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S') + "-" + str(random_nr)
report_folder = path_file.report_folder
report_folder_multiple_input = os.path.join(report_folder, 'reports-multiple-input-flatten-' + unique_folder_key)

os.mkdir(report_folder_multiple_input)

#%% callback, early stopping, checkpoint saving


histories = Histories(report_folder_multiple_input, tokenizer)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
mc = ModelCheckpoint(os.path.join(report_folder_multiple_input, "best_model.h5"), monitor='val_acc', mode='max', verbose=1,
                          save_best_only=True)

#%% construct model


method_body_input = keras.Input(shape=(window_size,), name='method_body')  # Variable-length sequence of ints
parameter_input = keras.Input(shape=(window_size,), name='parameters')  # Variable-length sequence of ints
return_type_input = keras.Input(shape=(window_size,), name='type')  # Variable-length sequence of ints

shared_embedding = layers.Embedding(vocab_size, 128, name='shared_embedding', input_length=window_size)

method_body_features = shared_embedding(method_body_input)
parameter_features = shared_embedding(parameter_input)
return_type_features = shared_embedding(return_type_input)


method_body_features = layers.Flatten()(method_body_features)
parameter_features = layers.Flatten()(parameter_features)
return_type_features = layers.Flatten()(return_type_features)


x = layers.concatenate([method_body_features, parameter_features, return_type_features])
x = layers.Dense(70, activation='elu', name='intermediate')(x)

method_name_pred = layers.Dense(vocab_size, activation='softmax', name='method_name')(x)


model = keras.Model(inputs=[method_body_input, parameter_input, return_type_input],
                    outputs=[method_name_pred])
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['acc'])
print(model.summary())

plot_model(model, to_file=os.path.join(report_folder_multiple_input,'model.png'))

#%% train model

trainX = {'method_body': train_method_body_padded,   # trainX
     'parameters': train_parameters_padded,
     'type': train_type_padded}

valX = {'method_body': val_method_body_padded,   # valX
     'parameters': val_parameters_padded,
     'type': val_type_padded}

history = model.fit(trainX, trainY,
          validation_data=(valX, valY),
          epochs=20,
          batch_size=128,
          verbose=0,
          callbacks=[histories, es, mc])


## to be refactored
#%% plots

acc_plot = os.path.join(report_folder_multiple_input, 'acc.png')
loss_plot = os.path.join(report_folder_multiple_input, 'loss.png')
acc_loss = os.path.join(report_folder_multiple_input, 'acc_loss.csv')

acc = history.history["acc"]
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

#hack
always_unknown_train_list = []
for x in epochs:
    always_unknown_train_list.append(perc_unk_train)

always_unknown_test_list = []
for x in epochs:
    always_unknown_test_list.append(perc_unk_val)


# save model data
model_data = {'acc': acc,
              'val_acc': val_acc,
              'unk_acc': always_unknown_train_list,
              'unk_val_acc': always_unknown_test_list,
              'loss': loss,
              'val_loss': val_loss}

df = pd.DataFrame(model_data, columns=['acc', 'val_acc', 'unk_acc', 'unk_val_acc', 'loss', 'val_loss'])
df.to_csv(acc_loss)


plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.plot(epochs, always_unknown_train_list, 'go', label='Unknown Training acc')
plt.plot(epochs, always_unknown_test_list, 'g', label='Unknown Test Acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(acc_plot)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(loss_plot)

#%% precision recall

predictions = model.predict(valX)  # get all predictions
predicted_classes = np.argmax(predictions, axis=1)
#predicted_classes = list(map(lambda x: [x], predicted_classes))

#get all possible target names

# Creating a reverse dictionary
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

# Function takes a tokenized sentence and returns the words
def sequence_to_text(list_of_indices):
    # Looking up words in dictionary
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return (words)

target_names = np.unique(np.append(predicted_classes, valY)).tolist()
target_names = list(map(lambda x: [x], target_names))
target_names = list(map(sequence_to_text, target_names))
target_names = list(map(lambda x: x[0], target_names))


report = metrics.classification_report(valY, predicted_classes, target_names=target_names, output_dict=True)
df = pd.DataFrame(report).transpose()


sklearn_report = os.path.join(report_folder_multiple_input, "report.csv")
df.to_csv(sklearn_report)

