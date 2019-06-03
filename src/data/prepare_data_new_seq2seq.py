from datetime import datetime
from pickle import dump
from random import randint

import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import src.data.utils.helper_functions as helper_functions
import logging
import os
from matplotlib import pyplot as plt
import src.utils.path as path_file


#%%

def add_start_end_token(y):
    splitted_list = ['starttoken'] + y + ['endtoken']
    return splitted_list

#%%

#basic init
filename = 'all_methods_train_without_platform-subtoken'
# get logger
logger = logging.getLogger(__name__)

data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
processed_decoded_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'processed'), 'decoded'),
                                           filename + '.json')  # get decoded path

#%% load dataset

df = pd.read_json(processed_decoded_full_path, orient='records')

#%%

window_size_params = 4
window_size_body = 12

window_size_name = 3

max_input_elemts = 1 + window_size_params + window_size_body + 2 #return type + ... + ... + startendtoken
max_output_elemts = 2 + window_size_name #startendtoken + ...


df['parametersSplitted'] = df['parametersSplitted'].apply(helper_functions.get_first_x_elem, args=(window_size_params,))
df['methodBodySplitted'] = df['methodBodySplitted'].apply(helper_functions.get_first_x_elem, args=(window_size_body,))
df["concatMethodBodySplittedClean"] = df['Type'].map(lambda x: [x]) + df["parametersSplitted"] + df["methodBodySplitted"]

df['methodNameSplitted'] = df['methodNameSplitted'].apply(helper_functions.get_first_x_elem, args=(window_size_name,))


#%% add start end token
df['concatMethodBodySplittedClean'] = df['concatMethodBodySplittedClean'].apply(add_start_end_token)
df['methodNameSplitted'] = df['methodNameSplitted'].apply(add_start_end_token)

#%% split dataset
x_train, x_test, y_train, y_test = train_test_split(df['concatMethodBodySplittedClean'], df['methodNameSplitted'],
                                                    test_size=0.2,
                                                    random_state=200)
method_body_cleaned_list_x = list(x_train)
method_name_x = list(y_train)

#%%dataset in training vocab format

training_vocab_x = helper_functions.get_training_vocab(method_body_cleaned_list_x, is_for_x=True)
training_vocab_y = helper_functions.get_training_vocab(method_name_x, is_for_x=True)

x_train = list(map(helper_functions.get_into_tokenizer_format, method_body_cleaned_list_x))

#%%word2idx

# fit on text the most common words from trainX and trainY
tokenizer = Tokenizer(oov_token=True)
# actual training data gets mapped on text
tokenizer.fit_on_texts(training_vocab_y)  # actual training data gets mapped on text

word_index = tokenizer.word_index
print('Found {} unique Y tokens.'.format(len(word_index) + 1))

tokenizer.fit_on_texts(training_vocab_x)

word_index = tokenizer.word_index
print('Found {} unique X+Y tokens.'.format(len(word_index) + 1))
#%% idx2word

# Creating a reverse dictionary
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

# Function takes a tokenized sentence and returns the words
def sequence_to_text(list_of_indices):
    # Looking up words in dictionary
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return (words)

#%%

# tokenize just trainX
vocab_size = len(word_index) + 1
x_train_tokenized = tokenizer.texts_to_sequences(x_train)
print(x_train[:10])
print(x_train_tokenized[:10])
x_train_rev = list(map(sequence_to_text, x_train_tokenized))
print(x_train_rev[:10])


#%%

# tokenize just trainY
y_train = list(y_train)
print(y_train[:20])
y_train_tokenized = tokenizer.texts_to_sequences(y_train)
print(y_train_tokenized[:20])
y_train_rev = list(map(sequence_to_text, y_train_tokenized))
print(y_train_rev[:20])



#%%

print(len(y_train_tokenized), len(x_train_tokenized))

encoder_input_data = np.zeros(
    (len(x_train_tokenized), max_input_elemts),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(y_train_tokenized), max_output_elemts),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(y_train_tokenized), max_output_elemts, vocab_size),
    dtype='float32')

#%%


for i, (input_text, target_text) in enumerate(zip(x_train_tokenized, y_train_tokenized)):
    for t, word in enumerate(input_text):
        #20 is the maximum length
        if t < max_input_elemts:
            encoder_input_data[i, t] = input_text[t]

    for t, word in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = target_text[t]
        if t > 0:
            # decoder_target_data will be ahead by one timestep (t=0 is always start)
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_text[t]] = 1.

# print(encoder_input_data[:100])
print(decoder_input_data[:10])
print(decoder_target_data[:10])



#%% run model

from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model

e = Embedding(vocab_size, 64)
encoder_inputs = Input(shape=(None,), name="encoder_input")
en_x = e(encoder_inputs)
encoder = LSTM(50, return_state=True)
encoder_outputs, state_h, state_c = encoder(en_x)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,), name='decoder_input')
dex = e
final_dex = dex(decoder_inputs)

decoder_lstm = LSTM(50, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder_lstm(final_dex,
                                     initial_state=encoder_states)

decoder_dense = Dense(vocab_size, activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

print(model.summary())


history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=128,
          epochs=2,
          validation_split=0.05)


#%% create report folder
# create unique report folder
random_nr = randint(0, 10000)
unique_folder_key = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S') + "-" + str(random_nr)
report_folder = os.path.join(path_file.report_folder, 'reports-seq2seq-' + unique_folder_key)

os.mkdir(report_folder)


#%% save model+ tokenizer
model.save(os.path.join(report_folder, 's2s.h5'))
dump(tokenizer, open(os.path.join(report_folder,'tokenizer.pkl'), 'wb'))


#%%
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    start_token_idx = tokenizer.texts_to_sequences(['starttoken'])
    start_token_idx_elem = start_token_idx[0][0]
    target_seq[0, 0] = start_token_idx_elem

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = sequence_to_text([sampled_token_index])
        sampled_char = str(sampled_char[0])  #in case of true which is oov

        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == 'endtoken' or
           len(decoded_sentence) > max_output_elemts):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

#%%

# Encode the input sequence to get the "thought vectors"
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(50,))
decoder_state_input_c = Input(shape=(50,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2= dex(decoder_inputs) # Get the embeddings of the decoder sequence

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2) # A dense softmax layer to generate prob dist. over the target vocabulary

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)


#%% generate some method names
i = 0
correct = 0
amnt = 10
while i < amnt:
    input_seq = encoder_input_data[i: i+1]
    input_seq_list = input_seq.tolist()[0] #get in right format for tokenizer
    correct_output = decoder_input_data[i: i + 1]
    correct_output_list = correct_output.tolist()[0] #get in right format for tokenizer
    decoded_correct_output_list = sequence_to_text(correct_output_list)

    input_enc = sequence_to_text(input_seq_list)


    print("this is the input seq decoded: {}".format(input_enc))
    decoded_sentence = decode_sequence(input_seq)
    print("Predicted: {}".format(decoded_sentence))
    print("Correct: {}".format(decoded_correct_output_list))
    i += 1

    if decoded_sentence == decoded_correct_output_list:
        correct += 1

accuracy = correct/amnt
print("total accuracy %.2f%%" %accuracy)


#%%

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(os.path.join(report_folder, "acc_plot.png"))
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(os.path.join(report_folder, "loss_plot.png"))


#%%

# summarize model
plot_model(encoder_model, to_file=os.path.join(report_folder, 'encoder_model.png'), show_shapes=True)
plot_model(decoder_model, to_file=os.path.join(report_folder, 'decoder_model.png'), show_shapes=True)
plot_model(model, to_file=os.path.join(report_folder, 'model.png'), show_shapes=True)