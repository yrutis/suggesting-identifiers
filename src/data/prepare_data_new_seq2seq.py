from pickle import dump

import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import src.data.utils.helper_functions as helper_functions
import logging
import os

#%%

def add_start_end_token(y):
    splitted_list = ['starttoken'] + y + ['endtoken']
    return splitted_list

#%%

#basic init
filename = 'Android-Universal-Image-Loader-subtoken'
# get logger
logger = logging.getLogger(__name__)

data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
processed_decoded_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'processed'), 'decoded'),
                                           filename + '.json')  # get decoded path

#%% load dataset

df = pd.read_json(processed_decoded_full_path, orient='records')

#%% add start end token
df['methodBodySplitted'] = df['methodBodySplitted'].apply(add_start_end_token)
df['methodNameSplitted'] = df['methodNameSplitted'].apply(add_start_end_token)

#%% split dataset
x_train, x_test, y_train, y_test = train_test_split(df['methodBodySplitted'], df['methodNameSplitted'], test_size=0.2,
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

#get longest method name TODO remove method names longer than ...
max_len_method = max([len(i) for i in y_train_tokenized])

maxlen = 0
longest_method = ""
for i in y_train_tokenized:
    if len(i) >= maxlen:
        maxlen = len(i)
        longest_method = i
print(longest_method)

#%%


print(sequence_to_text(longest_method))
print(max_len_method)
print(len(y_train_tokenized), len(x_train_tokenized))


encoder_input_data = np.zeros(
    (len(x_train_tokenized), 20),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(y_train_tokenized), max_len_method),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(y_train_tokenized), max_len_method, vocab_size),
    dtype='float32')

#%%


for i, (input_text, target_text) in enumerate(zip(x_train_tokenized, y_train_tokenized)):
    for t, word in enumerate(input_text):
        #20 is the maximum length
        if t < 20:
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

e = Embedding(vocab_size, 10)
encoder_inputs = Input(shape=(None,))
en_x = e(encoder_inputs)
encoder = LSTM(50, return_state=True)
encoder_outputs, state_h, state_c = encoder(en_x)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
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


model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=128,
          epochs=2,
          validation_split=0.05)

#%% save model+ tokenizer
model.save('s2s.h5')
dump(tokenizer, open('tokenizer.pkl', 'wb'))


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
           len(decoded_sentence) > 52):
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
while i < 10:
    input_seq = encoder_input_data[i: i+1]
    input_seq_list = input_seq.tolist()[0] #get in right format for tokenizer

    print("this is the input seq decoded: {}".format(input_seq_list))
    input_enc = sequence_to_text(input_seq_list)
    print("this is the input seq encoded: {}".format(input_enc))
    decoded_sentence = decode_sequence(input_seq)
    print("this is the output seq decoded: {}".format(decoded_sentence))
    i += 1