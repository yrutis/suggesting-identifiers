import pickle
from keras import Input, Model
from keras.engine.saving import load_model
import pandas as pd
import numpy as np
import logging
import os

#%%

# loading
from sklearn.model_selection import train_test_split
from src.data.utils import helper_functions

with open('reports-seq2seq/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)


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
df['methodBodySplitted'] = df['methodBodySplitted'].apply(add_start_end_token)
df['methodNameSplitted'] = df['methodNameSplitted'].apply(add_start_end_token)


#%% split dataset
x_train, x_test, y_train, y_test = train_test_split(df['methodBodySplitted'], df['methodNameSplitted'], test_size=0.2)
method_body_cleaned_list_x = list(x_train)
method_name_x = list(y_train)

#%%dataset in training vocab format


x_train = list(map(helper_functions.get_into_tokenizer_format, method_body_cleaned_list_x))


#%% idx2word

# Creating a reverse dictionary
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

# Function takes a tokenized sentence and returns the words
def sequence_to_text(list_of_indices):
    # Looking up words in dictionary
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return (words)

#%%

word_index = tokenizer.word_index

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

encoder_input_data = np.zeros(
    (len(x_train_tokenized), max_input_elemts),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(y_train_tokenized), max_output_elemts),
    dtype='float32')


for i, (input_text, target_text) in enumerate(zip(x_train_tokenized, y_train_tokenized)):
    for t, word in enumerate(input_text):
        #20 is the maximum length
        if t < 20:
            encoder_input_data[i, t] = input_text[t]

    for t, word in enumerate(target_text):
        decoder_input_data[i, t] = target_text[t]


#%%
#make model ready
model = load_model('reports-seq2seq/s2s.h5')

print(model.layers)
print("first layer {}".format(model.layers[1]))
print("second layer {}".format(model.layers[2]))

latent_dim = 50
encoder_inputs = model.input[0]   # input_1
dex = model.layers[2]

encoder_outputs, state_h_enc, state_c_enc = model.layers[3].output   # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]   # input_2
decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]


dec_emb2= dex(decoder_inputs) # Get the embeddings of the decoder sequence


decoder_lstm = model.layers[4]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    dec_emb2, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]


decoder_dense = model.layers[5]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)




#%% decoder setup

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


def perSubtokenStatistics(results):
    #check if in vocabulary
    complete_true = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for correct, predicted in results:
        if ''.join(correct) == ''.join(predicted):
            true_positive += len(correct)
            complete_true += 1
            continue

        for subtok in predicted:
            if subtok in correct:
                true_positive += 1
            else:
                false_positive += 1
        for subtok in correct:
            if not subtok in predicted:
                false_negative += 1

    return complete_true, true_positive, false_positive, false_negative

def calculate_results(complete_true, total, true_positive, false_positive, false_negative):
    accuracy = 0
    if total != 0:
        accuracy = complete_true / total
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return accuracy, precision, recall, f1

#%%

def filter_results(subtoken_list):
    subtoken_list = list(filter(None, subtoken_list))
    subtoken_list = [str(x) for x in subtoken_list]
    subtoken_list = list(filter(lambda x: x != "starttoken", subtoken_list))
    subtoken_list = list(filter(lambda x: x != "endtoken", subtoken_list))
    subtoken_list = list(filter(lambda x: x != "True", subtoken_list)) #oov
    return subtoken_list


#%% generate some method names
input = []
predictions = []
correct = []
i = 0
while i < 100:
    input_seq = encoder_input_data[i: i+1]
    correct_output = decoder_input_data[i: i+1]
    correct_output_list = correct_output.tolist()[0]
    decoded_correct_output_list = sequence_to_text(correct_output_list)
    input_seq_list = input_seq.tolist()[0] #get in right format for tokenizer

    #input
    input_enc = sequence_to_text(input_seq_list)
    print("Input: {}".format(input_enc))
    input.append(input_enc)

    #prediction
    decoded_sentence = decode_sequence(input_seq)
    decoded_sentence_list = decoded_sentence.split()
    print("Prediction List: {}".format(decoded_sentence_list))
    predictions.append(decoded_sentence_list)


    #correct
    print("Correct: {}".format(decoded_correct_output_list))
    correct.append(decoded_correct_output_list)

    i += 1

print(predictions)
print(correct)
print(len(correct))
print(len(predictions))

predictions = list(map(filter_results, predictions))
correct = list(map(filter_results, correct))
print(predictions)
print(correct)

complete_true, true_positive, false_positive, false_negative = perSubtokenStatistics(zip(correct, predictions))
print(complete_true, true_positive, false_positive, false_negative)
total = len(correct)
accuracy, precision, recall, f1 = calculate_results(complete_true, total, true_positive, false_positive, false_negative)
print((accuracy, precision, recall, f1))
