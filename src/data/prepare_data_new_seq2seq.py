import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import src.data.utils.helper_functions as helper_functions
import logging
import os
import re


def split_camel_case_and_snake_case_target(y):

    regex = "(?!^)([A-Z][a-z]+)|_"  # split by camelCase and snake_case
    splitted_target = re.sub(regex, r' \1', y).split()
    splitted_target_lower = [x.lower() for x in splitted_target]  # make all lowercase
    splitted_target_seq = ['starttoken'] + splitted_target_lower + ['endtoken']
    return splitted_target_seq

def split_camel_case_and_snake_case_body(y):

    splitted_list = []
    for target in y:
        regex = "(?!^)([A-Z][a-z]+)|_"  # split by camelCase and snake_case
        splitted_target = re.sub(regex, r' \1', target).split()
        splitted_target_lower = [x.lower() for x in splitted_target]  # make all lowercase
        splitted_list += splitted_target_lower
    splitted_list = ['starttoken'] + splitted_list + ['endtoken']
    return splitted_list


def main(filename, window_size):
    # get logger
    logger = logging.getLogger(__name__)

    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    processed_decoded_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'processed'), 'decoded'),
                                               filename + '.json')  # get decoded path

    df = pd.read_json(processed_decoded_full_path, orient='records')

    # some basic operations: preprocessing parameters
    df['parameters'] = df['parameters'].apply(helper_functions.turn_all_to_lower)
    df['parameters'] = df['parameters'].apply(helper_functions.split_params)

    # some basic operations: preprocessing method body
    df['methodBody'] = df['methodBody'].apply(helper_functions.removeOptional)
    df["methodBody"] = df['methodBody'].apply(helper_functions.turn_strings_to_list)
    df["methodBody"] = df['methodBody'].apply(helper_functions.delete_certain_strings)

    #clean from function structure
    df['methodBodyCleaned'] = df['methodBody'].apply(helper_functions.clean_from_function_structure)

    #concat type, params, method body
    df["concatMethodBodyCleaned"] = df['Type'].map(lambda x: [x]) + df["parameters"] + df["methodBodyCleaned"]

    df['methodNameSplitted'] = df['methodName'].apply(split_camel_case_and_snake_case_target)
    df['methodBodySplitted'] = df['concatMethodBodyCleaned'].apply(split_camel_case_and_snake_case_body)
    df["methodBodySplitted"] = df['methodBodySplitted'].apply(helper_functions.turn_all_to_lower)

    print(df['methodNameSplitted'].head())
    print(df['methodBodySplitted'].head())

    # compute some statistics
    avg_mean_body_uncleaned = df['methodNameSplitted'].apply(helper_functions.compute_col_length).mean()
    print(avg_mean_body_uncleaned- 2) #start end token
    avg_mean_body_uncleaned = df['methodBodySplitted'].apply(helper_functions.compute_col_length).mean()
    print(avg_mean_body_uncleaned) #start end token



    x_train, x_test, y_train, y_test = train_test_split(df['methodBodySplitted'], df['methodNameSplitted'], test_size=0.2)
    method_body_cleaned_list_x = list(x_train)
    method_name_x = list(y_train)

    training_vocab_x = helper_functions.get_training_vocab(method_body_cleaned_list_x, is_for_x=True)
    training_vocab_y = helper_functions.get_training_vocab(method_name_x, is_for_x=True)

    x_train = list(map(helper_functions.get_into_tokenizer_format, method_body_cleaned_list_x))
    # print(x_train[:10])

    # fit on text the most common words from trainX and trainY
    tokenizer = Tokenizer(oov_token=True)
    # actual training data gets mapped on text
    tokenizer.fit_on_texts(training_vocab_y)  # actual training data gets mapped on text

    word_index = tokenizer.word_index
    print('Found {} unique Y tokens.'.format(len(word_index) + 1))

    tokenizer.fit_on_texts(training_vocab_x)

    word_index = tokenizer.word_index
    print('Found {} unique X+Y tokens.'.format(len(word_index) + 1))

    # tokenize just trainX
    vocab_size = len(word_index) + 1
    x_train_tokenized = tokenizer.texts_to_sequences(x_train)
    print(x_train_tokenized[:10])


    # tokenize just trainY
    y_train = list(y_train)
    print(y_train[:10])
    y_train_tokenized = tokenizer.texts_to_sequences(y_train)
    print(y_train_tokenized[:10])

    max_len_method = max([len(i) for i in y_train_tokenized])

    maxlen = 0
    longest_method = ""
    for i in y_train_tokenized:
        if len(i) >= maxlen:
            maxlen = len(i)
            longest_method = i
    print(longest_method)

    # Creating a reverse dictionary
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

    # Function takes a tokenized sentence and returns the words
    def sequence_to_text(list_of_indices):
        # Looking up words in dictionary
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        return (words)

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






    from keras.layers import Input, LSTM, Embedding, Dense
    from keras.models import Model
    from keras.utils import plot_model

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


if __name__ == '__main__':
    main("Android-Universal-Image-Loader", 8)