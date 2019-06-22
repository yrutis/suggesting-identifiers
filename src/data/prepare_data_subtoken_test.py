import pandas as pd
import numpy as np
import logging
import os
from src.data.utils import helper_functions


def add_start_end_token(y):
    _list = ['starttoken'] + y + ['endtoken']
    return _list

#%%

def main(filename, tokenizer, window_size_body, window_size_params, window_size_name):
    #basic init

    # get logger
    logger = logging.getLogger(__name__)

    filename += '-processed'

    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    test_processed_decoded_full_path = os.path.join(
        os.path.join(os.path.join(os.path.join(os.path.join(data_folder, 'processed'),
                                               'decoded'), filename), 'test'), filename + '-subtoken.json')

    df = pd.read_json(test_processed_decoded_full_path, orient='records')

    max_input_elemts = 1 + window_size_params + window_size_body + 2  # return type + ... + ... + startendtoken
    max_output_elemts = 2 + window_size_name  # startendtoken + ...

    df['parameters'] = df['parameters'].apply(helper_functions.get_first_x_elem, args=(window_size_params,))
    df['methodBody'] = df['methodBody'].apply(helper_functions.get_first_x_elem, args=(window_size_body,))
    df["concatMethodBodyClean"] = df['Type'].map(lambda x: [x]) + df["parameters"] + df["methodBody"]

    df['methodName'] = df['methodName'].apply(helper_functions.get_first_x_elem, args=(window_size_name,))

    # %% add start end token
    df['concatMethodBodyClean'] = df['concatMethodBodyClean'].apply(add_start_end_token)
    df['methodName'] = df['methodName'].apply(add_start_end_token)

    # %% split dataset
    x_test, y_test = df['concatMethodBodyClean'], df['methodName']
    method_body_cleaned_list_x = list(x_test)
    method_name_x = list(y_test)

    # %%dataset in training vocab format

    x_test = list(map(helper_functions.get_into_tokenizer_format, method_body_cleaned_list_x))

    # %% idx2word

    # Creating a reverse dictionary
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

    # Function takes a tokenized sentence and returns the words
    def sequence_to_text(list_of_indices):
        # Looking up words in dictionary
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        return (words)

    word_index = tokenizer.word_index

    # tokenize just trainX
    vocab_size = len(word_index) + 1
    x_test_tokenized = tokenizer.texts_to_sequences(x_test)
    print(x_test[:10])
    print(x_test_tokenized[:10])
    x_test_rev = list(map(sequence_to_text, x_test_tokenized))
    print(x_test_rev[:10])

    # tokenize just trainY
    y_test = list(y_test)
    print(y_test[:20])
    y_test_tokenized = tokenizer.texts_to_sequences(y_test)
    print(y_test_tokenized[:20])
    y_test_rev = list(map(sequence_to_text, y_test_tokenized))
    print(y_test_rev[:20])


    encoder_input_data = np.zeros(
        (len(x_test_tokenized), max_input_elemts),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(y_test_tokenized), max_output_elemts),
        dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(x_test_tokenized, y_test_tokenized)):
        for t, word in enumerate(input_text):
            # max_input_elements is the maximum length
            if t < max_input_elemts:
                encoder_input_data[i, t] = input_text[t]

        for t, word in enumerate(target_text):
            decoder_input_data[i, t] = target_text[t]

    return encoder_input_data, decoder_input_data

