import pickle
from keras import Input, Model
from keras.engine.saving import load_model
import pandas as pd
import numpy as np
import logging
import os

# loading
from sklearn.model_selection import train_test_split
from src.data.utils import helper_functions


def add_start_end_token(y):
    _list = ['starttoken'] + y + ['endtoken']
    return _list

#%%

def main(filename, dictionary_path, window_size_body, window_size_params, window_size_name):
    #basic init
    filename = 'Android-Universal-Image-Loader-subtoken'
    with open(dictionary_path+'/tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # get logger
    logger = logging.getLogger(__name__)

    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    processed_decoded_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'processed'), 'decoded'),
                                               filename + '.json')  # get decoded path

    df = pd.read_json(processed_decoded_full_path, orient='records')

    max_input_elemts = 1 + window_size_params + window_size_body + 2  # return type + ... + ... + startendtoken
    max_output_elemts = 2 + window_size_name  # startendtoken + ...

    df['parameters'] = df['parameters'].apply(helper_functions.get_first_x_elem, args=(window_size_params,))
    df['methodBody'] = df['methodBody'].apply(helper_functions.get_first_x_elem, args=(window_size_body,))
    df["concatMethodBodyClean"] = df['Type'].map(lambda x: [x]) + df["parameters"] + df["methodBody"]

    df['methodName'] = df['methodName'].apply(helper_functions.get_first_x_elem, args=(window_size_name,))

    # %% add start end token
    df['methodBody'] = df['methodBody'].apply(add_start_end_token)
    df['methodName'] = df['methodName'].apply(add_start_end_token)

    # %% split dataset
    x_test, y_test = df['methodBody'], df['methodName']
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
            # 20 is the maximum length
            if t < max_input_elemts:
                encoder_input_data[i, t] = input_text[t]

        for t, word in enumerate(target_text):
            decoder_input_data[i, t] = target_text[t]

    return encoder_input_data, decoder_input_data, tokenizer, vocab_size, max_input_elemts, max_output_elemts


if __name__ == '__main__':
    filename = ""
    dictionary_path = ""
    window_size_body, window_size_params, window_size_name = 12, 4, 2
    main(filename, dictionary_path, window_size_body, window_size_params, window_size_name)