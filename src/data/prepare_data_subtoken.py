import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import src.data.utils.helper_functions as helper_functions
import logging
import os



def add_start_end_token(y):
    _list = ['starttoken'] + y + ['endtoken']
    return _list


def main(filename, window_size_params, window_size_body, window_size_name):
    # basic init
    # get logger
    logger = logging.getLogger(__name__)

    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    processed_decoded_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'processed'), 'decoded'),
                                               filename + '.json')  # get decoded path

    # %% load dataset

    df = pd.read_json(processed_decoded_full_path, orient='records')

    # %%


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
    x_train, x_test, y_train, y_test = train_test_split(df['concatMethodBodyClean'], df['methodName'],
                                                        test_size=0.2,
                                                        random_state=200)
    method_body_cleaned_list_x = list(x_train)
    method_name_x = list(y_train)

    # %%dataset in training vocab format

    training_vocab_x = helper_functions.get_training_vocab(method_body_cleaned_list_x, is_for_x=True)
    training_vocab_y = helper_functions.get_training_vocab(method_name_x, is_for_x=True)

    x_train = list(map(helper_functions.get_into_tokenizer_format, method_body_cleaned_list_x))

    # %%word2idx

    # fit on text the most common words from trainX and trainY
    tokenizer = Tokenizer(oov_token=True)
    # actual training data gets mapped on text
    tokenizer.fit_on_texts(training_vocab_y)  # actual training data gets mapped on text

    word_index = tokenizer.word_index
    print('Found {} unique Y tokens.'.format(len(word_index) + 1))

    tokenizer.fit_on_texts(training_vocab_x)

    word_index = tokenizer.word_index
    print('Found {} unique X+Y tokens.'.format(len(word_index) + 1))
    # %% idx2word

    # Creating a reverse dictionary
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

    # Function takes a tokenized sentence and returns the words
    def sequence_to_text(list_of_indices):
        # Looking up words in dictionary
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        return (words)

    # %%

    # tokenize just trainX
    vocab_size = len(word_index) + 1
    x_train_tokenized = tokenizer.texts_to_sequences(x_train)
    print(x_train[:10])
    print(x_train_tokenized[:10])
    x_train_rev = list(map(sequence_to_text, x_train_tokenized))
    print(x_train_rev[:10])

    # %%

    # tokenize just trainY
    y_train = list(y_train)
    print(y_train[:20])
    y_train_tokenized = tokenizer.texts_to_sequences(y_train)
    print(y_train_tokenized[:20])
    y_train_rev = list(map(sequence_to_text, y_train_tokenized))
    print(y_train_rev[:20])
    # %%

    # tokenize just testX
    x_test_tokenized = tokenizer.texts_to_sequences(x_test)
    print(x_test[:10])
    print(x_test_tokenized[:10])
    x_test_rev = list(map(sequence_to_text, x_test_tokenized))
    print(x_test_rev[:10])

    # %%

    # tokenize just testY
    y_test = list(y_test)
    print(y_test[:20])
    y_test_tokenized = tokenizer.texts_to_sequences(y_test)
    print(y_test_tokenized[:20])
    y_test_rev = list(map(sequence_to_text, y_test_tokenized))
    print(y_test_rev[:20])

    # %%

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

    # %%

    for i, (input_text, target_text) in enumerate(zip(x_train_tokenized, y_train_tokenized)):
        for t, word in enumerate(input_text):
            if t < max_input_elemts:
                encoder_input_data[i, t] = input_text[t]

        for t, word in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t] = target_text[t]
            if t > 0:
                # decoder_target_data will be ahead by one timestep (t=0 is always start)
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_text[t]] = 1.


    # %%
    print(len(y_test_tokenized), len(x_test_tokenized))

    val_encoder_input_data = np.zeros(
        (len(x_test_tokenized), max_input_elemts),
        dtype='float32')
    val_decoder_input_data = np.zeros(
        (len(y_test_tokenized), max_output_elemts),
        dtype='float32')
    val_decoder_target_data = np.zeros(
        (len(y_test_tokenized), max_output_elemts, vocab_size),
        dtype='float32')

    # %%

    for i, (input_text, target_text) in enumerate(zip(x_test_tokenized, y_test_tokenized)):
        for t, word in enumerate(input_text):
            if t < max_input_elemts:
                val_encoder_input_data[i, t] = input_text[t]

        for t, word in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            val_decoder_input_data[i, t] = target_text[t]
            if t > 0:
                # decoder_target_data will be ahead by one timestep (t=0 is always start)
                # and will not include the start character.
                val_decoder_target_data[i, t - 1, target_text[t]] = 1.



    # print(encoder_input_data[:100])
    print(decoder_input_data[:10])
    print(decoder_target_data[:10])

    trainX = [encoder_input_data, decoder_input_data]
    trainY = decoder_target_data
    valX = [val_encoder_input_data, val_decoder_input_data]
    valY = val_decoder_target_data

    return trainX, trainY, valX, valY, tokenizer, vocab_size, max_input_elemts, max_output_elemts


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main("Android-Universal-Image-Loader-subtoken", 2, 8, 3)