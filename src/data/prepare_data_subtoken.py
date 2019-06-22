import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
import src.data.utils.helper_functions as helper_functions
import logging
import os



def add_start_end_token(y):
    _list = ['starttoken'] + y + ['endtoken']
    return _list


def main(filename, window_size_params, window_size_body, window_size_name):
    # basic init
    # get logger
    # get logger
    logger = logging.getLogger(__name__)

    filename += '-processed'

    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    training_processed_decoded_full_path = os.path.join(
        os.path.join(os.path.join(os.path.join(os.path.join(data_folder, 'processed'),
                                               'decoded'), filename), 'training'), filename + '-subtoken.json')
    validation_processed_decoded_full_path = os.path.join(
        os.path.join(os.path.join(os.path.join(os.path.join(data_folder, 'processed'),
                                               'decoded'), filename), 'validation'), filename + '-subtoken.json')

    df_train = pd.read_json(training_processed_decoded_full_path, orient='records')
    df_val = pd.read_json(validation_processed_decoded_full_path, orient='records')


    # %%


    max_input_elemts = 1 + window_size_params + window_size_body + 2  # return type + ... + ... + startendtoken
    max_output_elemts = 2 + window_size_name  # startendtoken + ...

    df_train['parameters'] = df_train['parameters'].apply(helper_functions.get_first_x_elem, args=(window_size_params,))
    df_train['methodBody'] = df_train['methodBody'].apply(helper_functions.get_first_x_elem, args=(window_size_body,))
    df_train["concatMethodBodyClean"] = df_train['Type'].map(lambda x: [x]) + df_train["parameters"] + df_train["methodBody"]

    df_train['methodName'] = df_train['methodName'].apply(helper_functions.get_first_x_elem, args=(window_size_name,))

    # %% add start end token
    df_train['concatMethodBodyClean'] = df_train['concatMethodBodyClean'].apply(add_start_end_token)
    df_train['methodName'] = df_train['methodName'].apply(add_start_end_token)


    df_val['parameters'] = df_val['parameters'].apply(helper_functions.get_first_x_elem, args=(window_size_params,))
    df_val['methodBody'] = df_val['methodBody'].apply(helper_functions.get_first_x_elem, args=(window_size_body,))
    df_val["concatMethodBodyClean"] = df_val['Type'].map(lambda x: [x]) + df_val["parameters"] + df_val["methodBody"]

    df_val['methodName'] = df_val['methodName'].apply(helper_functions.get_first_x_elem, args=(window_size_name,))

    # %% add start end token
    df_val['concatMethodBodyClean'] = df_val['concatMethodBodyClean'].apply(add_start_end_token)
    df_val['methodName'] = df_val['methodName'].apply(add_start_end_token)


    method_body_cleaned_list_x = list(df_train['concatMethodBodyClean'])
    method_name_x = list(df_train['methodName'])

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
    logger.info('Found {} unique Y tokens.'.format(len(word_index) + 1))

    tokenizer.fit_on_texts(training_vocab_x)

    word_index = tokenizer.word_index
    logger.info('Found {} unique X+Y tokens.'.format(len(word_index) + 1))
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
    y_train = list(df_train['methodName'])
    print(y_train[:20])
    y_train_tokenized = tokenizer.texts_to_sequences(y_train)
    print(y_train_tokenized[:20])
    y_train_rev = list(map(sequence_to_text, y_train_tokenized))
    print(y_train_rev[:20])
    # %%

    # tokenize just valX
    x_val_tokenized = tokenizer.texts_to_sequences(df_val['concatMethodBodyClean'])
    print(df_val['concatMethodBodyClean'][:10])
    print(x_val_tokenized[:10])
    x_val_rev = list(map(sequence_to_text, x_val_tokenized))
    print(x_val_rev[:10])

    # %%

    # tokenize just valY
    y_val = list(df_val['methodName'])
    print(y_val[:20])
    y_val_tokenized = tokenizer.texts_to_sequences(y_val)
    print(y_val_tokenized[:20])
    y_val_rev = list(map(sequence_to_text, y_val_tokenized))
    print(y_val_rev[:20])

    # %%

    logger.info("len Y Train Tokenized {}, len X Train Tokenized {}"
                    .format(len(y_train_tokenized), len(x_train_tokenized)))


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
    logger.info("len Y val tokenized {}, len X val toknized {}"
                .format(len(y_val_tokenized), len(x_val_tokenized)))

    val_encoder_input_data = np.zeros(
        (len(x_val_tokenized), max_input_elemts),
        dtype='float32')
    val_decoder_input_data = np.zeros(
        (len(y_val_tokenized), max_output_elemts),
        dtype='float32')
    val_decoder_target_data = np.zeros(
        (len(y_val_tokenized), max_output_elemts, vocab_size),
        dtype='float32')

    # %%

    for i, (input_text, target_text) in enumerate(zip(x_val_tokenized, y_val_tokenized)):
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
    #print(decoder_input_data[:10])
    #print(decoder_target_data[:10])

    trainX = [encoder_input_data, decoder_input_data]
    trainY = decoder_target_data
    valX = [val_encoder_input_data, val_decoder_input_data]
    valY = val_decoder_target_data

    return trainX, trainY, valX, valY, tokenizer, vocab_size, max_input_elemts, max_output_elemts


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main("Android-Universal-Image-Loader-subtoken", 2, 8, 3)