import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import src.data.utils.helper_functions as helper_functions
import logging
import os


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
    df["methodBody"] = df['methodBody'].apply(helper_functions.turn_string_to_function)
    df["methodBody"] = df['methodBody'].apply(helper_functions.delete_certain_strings)
    df["methodBody"] = df['methodBody'].apply(helper_functions.turn_all_to_lower)

    # df['methodName']= df['methodName'].str.lower() should a function be all lower?

    # avg_mean = df['methodBody'].apply(compute_col_length).mean()
    df['methodBodyCleaned'] = df['methodBody'].apply(helper_functions.clean_from_function_structure)

    df["concatMethodBodyCleaned"] = df['Type'].map(lambda x: [x]) + df["parameters"] + df["methodBodyCleaned"]

    # avg_mean_cleaned = df['methodBodyCleaned'].apply(compute_col_length).mean()

    x_train, x_test, y_train, y_test = train_test_split(df['concatMethodBodyCleaned'], df['methodName'], test_size=0.2)
    method_body_cleaned_list_x = list(x_train)
    method_name_x = list(y_train)

    training_vocab_x = helper_functions.get_training_vocab(method_body_cleaned_list_x, is_for_x=True)
    training_vocab_y = helper_functions.get_training_vocab(method_name_x, is_for_x=False)

    x_train = list(map(helper_functions.get_into_tokenizer_format, method_body_cleaned_list_x))
    #print(x_train[:10])

    # fit on text the most common words from trainX and trainY
    tokenizer = Tokenizer(oov_token=True)
      # actual training data gets mapped on text
    tokenizer.fit_on_texts(training_vocab_y)  # actual training data gets mapped on text

    word_index = tokenizer.word_index
    logger.info('Found {} unique Y tokens.'.format(len(word_index) + 1))

    tokenizer.fit_on_texts(training_vocab_x)

    word_index = tokenizer.word_index
    logger.info('Found {} unique X+Y tokens.'.format(len(word_index) + 1))

    # tokenize just trainX
    vocab_size = len(word_index) + 1
    sequences = tokenizer.texts_to_sequences(x_train)
    #print(sequences[:10])
    trainX = pad_sequences(sequences, maxlen=window_size, value=0)
    #print(trainX[:10])

    # tokenize just trainY
    y_train = list(y_train)
    y_train_tokenized = tokenizer.texts_to_sequences(y_train)
    #print(y_train_tokenized[0:10])
    #print(y_train[0:10])
    y_train_tokenized = list(map(lambda x: x[0], y_train_tokenized))

    counter = 0
    for x in y_train_tokenized:
        if x == 1:
            counter += 1
    logger.info("has this amount of UNK functions in Y Train {}, percentage of total {}".format(counter, counter / len(
        y_train_tokenized)))

    always_unknown_train = counter / len(y_train_tokenized)

    trainY = np.array(y_train_tokenized)

    # tokenize just valX
    x_test_seq = tokenizer.texts_to_sequences(x_test)
    valX = pad_sequences(x_test_seq, maxlen=window_size, value=0)

    # tokenize just testY
    y_test = list(y_test)
    y_test_tokenized = tokenizer.texts_to_sequences(y_test)
    #print(y_test_tokenized[0:10])
    #print(y_test[0:10])
    y_test_tokenized = list(map(lambda x: x[0], y_test_tokenized))
    valY = np.array(y_test_tokenized)

    counter = 0
    for x in y_test_tokenized:
        if x == 1:
            counter += 1
    logger.info("has this amount of UNK functions in Y Val {} percentage of total {}".format(counter,
                                                                                       counter / len(y_test_tokenized)))

    always_unknown_test = counter / len(y_test_tokenized)
    return trainX, trainY, valX, valY, tokenizer, always_unknown_train, always_unknown_test

    # trainY = to_categorical(trainY, num_classes=vocab_size)
    # valY = to_categorical(valY, num_classes=vocab_size)

if __name__ == '__main__':
    main("all_methods_train", 8)