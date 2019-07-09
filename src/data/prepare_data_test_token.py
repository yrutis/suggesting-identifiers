import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import src.data.utils.helper_functions as helper_functions
import logging
import os

#%%


def main(filename, window_size_params, window_size_body, tokenizer, remove_test_unk=0):


    # get logger
    logger = logging.getLogger(__name__)

    filename += '-processed'

    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    test_processed_decoded_full_path = os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(data_folder, 'processed'),
                                                            'decoded'), filename), 'test'), filename + '-token.json')

    df_test = pd.read_json(test_processed_decoded_full_path, orient='records')


    # E.g. only keep the first 10 tokens in the method body,
    # E.g. only keep the first 2 tokens in parameters
    #concate parameters, method body, type

    max_input_elemts = 1 + window_size_params + window_size_body


    df_test['parameters'] = df_test['parameters'].apply(helper_functions.get_first_x_elem, args=(window_size_params,))
    df_test['methodBody'] = df_test['methodBody'].apply(helper_functions.get_first_x_elem, args=(window_size_body,))
    df_test["concatMethodBodyCleaned"] = df_test['Type'].map(lambda x: [x]) + df_test["parameters"] + df_test["methodBody"]


    # %% idx2word

    # Creating a reverse dictionary
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

    # Function takes a tokenized sentence and returns the words
    def sequence_to_text(list_of_indices):
        # Looking up words in dictionary
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        return (words)

    #%%



    # tokenize just testX
    testX_raw = list(df_test['concatMethodBodyCleaned'])
    ##logger.info(testX_raw[:3])
    x_test_seq = tokenizer.texts_to_sequences(testX_raw)
    testX = pad_sequences(x_test_seq, maxlen=max_input_elemts, value=0)
    ##logger.info(testX[:3])
    testX_decoded = list(map(sequence_to_text, testX))
    ##logger.info(testX_decoded[:3])


    # tokenize just testY
    y_test = list(df_test['methodName'])
    #logger.info(y_test[:3])
    y_test_tokenized = tokenizer.texts_to_sequences(y_test)
    #logger.info(y_test_tokenized[:3])
    y_test_decoded = list(map(sequence_to_text, y_test_tokenized))
    #logger.info(y_test_decoded[:3])
    y_test_tokenized = list(map(helper_functions.getFirstElem, y_test_tokenized))
    testY = np.array(y_test_tokenized)


    testX, testY, perc_unk_test = \
        helper_functions.remove_some_unknowns_test(testX, testY,
                                              remove_test=remove_test_unk)


    return testX, testY, perc_unk_test

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main("Android-Universal-Image-Loader", 2, 8)