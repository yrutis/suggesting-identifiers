from pickle import dump, load

import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import src.data.utils.helper_functions as helper_functions
import logging
import os

#%%


def main(filename, window_size_params, window_size_body, partition, report_folder='', remove_train_unk=0, remove_val_unk=0):


    # get logger
    logger = logging.getLogger(__name__)

    filename += '-processed'

    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'data')
    training_processed_decoded = os.path.join(
        os.path.join(os.path.join(os.path.join(data_folder, 'processed'),
                                               'decoded'), filename), 'training')
    validation_processed_decoded = os.path.join(
        os.path.join(os.path.join(os.path.join(data_folder, 'processed'),
                                               'decoded'), filename), 'validation')
    training_processed_decoded_full_path = os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(data_folder, 'processed'),
                                                        'decoded'), filename), 'training'), filename + '-token.json')
    validation_processed_decoded_full_path = os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(data_folder, 'processed'),
                                                            'decoded'), filename), 'validation'), filename + '-token.json')

    data_storage = os.path.join(
        os.path.join(os.path.join(os.path.join(os.path.join(data_folder, 'processed'),
                                               'decoded'), filename), 'training'),
        'partition-' + str(partition) + 'training-params-' + str(window_size_params) + '-body-' + str(window_size_body))


    if not os.path.exists(data_storage):

        df_train = pd.read_json(training_processed_decoded_full_path, orient='records')
        df_val = pd.read_json(validation_processed_decoded_full_path, orient='records')

        logger.info("folder created: {}".format(data_storage))
        os.mkdir(data_storage)

        # E.g. only keep the first 10 tokens in the method body,
        # E.g. only keep the first 2 tokens in parameters
        #concate parameters, method body, type

        max_input_elemts = 1 + window_size_params + window_size_body

        df_train['parameters'] = df_train['parameters'].apply(helper_functions.get_first_x_elem, args=(window_size_params,))
        df_train['methodBody'] = df_train['methodBody'].apply(helper_functions.get_first_x_elem, args=(window_size_body,))
        df_train["concatMethodBodyCleaned"] = df_train['Type'].map(lambda x: [x]) + df_train["parameters"] + df_train["methodBody"]

        df_val['parameters'] = df_val['parameters'].apply(helper_functions.get_first_x_elem, args=(window_size_params,))
        df_val['methodBody'] = df_val['methodBody'].apply(helper_functions.get_first_x_elem, args=(window_size_body,))
        df_val["concatMethodBodyCleaned"] = df_val['Type'].map(lambda x: [x]) + df_val["parameters"] + df_val["methodBody"]


        trainX = list(df_train['concatMethodBodyCleaned'])
        trainY = list(df_train['methodName'])


        #create training vocab
        training_vocab_x = helper_functions.get_training_vocab(trainX, is_for_x=True)
        training_vocab_y = helper_functions.get_training_vocab(trainY, is_for_x=False)

        # fit on text words that appear at least 3x from trainX and trainY
        tokenizer = Tokenizer(oov_token="UNK", filters='')

        # actual training data gets mapped on text
        tokenizer.fit_on_texts(training_vocab_y)
        logger.info('Found {} unique Y tokens.'.format(len(tokenizer.word_index) + 1))
        tokenizer.fit_on_texts(training_vocab_x)
        logger.info('Found {} unique X+Y tokens.'.format(len(tokenizer.word_index) + 1))

        # %% idx2word

        # Creating a reverse dictionary
        reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

        # Function takes a tokenized sentence and returns the words
        def sequence_to_text(list_of_indices):
            # Looking up words in dictionary
            words = [reverse_word_map.get(letter) for letter in list_of_indices]
            return (words)

        #%%

        # tokenize trainX
        x_train = list(map(helper_functions.get_into_tokenizer_format, trainX))
        sequences = tokenizer.texts_to_sequences(x_train)
        trainX = pad_sequences(sequences, maxlen=max_input_elemts, value=0)

        # tokenize trainY
        y_train = list(df_train['methodName'])
        y_train_tokenized = tokenizer.texts_to_sequences(y_train)
        #logger.info(y_train_tokenized[:5])
        y_train_tokenized = list(map(helper_functions.getFirstElem, y_train_tokenized))
        #logger.info(y_train_tokenized[:5])
        trainY = np.array(y_train_tokenized)


        # tokenize just valX
        valX_raw = list(df_val['concatMethodBodyCleaned'])
        x_test_seq = tokenizer.texts_to_sequences(valX_raw)
        valX = pad_sequences(x_test_seq, maxlen=max_input_elemts, value=0)
        #logger.info(valX[:3])
        #valX_decoded = list(map(sequence_to_text, valX))
        #logger.info(valX_decoded[:3])


        # tokenize just testY
        y_test = list(df_val['methodName'])
        #logger.info(y_test[:3])
        y_test_tokenized = tokenizer.texts_to_sequences(y_test)
        #logger.info(y_test_tokenized[:3])
        #y_test_decoded = list(map(sequence_to_text, y_test_tokenized))
        #logger.info(y_test_decoded[:3])
        y_test_tokenized = list(map(helper_functions.getFirstElem, y_test_tokenized))
        valY = np.array(y_test_tokenized)


        trainX, trainY, valX, valY, perc_unk_train, perc_unk_val = \
            helper_functions.remove_some_unknowns(trainX, trainY, valX, valY,
                                                  remove_train=remove_train_unk,
                                                  remove_val=remove_val_unk)



        def save_to_chunks(X: np.ndarray, Y:np.ndarray, folder, type: str):

            i = 0
            list_of_elements = []
            assert(X.shape[0] == Y.shape[0])
            while True:
                if i + partition < X.shape[0]:
                    current_name_x = type + 'X-' + str(i//partition)
                    current_name_y = type + 'Y-' + str(i//partition)
                    np.save(os.path.join(folder, current_name_x), X[i:i + partition])
                    np.save(os.path.join(folder, current_name_y), Y[i:i + partition])
                    list_of_elements.append(i)
                elif i < X.shape[0]:
                    current_name_x = type + 'X-' + str(i//partition)
                    current_name_y = type + 'Y-' + str(i//partition)
                    np.save(os.path.join(folder, current_name_x), X[i:X.shape[0]])
                    np.save(os.path.join(folder, 'X-last-'+type), X[i:X.shape[0]]) #save last file twice to count occurences
                    np.save(os.path.join(folder, current_name_y), Y[i:X.shape[0]])
                    list_of_elements.append(i)
                else:
                    break
                i += partition

            return list_of_elements


        all_train = save_to_chunks(trainX, trainY, data_storage, 'train')
        all_val = save_to_chunks(valX, valY, data_storage, 'val')


        vocab_size = len(tokenizer.word_index) + 1

        trainxshape = trainX.shape[0] / partition + 1
        len_alltrain = len(all_train)

        if trainX.shape[0] % partition != 0: #if not dividable by partition it is plus 1 else not
            assert(((trainX.shape[0] // partition) +1) == len(all_train)) #check if it was correctly broken into partions of partition000
        else:
            assert (trainX.shape[0] / partition == len(all_train))  # check if it was correctly broken into partions of partition000

        if valX.shape[0] % partition != 0: #if not dividable by partition it is plus 1 else not
            assert(((valX.shape[0] // partition) +1) == len(all_val))
        else:
            assert (valX.shape[0] / partition == len(all_val))


        all_train = [x for x in range(trainX.shape[0])]
        all_val = [x for x in range(valX.shape[0])]



        # safe tokenizer
        tokenizer_path = os.path.join(data_storage, 'tokenizer.pkl')
        dump(tokenizer, open(tokenizer_path, 'wb'))

        if report_folder != '':
            tokenizer_path = os.path.join(report_folder, 'tokenizer.pkl')  # also safe it in the report folder
            dump(tokenizer, open(tokenizer_path, 'wb'))


        #dump perc unk
        perc_unk_path = os.path.join(data_storage, 'perc_unk.pkl')
        perc_unk_list = [perc_unk_train, perc_unk_val]
        dump(perc_unk_list, open(perc_unk_path, 'wb'))

        logger.info("done saving training, val chunks, tokenizer, perc unk train val...")

        return all_train, all_val, vocab_size, max_input_elemts, data_storage, perc_unk_train, perc_unk_val


    else:
        logger.info("folder exists: {}".format(data_storage))

        all_train_files = [f for f in os.listdir(data_storage) if f.startswith('trainX')]
        all_val_files = [f for f in os.listdir(data_storage) if f.startswith('valX')]

        #count how many are last
        last_training = np.load(os.path.join(data_storage, 'X-last-train.npy'))
        last_val = np.load(os.path.join(data_storage, 'X-last-val.npy'))

        with open(os.path.join(data_storage, 'tokenizer.pkl'), "rb") as input_file:
            tokenizer = load(input_file)

        with open(os.path.join(data_storage, 'perc_unk.pkl'), "rb") as input_file:
            perc_unk = load(input_file)

        assert(len(perc_unk) == 2)
        perc_unk_train, perc_unk_val = perc_unk[0], perc_unk[1]

        vocab_size = len(tokenizer.word_index) + 1  # I only need the vocab size


        all_train = [x for x in range((len(all_train_files)-1)*partition+last_training.shape[0])] #load all regular partions + the length of last irregular partion
        all_val = [x for x in range((len(all_val_files)-1)*partition+last_val.shape[0])]

        max_input_elemts = 1 + window_size_params + window_size_body  # return type + ... + ... + startendtoken

        if report_folder != '':
            tokenizer_path = os.path.join(report_folder, 'tokenizer.pkl')  # also safe it in the report folder
            dump(tokenizer, open(tokenizer_path, 'wb'))

        return all_train, all_val, vocab_size, max_input_elemts, data_storage, perc_unk_train, perc_unk_val



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main("Android-Universal-Image-Loader", 2, 1, '')