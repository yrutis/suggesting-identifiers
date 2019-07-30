import math
from pickle import dump, load
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import src.data.utils.helper_functions as helper_functions
import logging
import os
from os import listdir
import collections



def add_start_end_token(y):
    _list = ['starttoken'] + y + ['endtoken']
    return _list


def main(config, report_folder=''):
    # basic init
    # get logger
    # get logger
    logger = logging.getLogger(__name__)
    compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

    filename, window_size_body, window_size_params, window_size_name, partition = \
        config.data_loader.name, config.data_loader.window_size_body, \
        config.data_loader.window_size_params, config.data_loader.window_size_name, \
        config.data_loader.partition

    filename += '-processed'


    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'data')
    training_processed_decoded_full_path = os.path.join(
        os.path.join(os.path.join(os.path.join(os.path.join(data_folder, 'processed'),
                                               'decoded'), filename), 'training'), filename + '-subtoken.json')
    validation_processed_decoded_full_path = os.path.join(
        os.path.join(os.path.join(os.path.join(os.path.join(data_folder, 'processed'),
                                               'decoded'), filename), 'validation'), filename + '-subtoken.json')

    data_storage = os.path.join(
        os.path.join(os.path.join(os.path.join(os.path.join(data_folder, 'processed'),
                                               'decoded'), filename), 'training'),
        'partition' + str(partition) + '-training-params-' + str(window_size_params) + '-body-' + str(window_size_body)
        + '-name-' + str(window_size_name))



    #the files tend to get large... hence I need to safe them in scratch/yrutis
    #data_storage = os.path.join(os.path.join(os.path.join('scratch', 'yrutis'), filename),
     #                           'partition' + str(partition) + '-training-params-' + str(
      #                              window_size_params) + '-body-' + str(window_size_body)
       #                         + '-name-' + str(window_size_name))



    df_train = pd.read_json(training_processed_decoded_full_path, orient='records')
    df_val = pd.read_json(validation_processed_decoded_full_path, orient='records')


    if not os.path.exists(data_storage):
        logger.info("folder created: {}".format(data_storage))
        os.mkdir(data_storage)



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


        #%% shuffle dataframe
        df_train = df_train.sample(frac=1)
        df_val = df_val.sample(frac=1)

        #%%
        method_body_cleaned_list_x = list(df_train['concatMethodBodyClean'])
        method_name_x = list(df_train['methodName'])

        # %%dataset in training vocab format

        training_vocab_x = helper_functions.get_training_vocab(method_body_cleaned_list_x, is_for_x=True)
        training_vocab_y = helper_functions.get_training_vocab(method_name_x, is_for_x=True)

        x_train = list(map(helper_functions.get_into_tokenizer_format, method_body_cleaned_list_x))

        # %%word2idx

        # fit on text the most common words from trainX and trainY
        tokenizer = Tokenizer(oov_token="UNK")
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

        # tokenize just trainX
        vocab_size = len(word_index) + 1
        x_train_tokenized = tokenizer.texts_to_sequences(x_train)
        x_train_tokenized = pad_sequences(x_train_tokenized, maxlen=max_input_elemts,
                                          padding='post', truncating='post')

        # print(x_train[:10])
        # print(x_train_tokenized[:10])
        x_train_rev = list(map(sequence_to_text, x_train_tokenized))
        # print(x_train_rev[:10])
        x_train_rev_pd = pd.Series(x_train_rev)

        # %%

        # tokenize just trainY
        y_train = list(df_train['methodName'])
        # print(y_train[:50000])
        y_train_tokenized = tokenizer.texts_to_sequences(y_train)
        y_train_tokenized = pad_sequences(y_train_tokenized, maxlen=max_output_elemts, padding='post',
                                          truncating='post')
        # print(y_train_tokenized[:50000])
        y_train_rev = list(map(sequence_to_text, y_train_tokenized))
        # print(y_train_rev[:50000])
        # %%

        # tokenize just valX
        x_val_tokenized = tokenizer.texts_to_sequences(df_val['concatMethodBodyClean'])
        x_val_tokenized = pad_sequences(x_val_tokenized, maxlen=max_input_elemts, padding='post', truncating='post')

        # print(df_val['concatMethodBodyClean'][:10])
        # print(x_val_tokenized[:10])
        x_val_rev = list(map(sequence_to_text, x_val_tokenized))
        # print(x_val_rev[:10])

        # %%

        # tokenize just valY
        y_val = list(df_val['methodName'])
        # print(y_val[:50000])
        y_val_tokenized = tokenizer.texts_to_sequences(y_val)
        y_val_tokenized = pad_sequences(y_val_tokenized, maxlen=max_output_elemts,
                                        padding='post', truncating='post')

        # print(y_val_tokenized[:50000])
        y_val_rev = list(map(sequence_to_text, y_val_tokenized))
        # print(y_val_rev[:50000])

        # %%

        logger.info("len Y Train Tokenized {}, len X Train Tokenized {}"
                        .format(len(y_train_tokenized), len(x_train_tokenized)))

        all_train = [x for x in range(len(x_train_tokenized))]
        all_val = [x for x in range(len(x_val_tokenized))]
        # %%

        def split_to_chunks(np_array):
            '''
            :param np_array: numpy array to be splitted into chunks
            :return:
            '''
            #1) retrieve indices
            np_array_split_indices = [partition * (i + 1) for i in range(math.floor(np_array.shape[0] / partition))]

            #only do splitting operation if size of the array is bigger than partition
            if len(np_array_split_indices) > 0:
                if np_array_split_indices[-1] == np_array.shape[0]:
                    del np_array_split_indices[-1]

                # 2) split np array according to indices
                np_array_splitted = np.array_split(np_array, np_array_split_indices, axis=0)

            else:
                np_array_splitted = [np_array] #save as list



            return np_array_splitted

        def save_to_chunks(x, y, type):
            for j, (x_chunk, y_chunk) in enumerate(
                    zip(x, y)):

                encoder_input_data = np.zeros(
                    (x_chunk.shape[0], max_input_elemts),
                    dtype='int')
                decoder_input_data = np.zeros(
                    (y_chunk.shape[0], max_output_elemts),
                    dtype='int')
                decoder_target_data = np.zeros(
                    (y_chunk.shape[0], max_output_elemts, vocab_size),
                    dtype='int')

                for i, (input_text, target_text) in enumerate(zip(x_chunk, y_chunk)):
                    assert (len(input_text) == max_input_elemts)  # make sure always whole matrix is filled
                    assert (len(target_text) == max_output_elemts)

                    for t, word in enumerate(input_text):
                        encoder_input_data[i, t] = input_text[t]

                    for t, word in enumerate(target_text):
                        # decoder_target_data is ahead of decoder_input_data by one timestep
                        decoder_input_data[i, t] = target_text[t]
                        if t > 0:
                            # decoder_target_data will be ahead by one timestep (t=0 is always start)
                            # and will not include the start character.
                            decoder_target_data[i, t - 1, target_text[t]] = 1

                np.save(os.path.join(data_storage,
                                     type+'X1-' + str(j)), encoder_input_data)
                np.save(os.path.join(data_storage,
                                     type+'X2-' + str(j)), decoder_input_data)
                np.save(os.path.join(data_storage,
                                     type+'Y-' + str(j)), decoder_target_data)

                #for the last one save the remainder
                if (j+1) == len(x):
                    remaining = x_chunk.shape[0]
                    remaining_path = os.path.join(data_storage, 'remaining_'+type+'.pkl')
                    dump(remaining, open(remaining_path, 'wb'))
                    print("total {}, remaining: {}".format(len(x), remaining))



        x_train_tokenized_splitted = split_to_chunks(x_train_tokenized)
        y_train_tokenized_splitted = split_to_chunks(y_train_tokenized)
        x_val_tokenized_splitted = split_to_chunks(x_val_tokenized)
        y_val_tokenized_splitted = split_to_chunks(y_val_tokenized)

        save_to_chunks(x_train_tokenized_splitted, y_train_tokenized_splitted, 'train')
        save_to_chunks(x_val_tokenized_splitted, y_val_tokenized_splitted, 'val')



        vocab_size = len(tokenizer.word_index) + 1

        # safe tokenizer
        tokenizer_path = os.path.join(data_storage, 'tokenizer.pkl') #save it in the data storage folder
        dump(tokenizer, open(tokenizer_path, 'wb'))

        if report_folder != '':
            tokenizer_path = os.path.join(report_folder, 'tokenizer.pkl')  # also safe it in the report folder
            dump(tokenizer, open(tokenizer_path, 'wb'))

        logger.info("done saving training, val chunks, tokenizer...")


    else:
        logger.info("folder exists: {}".format(data_storage))

        all_train_files = [f for f in listdir(data_storage) if f.startswith('trainX1')]
        all_val_files = [f for f in listdir(data_storage) if f.startswith('valX1')]

        with open(os.path.join(data_storage, 'remaining_train.pkl'), "rb") as input_file:
            remaining_training = load(input_file)

        with open(os.path.join(data_storage, 'remaining_val.pkl'), "rb") as input_file:
            remaining_val = load(input_file)

        with open(os.path.join(data_storage, 'tokenizer.pkl'), "rb") as input_file:
            tokenizer = load(input_file)

        vocab_size = len(tokenizer.word_index) + 1 #I only need the vocab size

        all_train = [x for x in range((len(all_train_files) - 1) * partition + remaining_training)]  # load all regular partions + the length of last irregular partion
        all_val = [x for x in range((len(all_val_files) - 1) * partition + remaining_val)]

        max_input_elemts = 1 + window_size_params + window_size_body + 2  # return type + ... + ... + startendtoken
        max_output_elemts = 2 + window_size_name  # startendtoken + ...

        tokenizer_path = os.path.join(report_folder, 'tokenizer.pkl') #also safe it in the report folder
        dump(tokenizer, open(tokenizer_path, 'wb'))


    return all_train, all_val, vocab_size, max_input_elemts, max_output_elemts, data_storage


