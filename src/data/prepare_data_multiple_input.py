#%%import some stuff

import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import src.data.utils.helper_functions as helper_functions
import logging
import os


def main(filename, window_size):

    #%% init important variables



    logger = logging.getLogger(__name__)

    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    processed_decoded_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'processed'), 'decoded'),
                                           filename + '.json')  # get decoded path

    #%%
    #read df
    df = pd.read_json(processed_decoded_full_path, orient='records')

    #%%
    #split train and validation

    train = df.sample(frac=0.8, random_state=200)
    val = df.drop(train.index)

    logger.info(df.shape)
    logger.info(train.shape)
    logger.info(val.shape)

    #%%

    #get all the input vocab only for vocab size
    all_input_vocab_x = train['concatMethodBodyCleaned']

    #retrieve different inputs seperately
    train_method_body= train['methodBodyCleaned']
    train_parameters = train['parameters']
    train_type = train['Type']

    #retrieve Y
    train_method_name = train['methodName']

    #%%

    #generate training vocab (all tokens that appear >=3 are considered
    training_vocab_x = helper_functions.get_training_vocab(all_input_vocab_x, is_for_x=True)
    training_vocab_y = helper_functions.get_training_vocab(train_method_name, is_for_x=False)

    # fit on text the most common words from X and Y
    tokenizer = Tokenizer(oov_token=True)
    # actual training data gets mapped on text
    tokenizer.fit_on_texts(training_vocab_x)  # actual training data gets mapped on text

    word_index = tokenizer.word_index

    logger.info('Found {} unique X tokens.'.format(len(word_index) + 1))

    tokenizer.fit_on_texts(training_vocab_y)
    word_index = tokenizer.word_index
    logger.info('Found {} unique X+Y tokens.'.format(len(word_index) + 1))

    vocab_size = len(word_index) + 1



    #%%

    #get into tokenizer format
    train_method_body = list(map(helper_functions.get_into_tokenizer_format, train_method_body))
    train_parameters = list(map(helper_functions.get_into_tokenizer_format, train_parameters))

    train_type = list(train_type)

    # tokenize just train_method_body
    sequences = tokenizer.texts_to_sequences(train_method_body)
    train_method_body_padded = pad_sequences(sequences, maxlen=window_size, value=0)

    # tokenize just train_parameters
    sequences = tokenizer.texts_to_sequences(train_parameters)
    train_parameters_padded = pad_sequences(sequences, maxlen=window_size, value=0)

    # tokenize just train_type
    sequences = tokenizer.texts_to_sequences(train_type)
    train_type_padded = pad_sequences(sequences, maxlen=window_size, value=0)


    #%%
    # tokenize just trainY
    y_train = list(train_method_name)
    y_train_tokenized = tokenizer.texts_to_sequences(y_train)
    y_train_tokenized = list(map(lambda x: x[0], y_train_tokenized))
    trainY = np.array(y_train_tokenized)

    #%% remove some unknowns from training

    logger.info("TRAINX before X: {}".format(train_method_body_padded[:10]))
    logger.info("TRAINY before Y: {}".format(trainY[:10]))

    # hack to remove some unk from training
    # ----------------------------------------

    train_df = pd.DataFrame({'trainY': trainY,
                             'train_method_body': list(train_method_body_padded),
                             'train_parameters': list(train_parameters_padded),
                             'train_type': list(train_type_padded)})


    logger.info(train_df.head())

    #%%
    cnt_unk = len(train_df[(train_df['trainY'] == 1)])
    cnt_all = len(train_df.index)
    perc_unk = cnt_unk / cnt_all
    print(perc_unk)

    train_df = train_df.drop(train_df[train_df['trainY'] == 1].sample(frac=.5).index)
    cnt_unk = len(train_df[(train_df['trainY'] == 1)])
    cnt_all = len(train_df.index)
    perc_unk_train = cnt_unk / cnt_all
    print(perc_unk_train)

    train_method_body_padded = np.array(train_df['train_method_body'].values.tolist()) #hack: convert to list then to numpy
    train_parameters_padded = np.array(train_df['train_parameters'].values.tolist())
    train_type_padded = np.array(train_df['train_type'].values.tolist())
    trainY = train_df['trainY'].values
    print("TRAINX after X: {}".format(train_method_body_padded[:10]))
    print("TRAINY after Y: {}".format(trainY[:10]))


    #%%
    #retrieve different VAL inputs seperately
    val_method_body= val['methodBodyCleaned']
    val_parameters = val['parameters']
    val_type = val['Type']
    val_method_name = val['methodName']

    #%%
    #get into tokenizer format
    val_method_body = list(map(helper_functions.get_into_tokenizer_format, val_method_body))
    val_parameters = list(map(helper_functions.get_into_tokenizer_format, val_parameters))

    val_type = list(val_type)

    # tokenize just val_method_body
    sequences = tokenizer.texts_to_sequences(val_method_body)
    val_method_body_padded = pad_sequences(sequences, maxlen=window_size, value=0)

    # tokenize just val_parameters
    sequences = tokenizer.texts_to_sequences(val_parameters)
    val_parameters_padded = pad_sequences(sequences, maxlen=window_size, value=0)

    # tokenize just val_type
    sequences = tokenizer.texts_to_sequences(val_type)
    val_type_padded = pad_sequences(sequences, maxlen=window_size, value=0)


    #%%
    # tokenize just valY
    y_val = list(val_method_name)
    y_val_tokenized = tokenizer.texts_to_sequences(y_val)
    y_val_tokenized = list(map(lambda x: x[0], y_val_tokenized))
    valY = np.array(y_val_tokenized)



    #%% remove some unknowns from training

    logger.info("valX before X: {}".format(val_method_body_padded[:10]))
    logger.info("valY before Y: {}".format(valY[:10]))

    # hack to remove some unk from training
    # ----------------------------------------

    val_df = pd.DataFrame({'valY': valY,
                             'val_method_body': list(val_method_body_padded),
                             'val_parameters': list(val_parameters_padded),
                             'val_type': list(val_type_padded)})


    logger.info(val_df.head())

    #%%
    cnt_unk = len(val_df[(val_df['valY'] == 1)])
    cnt_all = len(val_df.index)
    perc_unk = cnt_unk / cnt_all
    print(perc_unk)

    val_df = val_df.drop(val_df[val_df['valY'] == 1].sample(frac=.5).index)
    cnt_unk = len(val_df[(val_df['valY'] == 1)])
    cnt_all = len(val_df.index)
    perc_unk_val = cnt_unk / cnt_all
    print(perc_unk_val)

    val_method_body_padded = np.array(val_df['val_method_body'].values.tolist()) #hack: convert to list then to numpy
    val_parameters_padded = np.array(val_df['val_parameters'].values.tolist())
    val_type_padded = np.array(val_df['val_type'].values.tolist())
    valY = val_df['valY'].values
    print("valX after X: {}".format(val_method_body_padded[:10]))
    print("valY after Y: {}".format(valY[:10]))

    return train_method_body_padded, train_parameters_padded, train_type_padded, trainY, \
           val_method_body_padded, val_parameters_padded, val_type_padded, valY, \
           tokenizer, perc_unk_train, perc_unk_val


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    filename = 'all_methods_train_without_platform'
    window_size = 8
    main(filename, window_size)

