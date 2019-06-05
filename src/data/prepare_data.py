import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import src.data.utils.helper_functions as helper_functions
import logging
import os

#%%


def main(filename, window_size_params, window_size_body):

    #%%

    # get logger
    logger = logging.getLogger(__name__)

    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    processed_decoded_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'processed'), 'decoded'),
                                               filename + '.json')  # get decoded path

    #%%
    df = pd.read_json(processed_decoded_full_path, orient='records')

    #%% keep only data where method name is not empty
    # delete any rows where there is no method name for some reason...
    logger.info(df.shape)
    df = df[df['methodName'] != ' ']
    df = df[df['methodName'] != '']
    logger.info(df.shape)


    #%% only keep max first window_size_params params
    # only keep max first window_size_body words in methodbody

    max_input_elemts = 1 + window_size_params + window_size_body


    df['parameters'] = df['parameters'].apply(helper_functions.get_first_x_elem, args=(window_size_params,))
    df['methodBodyCleaned'] = df['methodBodyCleaned'].apply(helper_functions.get_first_x_elem, args=(window_size_body,))
    df["concatMethodBodyCleaned"] = df['Type'].map(lambda x: [x]) + df["parameters"] + df["methodBodyCleaned"]


    #%%

    def getFirstElem(x):
        try:
            return x[0]
        except IndexError:
            #if for some reason there is no method name -> map it to unknown
            logger.info(x)
            return 1
    #%%

    x_train, x_test, y_train, y_test = train_test_split(df['concatMethodBodyCleaned'], df['methodName'], test_size=0.2, random_state=200)
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
    trainX = pad_sequences(sequences, maxlen=max_input_elemts, value=0)

    # tokenize just trainY
    y_train = list(y_train)
    y_train_tokenized = tokenizer.texts_to_sequences(y_train)
    logger.info(y_train_tokenized[:5])
    y_train_tokenized = list(map(getFirstElem, y_train_tokenized))
    logger.info(y_train_tokenized[:5])
    trainY = np.array(y_train_tokenized)


    # tokenize just valX
    x_test_seq = tokenizer.texts_to_sequences(x_test)
    valX = pad_sequences(x_test_seq, maxlen=max_input_elemts, value=0)

    # tokenize just testY
    y_test = list(y_test)
    y_test_tokenized = tokenizer.texts_to_sequences(y_test)
    y_test_tokenized = list(map(getFirstElem, y_test_tokenized))
    valY = np.array(y_test_tokenized)

    print("TRAINX before X: {}".format(trainX[:10]))
    print("TRAINY before Y: {}".format(trainY[:10]))


    #hack to remove some unk from training
    #----------------------------------------

    train_df = pd.DataFrame({'trainY': trainY, 'trainX': list(trainX)})
    print(train_df.head())
    cnt_unk = len(train_df[(train_df['trainY'] == 1)])
    cnt_all = len(train_df.index)
    perc_unk = cnt_unk / cnt_all
    print(perc_unk)

    train_df = train_df.drop(train_df[train_df['trainY'] == 1].sample(frac=.5).index)
    cnt_unk = len(train_df[(train_df['trainY'] == 1)])
    cnt_all = len(train_df.index)
    perc_unk_train = cnt_unk / cnt_all
    print(perc_unk_train)

    trainX = np.array(train_df['trainX'].values.tolist())
    trainY = train_df['trainY'].values
    print("TRAINX after X: {}".format(trainX[:10]))
    print("TRAINY after Y: {}".format(trainY[:10]))

    #-------------------------------------------

    print("VALX before X: {}".format(valX[:10]))
    print("VALY before Y: {}".format(valY[:10]))

    #hack to remove some unk from validation
    #----------------------------------------

    val_df = pd.DataFrame({'valY': valY, 'valX': list(valX)})
    print(val_df.head())
    cnt_unk = len(val_df[(val_df['valY'] == 1)])
    cnt_all = len(val_df.index)
    perc_unk = cnt_unk / cnt_all
    print(perc_unk)

    val_df = val_df.drop(val_df[val_df['valY'] == 1].sample(frac=.5).index)
    cnt_unk = len(val_df[(val_df['valY'] == 1)])
    cnt_all = len(val_df.index)
    perc_unk_test = cnt_unk / cnt_all
    print(perc_unk_test)

    #val_df['valX'] = val_df['valX'].apply(lambda x: np.array(x))
    valX = np.array(val_df['valX'].values.tolist())
    valY = val_df['valY'].values
    print("VALX after X: {}".format(valX[:10]))
    print("VALY after Y: {}".format(valY[:10]))

    #-------------------------------------------


    return trainX, trainY, valX, valY, tokenizer, perc_unk_train, perc_unk_test, max_input_elemts

    # trainY = to_categorical(trainY, num_classes=vocab_size)
    # valY = to_categorical(valY, num_classes=vocab_size)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main("Android-Universal-Image-Loader", 2, 8)