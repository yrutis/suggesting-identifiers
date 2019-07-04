import pandas as pd

import src.data.utils.helper_functions as helper_functions
import logging
import os
from sklearn.model_selection import train_test_split


#%% get logger to work
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
filename = "java-med-processed"
token_model = False

# get logger
logger = logging.getLogger(__name__)
logger.info(filename)


data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')


processed_decoded_full_path = os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(data_folder, 'processed'),
                                                        'decoded'), filename), 'training'), filename + '-subtoken.json')

#%%
df = pd.read_json(processed_decoded_full_path, orient='records')

#%%get shape of train

len_train = df.shape



#%% calc method body length

len_method_body = df['methodBody'].apply(helper_functions.compute_col_length)
len_method_body = len_method_body.describe()

logger.info(len_method_body)



#%% calc parameter length

len_parameters = df['parameters'].apply(helper_functions.compute_col_length)
len_parameters = len_parameters.describe()

logger.info(len_parameters)

#%%

if token_model:

    window_size_params = 2
    window_size_body = 20

    df['parameters'] = df['parameters'].apply(helper_functions.get_first_x_elem, args=(window_size_params,))
    df['methodBody'] = df['methodBody'].apply(helper_functions.get_first_x_elem, args=(window_size_body,))
    df["concatMethodBodyCleaned"] = df['Type'].map(lambda x: [x]) + df["parameters"] + df["methodBody"]



    #%% only for token model get perc of unk!


    import numpy as np

    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences

    max_input_elemts = 1 + window_size_params + window_size_body

    x_train, y_train = df['concatMethodBodyCleaned'], df['methodName']
    method_body_cleaned_list_x = list(x_train)
    method_name_x = list(y_train)

    training_vocab_x = helper_functions.get_training_vocab(method_body_cleaned_list_x, is_for_x=True)
    training_vocab_y = helper_functions.get_training_vocab(method_name_x, is_for_x=False)

    x_train = list(map(helper_functions.get_into_tokenizer_format, method_body_cleaned_list_x))
    #print(x_train[:10])

    # fit on text the most common words from trainX and trainY
    tokenizer = Tokenizer(oov_token='UNK')
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
    y_train_tokenized = list(map(helper_functions.getFirstElem, y_train_tokenized))
    trainY = np.array(y_train_tokenized)


    #----------------------------------------

    train_df = pd.DataFrame({'trainY': trainY, 'trainX': list(trainX)})
    cnt_unk = len(train_df[(train_df['trainY'] == 1)])
    cnt_all = len(train_df.index)
    perc_unk = cnt_unk / cnt_all
    print(cnt_unk)
    print(cnt_all)
    print(perc_unk)

    #-------------------------------------------
