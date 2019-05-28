import pandas as pd

import src.data.utils.helper_functions as helper_functions
import logging
import os
from sklearn.model_selection import train_test_split


#%% get logger to work
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
filename = "all_methods_train_without_platform-subtoken"

# get logger
logger = logging.getLogger(__name__)
logger.info(filename)


data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
processed_decoded_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'processed'), 'decoded'),
                                           filename + '.json')  # get decoded path
logger.info(processed_decoded_full_path)

#%%
df = pd.read_json(processed_decoded_full_path, orient='records')

#%%get shape of train and val
train, val, _, _ = train_test_split(df['methodBodyCleaned'], df['methodName'], test_size=0.2, random_state=200)

len_train = train.shape
len_val = val.shape



#%% calc method body length
train_method_body, _, _, _ = train_test_split(df['methodBodyCleaned'], df['methodName'], test_size=0.2,
                                                    random_state=200)

len_method_body = train_method_body.apply(helper_functions.compute_col_length)
len_method_body = len_method_body.describe()


#%% calc method body length
train_method_body_splitted, _, _, _ = train_test_split(df['methodBodySplitted'], df['methodName'], test_size=0.2,
                                                    random_state=200)

len_method_body_splitted = train_method_body_splitted.apply(helper_functions.compute_col_length)
len_method_body_splitted = len_method_body_splitted.describe()

#%%
train_parameters, _, _, _ = train_test_split(df['parameters'], df['methodName'], test_size=0.2,
                                                    random_state=200)

len_parameters = train_parameters.apply(helper_functions.compute_col_length)
len_parameters = len_parameters.describe()

#%%
train_parameters_splitted, _, _, _ = train_test_split(df['parametersSplitted'], df['methodName'], test_size=0.2,
                                                    random_state=200)

len_parameters_splitted = train_parameters_splitted.apply(helper_functions.compute_col_length)
len_parameters_splitted = len_parameters_splitted.describe()

#%%

window_size_params = 2
window_size_body = 8

df['parameters'] = df['parameters'].apply(helper_functions.get_first_x_elem, args=(window_size_params,))
df['methodBodyCleaned'] = df['methodBodyCleaned'].apply(helper_functions.get_first_x_elem, args=(window_size_body,))
df["concatMethodBodyCleaned"] = df['Type'].map(lambda x: [x]) + df["parameters"] + df["methodBodyCleaned"]


#%%

import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



#%%

max_input_elemts = 1 + window_size_params + window_size_body

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
y_train_tokenized = list(map(lambda x: x[0], y_train_tokenized))
trainY = np.array(y_train_tokenized)



# tokenize just valX
x_test_seq = tokenizer.texts_to_sequences(x_test)
valX = pad_sequences(x_test_seq, maxlen=max_input_elemts, value=0)

# tokenize just testY
y_test = list(y_test)
y_test_tokenized = tokenizer.texts_to_sequences(y_test)
y_test_tokenized = list(map(lambda x: x[0], y_test_tokenized))
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
print(cnt_unk)
print(cnt_all)
print(perc_unk)

train_df = train_df.drop(train_df[train_df['trainY'] == 1].sample(frac=.5).index)
cnt_unk_removed = len(train_df[(train_df['trainY'] == 1)])
cnt_all_removed = len(train_df.index)
perc_unk_train_removed = cnt_unk_removed / cnt_all_removed
print(cnt_unk_removed)
print(cnt_all_removed)
print(perc_unk_train_removed)

trainX = np.array(train_df['trainX'].values.tolist())
trainY = train_df['trainY'].values

#-------------------------------------------
