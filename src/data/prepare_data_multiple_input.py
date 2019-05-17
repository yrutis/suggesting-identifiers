#%%import some stuff

import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import src.data.utils.helper_functions as helper_functions
import logging
import os


#%%init important variables

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
filename = 'all_methods_train_without_platform'
window_size = 8

#%%init important variables
# get logger
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

print(df.shape)
print(train.shape)
print(val.shape)

#%%

#get all the input vocab only for vocab size
all_input_vocab_x = train['concatMethodBodyCleaned']

#retrieve different inputs seperately
method_body_x= train['methodBodyCleaned']
parameters_x = train['parameters']
type_x = train['Type']
method_name_x = train['methodName']

#%%

#generate training vocab (all tokens that appear >=3 are considered
training_vocab_x = helper_functions.get_training_vocab(all_input_vocab_x, is_for_x=True)
training_vocab_y = helper_functions.get_training_vocab(method_name_x, is_for_x=False)

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
method_body_x = list(map(helper_functions.get_into_tokenizer_format, method_body_x))
parameters_x = list(map(helper_functions.get_into_tokenizer_format, parameters_x))



#%%

type_x = list(type_x)

# tokenize just method_body_x
sequences = tokenizer.texts_to_sequences(method_body_x)
method_body_x_padded = pad_sequences(sequences, maxlen=window_size, value=0)

# tokenize just parameters_x
sequences = tokenizer.texts_to_sequences(parameters_x)
parameters_x_padded = pad_sequences(sequences, maxlen=window_size, value=0)

# tokenize just type_x
sequences = tokenizer.texts_to_sequences(type_x)
type_x_padded = pad_sequences(sequences, maxlen=window_size, value=0)


#%%
# tokenize just trainY
y_train = list(method_name_x)
y_train_tokenized = tokenizer.texts_to_sequences(y_train)
y_train_tokenized = list(map(lambda x: x[0], y_train_tokenized))
trainY = np.array(y_train_tokenized)


#%%
# tokenize all Validation