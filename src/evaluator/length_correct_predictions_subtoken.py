import ast

import pandas as pd

from src.data.utils import helper_functions


# show length of method name and input and compare to the one from the training set



def compute_col_length(row):
    '''

    :param method_body: methodbody
    :return: length for each methodybody
    '''

    #row = ast.literal_eval(row) #change to list
    length = len(row)
    return length

#%%
df = pd.read_csv('correct_predictions_seq2seq_attention.csv')


#%%
df['prediction_length'] = df['prediction'].apply(compute_col_length)
len_method_name = df['prediction_length'].describe()

#%%

def filter_results(subtoken_list):
    subtoken_list = ast.literal_eval(subtoken_list) #change to list
    return filter_results

#%%
def filter_now(subtoken_list):
    subtoken_list = list(filter(None, subtoken_list))
    subtoken_list = [str(x) for x in subtoken_list]
    subtoken_list = list(filter(lambda x: x != "starttoken", subtoken_list))
    subtoken_list = list(filter(lambda x: x != "endtoken", subtoken_list))
    return subtoken_list

#%%
df['input'] = df['input'].apply(filter_results)

#%%
df['input'] = df['input'].apply(filter_now)

#%%
df['input_length'] = df['input'].apply(compute_col_length)

#%%
len_input = df['input_length'].describe()


