import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ast

#%%

def remove_none(row):
    row = ast.literal_eval(row) #change to list
    after_remove_none = list(filter(None, row))
    return after_remove_none

def count_input(row):
    sample_length = len(row)
    return sample_length


#%% get accuracy and input length

def get_accuracy_and_input_length(df):
    '''
    :param df: dataframe
    1) remove nones, 2) count input elements for each sample, 3) save for some threadshold
    :return: input_length, acc_list
    '''

    df['Input_without_none'] = df['Input'].apply(remove_none) #remove nones
    df['Input_length'] = df['Input_without_none'].apply(count_input) #compute length
    input_length = list(range(5, 25, 2))

    acc_list = []

    for length in input_length:
        df_temp = df[df['Input_length'] <= length]
        df_temp_correct = df_temp[df_temp['Correct'] == df_temp['Prediction']]

        acc = df_temp_correct.shape[0] / df_temp.shape[0]
        acc_list.append(acc)

    input_length = np.array(input_length)
    acc_list = np.array(acc_list)

    return input_length, acc_list



#%% load data

df_simpleNN = pd.read_csv('predictions_test_simpleNN.csv')
df_GRU = pd.read_csv('predictions_test_GRU.csv')
df_LSTM = pd.read_csv('predictions_test_LSTM.csv')
df_LSTMBid = pd.read_csv('predictions_test_LSTMBid.csv')

#%%
df_simpleNN = df_simpleNN[df_simpleNN['Correct'] != 'UNK']
df_GRU = df_GRU[df_GRU['Correct'] != 'UNK']
df_LSTM = df_LSTM[df_LSTM['Correct'] != 'UNK']
df_LSTMBid = df_LSTMBid[df_LSTMBid['Correct'] != 'UNK']

#%%
input_length_simpleNN, acc_list_simpleNN = get_accuracy_and_input_length(df_simpleNN)
input_length_GRU, acc_list_GRU = get_accuracy_and_input_length(df_GRU)
input_length_LSTM, acc_list_LSTM = get_accuracy_and_input_length(df_LSTM)
input_length_LSTMBid, acc_list_LSTMBid = get_accuracy_and_input_length(df_LSTMBid)

#%% plotting

fig, ax = plt.subplots()
ax.plot(input_length_simpleNN, acc_list_simpleNN, 'k--', label='simpleNN')
ax.plot(input_length_GRU, acc_list_GRU, 'b', label='GRU')
ax.plot(input_length_LSTM, acc_list_LSTM, 'g', label='LSTM')
ax.plot(input_length_LSTMBid, acc_list_LSTMBid, 'y', label='LSTMBid')
plt.xlabel('Input length')
plt.ylabel('Accuracy')
plt.title('Accuracy of different Models for the Allamanis Dataset')

legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
#legend.get_frame().set_facecolor('C0')

plt.show()
