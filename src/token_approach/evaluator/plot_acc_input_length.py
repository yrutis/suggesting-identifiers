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
    input_length = list(range(5, 34, 2))

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

df = pd.read_csv('predictions_test_simpleNN.csv')

input_length_simpleNN, acc_list_simpleNN = get_accuracy_and_input_length(df)

#%% plotting

fig, ax = plt.subplots()
ax.plot(input_length_simpleNN, acc_list_simpleNN, 'k--', label='simpleNN')
plt.xlabel('Input length')
plt.ylabel('Accuracy')
plt.title('Accuracy of different Models for the Allamanis Dataset')

legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
#legend.get_frame().set_facecolor('C0')

plt.show()
