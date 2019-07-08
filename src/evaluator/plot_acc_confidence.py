import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_accuracy_by_confidence(df):
    '''
    :param df: df
    1) sort by prob in descending order 2) cut df 3) calc acc
    :return: threadshold list, accuracy list
    '''


    df.sort_values(by=['PredictionProb'], inplace=True, ascending=False)

    threadshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    acc_list = []

    for threadshold in threadshold_list:
        length = int(threadshold * df.shape[0])

        #print(df.shape[0])
        df_01 = df[:length]
        #print(df_01.shape[0])
        df_01_correct = df_01[df_01['Correct'] == df_01['Prediction']]
        #print(df_01_correct.shape[0])

        acc_01 = df_01_correct.shape[0] / df_01.shape[0]
        #print(acc_01)
        acc_list.append(acc_01)


    threadshold_list = np.array(threadshold_list)
    acc_list = np.array(acc_list)

    return threadshold_list, acc_list


#%% load data

path_to_predictions = 'predictions_test.csv'
df = pd.read_csv('predictions_test.csv')

threadshold_list_simpleNN, acc_list_simpleNN = get_accuracy_by_confidence(df)

#%% actual plotting

fig, ax = plt.subplots()
ax.plot(threadshold_list_simpleNN, acc_list_simpleNN, 'k--', label='simpleNN')
plt.xlabel('Confidence')
plt.ylabel('Accuracy')
#plt.axis([0, 1, 0, 1])
plt.title('Accuracy of different Models for the Allamanis Dataset')

legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
#legend.get_frame().set_facecolor('C0')

plt.show()

