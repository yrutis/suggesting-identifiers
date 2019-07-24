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

df_simpleNN = pd.read_csv('predictions_test_javasmall_simpleNN.csv')
df_GRU = pd.read_csv('predictions_test_javasmall_GRU.csv')
df_LSTM = pd.read_csv('predictions_test_javasmall_LSTM.csv')
df_LSTMBid = pd.read_csv('predictions_test_javasmall_LSTMBid.csv')

#%%remove UNK method names
df_simpleNN = df_simpleNN[df_simpleNN['Correct'] != 'UNK']
df_GRU = df_GRU[df_GRU['Correct'] != 'UNK']
df_LSTM = df_LSTM[df_LSTM['Correct'] != 'UNK']
df_LSTMBid = df_LSTMBid[df_LSTMBid['Correct'] != 'UNK']


#%% get data for plot
threadshold_list_simpleNN, acc_list_simpleNN = get_accuracy_by_confidence(df_simpleNN)
threadshold_list_GRU, acc_list_GRU = get_accuracy_by_confidence(df_GRU)
threadshold_list_LSTM, acc_list_LSTM = get_accuracy_by_confidence(df_LSTM)
threadshold_list_LSTMBid, acc_list_LSTMBid = get_accuracy_by_confidence(df_LSTMBid)

#%% actual plotting

fig, ax = plt.subplots()
ax.plot(threadshold_list_simpleNN, acc_list_simpleNN, 'k--', label='feed-forward Token Model')
ax.plot(threadshold_list_GRU, acc_list_GRU, 'b', label='GRU Token Model')
ax.plot(threadshold_list_LSTM, acc_list_LSTM, 'g', label='LSTM Token Model')
ax.plot(threadshold_list_LSTMBid, acc_list_LSTMBid, 'y', label='bidirectional LSTM Token Model')
plt.xlabel('Suggestion Frequency')
plt.ylabel('Accuracy')
#plt.axis([0, 1, 0, 1])
plt.title('Accuracy of different Models for the Java-small Dataset')

legend = ax.legend(loc='upper right', shadow=True, fontsize='large')

# Put a nicer background color on the legend.
#legend.get_frame().set_facecolor('C0')

plt.show()
plt.savefig("acc_suggest_frequ.png")