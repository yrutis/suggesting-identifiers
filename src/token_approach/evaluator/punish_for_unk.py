import pandas as pd

df_simpleNN = pd.read_csv('predictions_test_javasmall_simpleNN.csv')
df_GRU = pd.read_csv('predictions_test_javasmall_GRU.csv')
df_LSTM = pd.read_csv('predictions_test_javasmall_LSTM.csv')
df_LSTMBid = pd.read_csv('predictions_test_javasmall_LSTMBid.csv')

assert(df_simpleNN.shape[0] == df_GRU.shape[0] == df_LSTM.shape[0] == df_LSTMBid.shape[0])

#%%

def change_unk_value(row):
    if row == 'UNK':
        row = 'UNK2'
    return row


#%%
complete_length = df_simpleNN.shape[0]

#%%
df_simpleNN['Correct'] = df_simpleNN['Correct'].apply(change_unk_value)
df_GRU['Correct'] = df_GRU['Correct'].apply(change_unk_value)
df_LSTM['Correct'] = df_LSTM['Correct'].apply(change_unk_value)
df_LSTMBid['Correct'] = df_LSTMBid['Correct'].apply(change_unk_value)
#%%
df_simpleNN = df_simpleNN[df_simpleNN['Correct'] == df_simpleNN['Prediction']]
df_GRU = df_GRU[df_GRU['Correct'] == df_GRU['Prediction']]
df_LSTM = df_LSTM[df_LSTM['Correct'] == df_LSTM['Prediction']]
df_LSTMBid = df_LSTMBid[df_LSTMBid['Correct'] == df_LSTMBid['Prediction']]

#%%

#df_simpleNN = df_simpleNN[df_simpleNN['Prediction'] != 'UNK']
#df_GRU = df_GRU[df_GRU['Prediction'] != 'UNK']
#df_LSTM = df_LSTM[df_LSTM['Prediction'] != 'UNK']
#df_LSTMBid = df_LSTMBid[df_LSTMBid['Prediction'] != 'UNK']

#%%


length_simpleNN = df_simpleNN.shape[0] / complete_length
length_GRU = df_GRU.shape[0] / complete_length
length_LSTM = df_LSTM.shape[0] / complete_length
length_LSTMBid = df_LSTMBid.shape[0] / complete_length
