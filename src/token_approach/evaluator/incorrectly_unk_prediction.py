import pandas as pd

df_simpleNN = pd.read_csv('predictions_test_simpleNN.csv')
df_GRU = pd.read_csv('predictions_test_GRU.csv')
df_LSTM = pd.read_csv('predictions_test_LSTM.csv')
df_LSTMBid = pd.read_csv('predictions_test_LSTMBid.csv')

assert(df_simpleNN.shape[0] == df_GRU.shape[0] == df_LSTM.shape[0] == df_LSTMBid.shape[0])

#%%
complete_length = df_simpleNN.shape[0]

#%%
df_simpleNN = df_simpleNN[df_simpleNN['Correct'] != 'UNK']
df_GRU = df_GRU[df_GRU['Correct'] != 'UNK']
df_LSTM = df_LSTM[df_LSTM['Correct'] != 'UNK']
df_LSTMBid = df_LSTMBid[df_LSTMBid['Correct'] != 'UNK']

#%%
df_simpleNN = df_simpleNN[df_simpleNN['Prediction'] == 'UNK']
df_GRU = df_GRU[df_GRU['Prediction'] == 'UNK']
df_LSTM = df_LSTM[df_LSTM['Prediction'] == 'UNK']
df_LSTMBid = df_LSTMBid[df_LSTMBid['Prediction'] == 'UNK']

#%%

length_simpleNN = df_simpleNN.shape[0] / complete_length
length_GRU = df_GRU.shape[0] / complete_length
length_LSTM = df_LSTM.shape[0] / complete_length
length_LSTMBid = df_LSTMBid.shape[0] / complete_length

#%%

df_simpleNN_new = df_simpleNN[df_simpleNN['Correct'] == df_simpleNN['Prediction']]
