import os


#------------------------------------------------------------------------------------------------------------------

#JSON config files

token_model_config_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'configs'), "token_model.json")

simpleNN_config_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'configs'), "simpleNN.json")

GRU_config_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'configs'), "GRU.json")

LSTM_config_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'configs'), "LSTM.json")

LSTMBid_config_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'configs'), "LSTMBid.json")

seq2seq_config_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'configs'), "seq2seq.json")

seq2seq_attention_config_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'configs'), "Seq2SeqAttention.json")

#------------------------------------------------------------------------------------------------------------------



# path to model folder
model_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                    'models')

# path to report folder
report_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                    'reports')

