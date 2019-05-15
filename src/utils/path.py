from dotmap import DotMap
import os


#------------------------------------------------------------------------------------------------------------------

#JSON config files
simpleNN_config_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'configs'), "simpleNN.json")

LSTM_config_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'configs'), "LSTM.json")

LSTMBid_config_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'configs'), "LSTMBid.json")

seq2seq_config_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'configs'), "seq2seq.json")

#------------------------------------------------------------------------------------------------------------------

#optimize path
LSTM_opt_config_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'configs'), "LSTM_opt.json")

#------------------------------------------------------------------------------------------------------------------



# path to model folder
model_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                    'models')

# path to report folder
report_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                    'reports')

