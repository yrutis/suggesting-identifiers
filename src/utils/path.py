from dotmap import DotMap
import os
import src.utils.config as config_loader



#------------------------------------------------------------------------------------------------------------------

#JSON config files
simpleNN_config_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'configs'), "simpleNN.json")

LSTM_config_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'configs'), "LSTM.json")

seq2seq_config_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'configs'), "seq2seq.json")

#------------------------------------------------------------------------------------------------------------------


# path to model folder
model_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                    'models')

# path to report folder
report_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                    'reports')

