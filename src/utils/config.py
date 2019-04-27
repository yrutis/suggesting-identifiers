import json
import os
from dotmap import DotMap
import src.utils.path as path_file

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    config = DotMap(config_dict)
    return config



def load_configs():

    simpleNN_config_path = path_file.simpleNN_config_path
    print(simpleNN_config_path)
    simpleNN_config = get_config_from_json(simpleNN_config_path)

    LSTM_config_path = path_file.LSTM_config_path
    LSTM_config = get_config_from_json(LSTM_config_path)

    return simpleNN_config, LSTM_config


simpleNN_config, LSTM_config = load_configs()
