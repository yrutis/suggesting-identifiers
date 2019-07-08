from random import randint


import json
import logging
import os
from datetime import datetime
from random import randint

import src.utils.path as path_file
from src.data import prepare_data_token
import src.utils.config as config_loader

def data():

    token_model_config_path = path_file.token_model_config_path
    token_model_config = config_loader.get_config_from_json(token_model_config_path)


    # create unique report folder
    random_nr = randint(0, 10000)
    unique_folder_key = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S') + "-" + str(random_nr)
    report_folder = path_file.report_folder
    report_folder_token_model = os.path.join(report_folder, 'reports-hyperopt' + token_model_config.name + '-' + unique_folder_key)

    os.mkdir(report_folder_token_model)

    # get data
    trainX, trainY, valX, valY, tokenizer, always_unknown_train, always_unknown_test, window_size = \
        prepare_data_token.main(token_model_config.data_loader.name,
                                token_model_config.data_loader.window_size_params,
                                token_model_config.data_loader.window_size_body,
                                remove_val_unk=0.6, report_folder=report_folder)

    vocab_size = len(tokenizer.word_index) + 1
    print('Found {} unique tokens.'.format(vocab_size))

    return trainX, trainY, valX, valY, vocab_size, token_model_config, report_folder_token_model, window_size
