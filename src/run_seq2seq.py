from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from datetime import datetime
from random import randint
from pickle import dump
import logging
import os
import json
import tensorflow as tf


import src.data.prepare_data_subtoken as prepare_data
from src.models.Seq2SeqModel import Seq2SeqModel
import src.utils.config as config_loader
import src.utils.path as path_file
from src.trainer.Seq2SeqTrain import Seq2SeqTrain


def main(config_path):
    # get logger
    logger = logging.getLogger(__name__)

    config = config_loader.get_config_from_json(config_path)

    FLAGS = tf.app.flags.FLAGS
    #define some tf flags

    tf.app.flags.DEFINE_string('data', config.data_loader.name,
                               'must be either Android-Universal-Image-Loader or all_methods_train')
    tf.app.flags.DEFINE_integer('window_size_body', config.data_loader.window_size_body, 'must be between 2+')
    tf.app.flags.DEFINE_integer('window_size_params', config.data_loader.window_size_params, 'must be between 2+')
    tf.app.flags.DEFINE_integer('window_size_name', config.data_loader.window_size_name, 'must be between 2+')
    tf.app.flags.DEFINE_integer('epochs', config.trainer.num_epochs, 'must be between 1-100')
    tf.app.flags.DEFINE_integer('batch_size', config.trainer.batch_size, 'must be a power of 2 2^1 - 2^6')


    config.data_loader.name = FLAGS.data
    logger.info("data used is {}".format(config.data_loader.name))

    config.data_loader.window_size_body = FLAGS.window_size_body
    logger.info("window size body is {}".format(config.data_loader.window_size_body))

    config.data_loader.window_size_params = FLAGS.window_size_params
    logger.info("window size params is {}".format(config.data_loader.window_size_params))

    config.data_loader.window_size_name = FLAGS.window_size_name
    logger.info("window size name is {}".format(config.data_loader.window_size_name))

    config.trainer.num_epochs = FLAGS.epochs
    logger.info("epochs num is {}".format(config.trainer.num_epochs))

    config.trainer.batch_size = FLAGS.batch_size
    logger.info("batch size is {}".format(config.trainer.batch_size))


    #get data
    trainX, trainY, valX, valY, tokenizer, vocab_size, \
    window_size, max_output_elemts = prepare_data.main(config.data_loader.name,
                                                  config.data_loader.window_size_body,
                                                  config.data_loader.window_size_params,
                                                  config.data_loader.window_size_name)

    word_index = tokenizer.word_index
    logger.info('Found {} unique tokens.'.format(len(word_index) + 1))

    vocab_size = len(word_index) + 1

    print(FLAGS.data)

    #create unique report folder
    random_nr = randint(0, 10000)
    unique_folder_key = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S') + "-" + str(random_nr)
    report_folder = os.path.join(path_file.report_folder, 'reports-' + config.name + '-' + unique_folder_key)

    os.mkdir(report_folder)


    # write in report folder
    with open(os.path.join(report_folder, config.name+'.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    logger.info("create seq2seq Model...")
    model = Seq2SeqModel(context_vocab_size=vocab_size,
                       windows_size=window_size,
                       config=config, report_folder=report_folder)

    data = [trainX, trainY, valX, valY]


    logger.info("create trainer...")
    trainer2 = Seq2SeqTrain(model=model.model,
                            encoder_model=model.encoder_model,
                            decoder_model=model.decoder_model,
                            data=data,
                            tokenizer=tokenizer, config=config,
                            report_folder=report_folder)

    logger.info("start simpleNN training...")
    trainer2.train()


    # %% idx2word

    # Creating a reverse dictionary
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

    # Function takes a tokenized sentence and returns the words
    def sequence_to_text(list_of_indices):
        # Looking up words in dictionary
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        return (words)


    # %% generate some method names
    encoder_input_data = valX[0]
    decoder_input_data = valX[1]
    i = 0
    correct = 0
    amnt = 10
    while i < amnt:
        input_seq = encoder_input_data[i: i + 1]
        input_seq_list = input_seq.tolist()[0]  # get in right format for tokenizer
        correct_output = decoder_input_data[i: i + 1]
        correct_output_list = correct_output.tolist()[0]  # get in right format for tokenizer
        decoded_correct_output_list = sequence_to_text(correct_output_list)

        input_enc = sequence_to_text(input_seq_list)

        print("this is the input seq decoded: {}".format(input_enc))
        decoded_sentence = trainer2.predict(input_seq)
        print("Predicted: {}".format(decoded_sentence))
        print("Correct: {}".format(decoded_correct_output_list))
        i += 1

        if decoded_sentence == decoded_correct_output_list:
            correct += 1

    accuracy = correct / amnt
    print("total accuracy %.2f%%" % accuracy)




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    seq2seq_config_path = path_file.seq2seq_config_path
    main(seq2seq_config_path)