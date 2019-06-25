import json
from datetime import datetime
import logging
import os
from pickle import load
from random import randint

from src.Vocabulary.Vocabulary import Vocabulary
from src.evaluator.EvaluatorSubtoken import Evaluator

from src.models.Seq2SeqAttentionModel import Encoder, Decoder
from src.trainer.Seq2SeqAttentionTrain import Seq2SeqAttentionTrain

import src.data.prepare_data_subtoken as prepare_data
import src.data.prepare_data_subtoken_test as prepare_data_test

import tensorflow as tf

tf.enable_eager_execution()

import src.utils.config as config_loader
import src.utils.path as path_file

def get_start_token(report_folder):
    with open(os.path.join(report_folder, 'tokenizer.pkl'), "rb") as input_file:
        tokenizer = load(input_file)
    start_token = tokenizer.texts_to_sequences(['starttoken'])
    return start_token[0][0]


def train_model(config, report_folder):
    # get logger
    logger = logging.getLogger(__name__)



    # get data
    all_train, all_val, vocab_size, window_size, max_output_elemts, data_storage\
        = prepare_data.main(config.data_loader.name,
                            config.data_loader.window_size_body,
                            config.data_loader.window_size_params,
                            config.data_loader.window_size_name,
                            report_folder=report_folder, using_generator=True)

    start_token = get_start_token(report_folder)


    n_batches = len(all_train) // config.trainer.batch_size
    val_n_batches = len(all_val) // config.trainer.batch_size

    logger.info('Found {} unique tokens.'.format(vocab_size))

    logger.info("create seq2seq Attention Model...")
    encoder = Encoder(vocab_size, config)
    decoder = Decoder(vocab_size, config, encoder.embedding) #same embedding


    logger.info("create trainer...")
    trainer = Seq2SeqAttentionTrain(encoder=encoder, decoder=decoder, n_batches=n_batches, val_n_batches=val_n_batches,
                                    window_size=window_size, max_output_elements=max_output_elemts,
                                    config=config, report_folder=report_folder, start_token=start_token)

    logger.info("start training...")
    trainer.train()

    logger.info("deleting temp files...")

    return trainer

def eval_model(config, report_folder, trainer:Seq2SeqAttentionTrain):
    logger = logging.getLogger(__name__)
    with open(os.path.join(report_folder, 'tokenizer.pkl'), "rb") as input_file:
        tokenizer = load(input_file)


    testX, testY = prepare_data_test.main(config.data_loader.name,
                                          tokenizer,
                                          config.data_loader.window_size_body,
                                          config.data_loader.window_size_params,
                                          config.data_loader.window_size_name)

    evaluator = Evaluator(trainer=trainer, report_folder=report_folder)

    accuracy, precision, recall, f1 = evaluator.evaluate(testX=testX, testY=testY, Vocabulary=Vocabulary,
                                                         tokenizer=tokenizer, trainer=trainer)

    logger.info("acc {} prec {} recall {} f1 {}".format(accuracy, precision, recall, f1))


def main(config_path):
    # get logger
    logger = logging.getLogger(__name__)

    config = config_loader.get_config_from_json(config_path)

    FLAGS = tf.app.flags.FLAGS
    # define some tf flags

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

    # create unique report folder
    random_nr = randint(0, 10000)
    unique_folder_key = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S') + "-" + str(random_nr)
    report_folder = os.path.join(path_file.report_folder, 'reports-' + config.name + '-' + unique_folder_key)

    os.mkdir(report_folder)

    # write in report folder
    with open(os.path.join(report_folder, config.name + '.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    trainer = train_model(config=config, report_folder=report_folder)

    eval_model(config=config, report_folder=report_folder, trainer=trainer)




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    seq2seq_config_path = path_file.seq2seq_attention_config_path
    main(seq2seq_config_path)

