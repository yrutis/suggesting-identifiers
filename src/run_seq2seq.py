from numpy.random import seed
from src.subtoken_approach.evaluator.EvaluatorSubtoken import Evaluator

seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from datetime import datetime
from random import randint
from pickle import load
import logging
import os
import json
import tensorflow as tf

import src.subtoken_approach.data.prepare_data_subtoken as prepare_data
import src.subtoken_approach.data.prepare_data_subtoken_test as prepare_data_test
from src.subtoken_approach.models.Seq2SeqModel import Seq2SeqModel
import src.utils.config as config_loader
import src.utils.path as path_file
from src.subtoken_approach.trainer.Seq2SeqTrain import Seq2SeqTrain
from src.Vocabulary.Vocabulary import Vocabulary


def train_model(config, report_folder):
    # get logger
    logger = logging.getLogger(__name__)

    # get data
    all_train, all_val, vocab_size, window_size, max_output_elemts, data_storage\
        = prepare_data.main(config=config, report_folder=report_folder)

    logger.info('Found {} unique tokens.'.format(vocab_size))


    logger.info("create seq2seq Model...")
    model = Seq2SeqModel(context_vocab_size=vocab_size,
                         windows_size=window_size,
                         config=config, report_folder=report_folder)



    if config.mode == 'train':
        # build graph
        model.build_model()

        logger.info("create trainer...")
        trainer = Seq2SeqTrain(model=model.model,
                               encoder_model=model.encoder_model,
                               decoder_model=model.decoder_model,
                               config=config,
                               report_folder=report_folder)
        logger.info("start seq2seq training...")
        trainer.train(all_train, all_val, window_size, max_output_elemts, vocab_size, data_storage)

        logger.info("saving the model...")
        #trainer.model.save(os.path.join(report_folder, "best_model.h5"))
        trainer.model.save_weights(os.path.join(report_folder, "best_model_weights.h5"))

    else:
        trained_model_weights_path = os.path.join(os.path.join(path_file.model_folder, config.data_loader.name),
                                          config.name + '_weights_model_window_size_body_' + str(
                                              config.data_loader.window_size_body)
                                          + '_params_' + str(config.data_loader.window_size_params) + '.h5')

        logger.info("loading the model from {}".format(trained_model_weights_path))
        model.load_model_weights(trained_model_weights_path)
        logger.info("create trainer...")
        trainer = Seq2SeqTrain(model=model.model,
                               encoder_model=model.encoder_model,
                               decoder_model=model.decoder_model,
                               config=config,
                               report_folder=report_folder)

    return trainer




def eval_model(config, report_folder, trainer:Seq2SeqTrain):
    logger = logging.getLogger(__name__)
    with open(os.path.join(report_folder, 'tokenizer.pkl'), "rb") as input_file:
        tokenizer = load(input_file)


    testX, testY = prepare_data_test.main(config.data_loader.name,
                                          tokenizer,
                                          config.data_loader.window_size_body,
                                          config.data_loader.window_size_params,
                                          config.data_loader.window_size_name)


    evaluator = Evaluator(trainer=trainer, report_folder=report_folder)

    accuracy, precision, recall, f1, \
    accuracy_k1, precision_k1, recall_k1, f1_k1 = \
        evaluator.evaluate(testX=testX, testY=testY, Vocabulary=Vocabulary,
                           tokenizer=tokenizer, trainer=trainer, is_attention=False)

    logger.info("acc k100 {} prec k100 {} recall k100 {} f1 k100 {}".format(accuracy, precision, recall, f1))
    logger.info("acc k1 {} prec k1 {} recall k1 {} f1 k1 {}".format(accuracy_k1, precision_k1, recall_k1, f1_k1))


#evaluator.get_accuracy_precision_recall_f1_score(correct, predictions_k_1, 'k1')
    #evaluator.get_accuracy_precision_recall_f1_score(correct, predictions_k_100, 'k100')


def main(config_path):
    # get logger
    logger = logging.getLogger(__name__)

    config = config_loader.get_config_from_json(config_path)

    FLAGS = tf.app.flags.FLAGS
    #define some tf flags

    tf.app.flags.DEFINE_string('data', config.data_loader.name,
                               'must be either must be either java-small-project-split or java-small')
    tf.app.flags.DEFINE_string('mode', config.mode,
                               'must be either train or eval')
    tf.app.flags.DEFINE_integer('window_size_body', config.data_loader.window_size_body, 'must be between 2+')
    tf.app.flags.DEFINE_integer('window_size_params', config.data_loader.window_size_params, 'must be between 2+')
    tf.app.flags.DEFINE_integer('window_size_name', config.data_loader.window_size_name, 'must be between 2+')
    tf.app.flags.DEFINE_integer('partition', config.data_loader.partition, 'should be between 10k-100k and dividable by batch_size')
    tf.app.flags.DEFINE_integer('epochs', config.trainer.num_epochs, 'must be between 1-100')
    tf.app.flags.DEFINE_integer('batch_size', config.trainer.batch_size, 'must be a power of 2 2^1 - 2^6')


    config.data_loader.name = FLAGS.data
    logger.info("data used is {}".format(config.data_loader.name))

    config.mode = FLAGS.mode
    logger.info("mode is {}".format(config.mode))

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

    config.data_loader.partition = FLAGS.partition
    logger.info("partition is {}".format(config.data_loader.partition))


    #create unique report folder
    random_nr = randint(0, 10000)
    unique_folder_key = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S') + "-" + str(random_nr)
    report_folder = os.path.join(path_file.report_folder, 'reports-' + config.name + '-' + unique_folder_key)

    os.mkdir(report_folder)

    # write in report folder
    with open(os.path.join(report_folder, config.name+'.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)
        

    trainer = train_model(config=config, report_folder=report_folder)

    eval_model(config=config, report_folder=report_folder, trainer=trainer)




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    seq2seq_config_path = path_file.seq2seq_config_path
    main(seq2seq_config_path)