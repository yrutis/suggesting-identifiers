from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


import logging
import os
import tensorflow as tf


import src.data.prepare_data_subtoken_test as prepare_data
import src.utils.config as config_loader
import src.utils.path as path_file
from src.trainer.Seq2SeqTrain import Seq2SeqTrain
from src.evaluator.EvaluatorSubtoken import Evaluator


def main(config_path):
    # get logger
    logger = logging.getLogger(__name__)

    #get report folder
    FLAGS = tf.app.flags.FLAGS

    tf.app.flags.DEFINE_string('report_folder_ID', '2019-06-08-09-57-06-7158',
                               'must be a valid report folder ID')

    report_folder_ID = FLAGS.report_folder_ID


    report_folder = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports'),
                                 'reports-seq2seq-'+report_folder_ID)


    logger.info("report folder path is {}".format(report_folder))

    #get config_path and load the corresponding config file
    config_path = os.path.join(report_folder, 'seq2seq.json')
    config = config_loader.get_config_from_json(config_path)

    #get data
    testX, testY, tokenizer, vocab_size, \
    max_input_elemts, max_output_elemts = prepare_data.main(config.data_loader.name,
                                                            report_folder,
                                                            config.data_loader.window_size_body,
                                                            config.data_loader.window_size_params,
                                                            config.data_loader.window_size_name)


    logger.info('Found {} unique tokens.'.format(vocab_size))


    #TODO create new report_folder and safe everything there
    #create unique new report folder
    #random_nr = randint(0, 10000)
    #unique_folder_key = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S') + "-" + str(random_nr)
    #report_folder = os.path.join(path_file.report_folder, 'reports-' + config.name + '-' + unique_folder_key)

    #os.mkdir(report_folder)


    # write in report folder
    #with open(os.path.join(report_folder, config.name+'.json'), 'w') as outfile:
    #    json.dump(config, outfile, indent=4)

    logger.info("no need to create a seq2seq Model, just load the trained one")

    data = [testX, testY, 0, 0]


    logger.info("create trainer...")
    trainer2 = Seq2SeqTrain(model=0,
                            encoder_model=0,
                            decoder_model=0,
                            data=data,
                            tokenizer=tokenizer, config=config,
                            report_folder=report_folder)

    logger.info("load a pretrained model...")
    trainer2.load_trained_model(report_folder)


    # %% idx2word

    # Creating a reverse dictionary
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

    # Function takes a tokenized sentence and returns the words
    def sequence_to_text(list_of_indices):
        # Looking up words in dictionary
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        return (words)

    # %% generate some method names
    input = []
    predictions = []
    correct = []
    i = 0
    while i < 100:
        input_seq = testX[i: i + 1]
        correct_output = testY[i: i + 1]
        correct_output_list = correct_output.tolist()[0]
        decoded_correct_output_list = sequence_to_text(correct_output_list)
        input_seq_list = input_seq.tolist()[0]  # get in right format for tokenizer

        # input
        input_enc = sequence_to_text(input_seq_list)
        print("Input: {}".format(input_enc))
        input.append(input_enc)

        # prediction
        decoded_sentence = trainer2.predict(input_seq)
        decoded_sentence_list = decoded_sentence.split()
        print("Prediction List: {}".format(decoded_sentence_list))
        predictions.append(decoded_sentence_list)

        # correct
        print("Correct: {}".format(decoded_correct_output_list))
        correct.append(decoded_correct_output_list)

        i += 1

    evaluator = Evaluator(trainer2.model, report_folder)
    accuracy, precision, recall, f1 = evaluator.get_accuracy_precision_recall_f1_score(correct, predictions)
    print("coming from main {} {} {} {}".format(accuracy, precision, recall, f1))



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    seq2seq_config_path = path_file.seq2seq_config_path
    main(seq2seq_config_path)