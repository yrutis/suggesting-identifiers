import os
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
from sklearn import metrics
from src.trainer.AbstractTrainSubtoken import AbstractTrainSubtoken
from src.trainer.Seq2SeqAttentionTrain import Seq2SeqAttentionTrain
import matplotlib


class Evaluator(object):
    def __init__(self, trainer:AbstractTrainSubtoken, report_folder):
        self.__trained_model = trainer
        self.report_folder = report_folder
        self.fig_folder = os.path.join(report_folder, 'figures')
        self.correct_predictions_file = os.path.join(self.report_folder, 'correct_predictions.csv')


    def load_correct_prediction_file(self, input, prediction, correct, i):
        correct_prediction = {'input': [input],
                              'prediction': [prediction],
                              'correct': [correct],
                              'i': [i]
                              }

        correct_prediction = pd.DataFrame(correct_prediction, columns=['input', 'prediction', 'correct', 'i'])


        if os.path.exists(self.correct_predictions_file):



            correct_predictions = pd.read_csv(self.correct_predictions_file)

            correct_predictions = correct_predictions.append(correct_prediction, sort=False) #append

        else:
            correct_predictions = correct_prediction

        df = correct_predictions.to_csv(self.correct_predictions_file, index=False)



    def evaluate(self, testX, testY, Vocabulary, tokenizer, trainer:AbstractTrainSubtoken):
        logger = logging.getLogger(__name__)

        complete_true = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0

        i = 0
        progress = 0
        while i < testX.shape[0]:

            if progress % 10 == 0:
                logger.info("{} / {} completed".format(progress, testX.shape[0]))
            progress +=1

            input_seq = testX[i: i + 1]
            correct_output = testY[i: i + 1]

            decoded_correct_output_list = Vocabulary.revert_back(tokenizer=tokenizer, sequence=correct_output.tolist()[0])
            input_seq_dec = Vocabulary.revert_back(tokenizer=tokenizer, sequence=input_seq.tolist()[0])

            decoded_sentence_k_100 = trainer.predict(tokenizer=tokenizer, input_seq=input_seq, k=100, return_top_n=1)
            decoded_sentence = trainer.predict(tokenizer=tokenizer, input_seq=input_seq, k=1, return_top_n=1)

            current_result = decoded_sentence_k_100 #just for attention


            decoded_sentence_k_100 = self.filter_results(decoded_sentence_k_100[0])
            decoded_correct_output_list = self.filter_results(decoded_correct_output_list)


            current_complete_true, current_true_positive, \
            current_false_positive, current_false_negative = \
                self.get_subtoken_stats(decoded_correct_output_list, decoded_sentence_k_100)

            complete_true += current_complete_true
            true_positive += current_true_positive
            false_positive += current_false_positive
            false_negative += current_false_negative


            if ((current_complete_true == 1) and (len(decoded_correct_output_list)>0)): #not just unk
                if isinstance(trainer, Seq2SeqAttentionTrain):
                    # logger.info("I am an attention")
                    attention_plot = current_result[1]

                    attention_plot = attention_plot[:len(current_result[0]), :len(input_seq_dec)]
                    self.plot_attention(attention_plot, input_seq_dec, current_result[0], i)

                #logger.info("current_complete_true == 1 {}".format((current_complete_true == 1)))
                #logger.info("len(decoded_correct_output_list)>0) {}".format((len(decoded_correct_output_list)>0))) #not just unk
                #logger.info("Complete True! input: {} \n correct: {}\n prediction: {}".format(input_seq_dec, decoded_correct_output_list, decoded_sentence_k_100))

                self.load_correct_prediction_file(input=input_seq_dec, prediction=decoded_sentence_k_100,
                                          correct=decoded_correct_output_list, i = i)



            i += 1

        accuracy, precision, recall, f1 = self.calculate_results(complete_true, testX.shape[0], true_positive, false_positive, false_negative)

        return accuracy, precision, recall, f1



    @staticmethod
    def calculate_results(complete_true, total, true_positive, false_positive, false_negative):
        accuracy = 0
        if total != 0:
            accuracy = complete_true / total
        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0
        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        return accuracy, precision, recall, f1

    @staticmethod
    def filter_results(subtoken_list):
        subtoken_list = list(filter(None, subtoken_list))
        subtoken_list = [str(x) for x in subtoken_list]
        subtoken_list = list(filter(lambda x: x != "starttoken", subtoken_list))
        subtoken_list = list(filter(lambda x: x != "endtoken", subtoken_list))
        subtoken_list = list(filter(lambda x: x != "UNK", subtoken_list))  # oov
        subtoken_list = list(filter(lambda x: x != "True", subtoken_list))  # oov
        subtoken_list = list(filter(lambda x: x != '1', subtoken_list))  # oov
        return subtoken_list


    def get_subtoken_stats(self, correct, predicted):
        # check if in vocabulary
        complete_true = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0
        if ''.join(correct) == ''.join(predicted):
            true_positive += len(correct)
            complete_true += 1
            return complete_true, true_positive, false_positive, false_negative #don't need to check the rest

        for subtok in predicted:
            if subtok in correct:
                true_positive += 1
            else:
                false_positive += 1
        for subtok in correct:
            if not subtok in predicted:
                false_negative += 1

        return complete_true, true_positive, false_positive, false_negative

    # function for plotting the attention weights
    def plot_attention(self, attention, sentence, predicted_sentence, i):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(attention, cmap='viridis')

        fontdict = {'fontsize': 14}

        sentence = list(map(lambda x: str(x), sentence)) #to get nones

        ax.set_xticks(range(len(sentence)))
        ax.set_yticks(range(len(predicted_sentence)))

        ax.set_xticklabels(sentence, fontdict=fontdict, rotation=90)
        ax.set_yticklabels(predicted_sentence, fontdict=fontdict)

        if not os.path.exists(self.fig_folder):
            os.mkdir(self.fig_folder)

        plt.savefig(os.path.join(self.fig_folder, 'fig-' + str(i) + '.png'))
        plt.close()

