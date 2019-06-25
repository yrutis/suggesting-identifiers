import os
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
from sklearn import metrics
from src.trainer.AbstractTrainSubtoken import AbstractTrainSubtoken


class Evaluator(object):
    def __init__(self, trainer:AbstractTrainSubtoken, report_folder):
        self.__trained_model = trainer
        self.report_folder = report_folder



    def evaluate(self, testX, testY, Vocabulary, tokenizer, trainer:AbstractTrainSubtoken):
        logger = logging.getLogger(__name__)

        complete_true = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0

        i = 0
        progress = 0
        while i < testX.shape[0]:

            if progress % 1000 == 0:
                logger.info("{} / {} completed".format(progress, testX.shape[0]))
            progress +=1

            input_seq = testX[i: i + 1]
            correct_output = testY[i: i + 1]

            decoded_correct_output_list = Vocabulary.revert_back(tokenizer=tokenizer, sequence=correct_output.tolist()[0])
            input_seq_dec = Vocabulary.revert_back(tokenizer=tokenizer, sequence=input_seq.tolist()[0])

            decoded_sentence_k_100 = trainer.predict(tokenizer=tokenizer, input_seq=input_seq, k=100, return_top_n=1)
            decoded_sentence = trainer.predict(tokenizer=tokenizer, input_seq=input_seq, k=1, return_top_n=1)



            decoded_sentence_k_100 = self.filter_results(decoded_sentence_k_100[0])
            decoded_correct_output_list = self.filter_results(decoded_correct_output_list)


            current_complete_true, current_true_positive, \
            current_false_positive, current_false_negative = \
                self.get_subtoken_stats(decoded_correct_output_list, decoded_sentence_k_100)

            complete_true += current_complete_true
            true_positive += current_true_positive
            false_positive += current_false_positive
            false_negative += current_false_negative

            if ((complete_true == 1) and (len(decoded_correct_output_list)>0)): #not just unk
                logger.info("Complete True! input: {} \n correct: {}\n prediction: {}".format(input_seq_dec, decoded_correct_output_list, decoded_sentence_k_100))

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

