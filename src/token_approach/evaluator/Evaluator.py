import os
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib


class Evaluator(object):
    def __init__(self, report_folder):
        self.report_folder = report_folder
        self.correct_predictions_file = os.path.join(self.report_folder, 'correct_predictions.csv')


    def get_accuracy_precision_recall_f1_score(self, correct, predictions):
        # get logger
        logger = logging.getLogger(__name__)


        complete_true, true_positive, false_positive, false_negative = self.perSubtokenStatistics(zip(correct, predictions))
        logger.info("Complete True {}, TP {}, FP {}, FN {} "
                    .format(complete_true, true_positive, false_positive, false_negative)
                    )
        total = len(correct)
        accuracy, precision, recall, f1 = self.calculate_results(complete_true, total, true_positive, false_positive,
                                                            false_negative)
        logger.info("Accuracy {}, Precision {}, Recall {}, F1 {}".format(accuracy, precision, recall, f1))

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
    def perSubtokenStatistics(results):
        # check if in vocabulary
        complete_true = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0
        for correct, predicted in results:
            if ''.join(correct) == ''.join(predicted):
                true_positive += len(correct)
                complete_true += 1
                continue

            for subtok in predicted:
                if subtok in correct:
                    true_positive += 1
                else:
                    false_positive += 1
            for subtok in correct:
                if not subtok in predicted:
                    false_negative += 1

        return complete_true, true_positive, false_positive, false_negative


