import os
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
from sklearn import metrics
from src.trainer.AbstractTrain import AbstractTrain


class Evaluator(object):
    def __init__(self, trained_model, report_folder):
        self.__trained_model = trained_model
        self.report_folder = report_folder
        self.report_file = os.path.join(report_folder, 'f1_report.csv')


    def get_accuracy_precision_recall_f1_score(self, correct, predictions, type):
        # get logger
        logger = logging.getLogger(__name__)
        predictions = list(map(self.filter_results, predictions))
        correct = list(map(self.filter_results, correct))

        complete_true, true_positive, false_positive, false_negative = self.perSubtokenStatistics(zip(correct, predictions))
        logger.info("Complete True {}, TP {}, FP {}, FN {} "
                    .format(complete_true, true_positive, false_positive, false_negative)
                    )
        total = len(correct)
        accuracy, precision, recall, f1 = self.calculate_results(complete_true, total, true_positive, false_positive,
                                                            false_negative)
        logger.info("Accuracy {}, Precision {}, Recall {}, F1 {}".format(accuracy, precision, recall, f1))


        # save metrics
        metrics = {'Description': type,
                      'Accuracy': accuracy,
                      'Precision': precision,
                      'Recall': recall,
                      'F1': f1}

        df = pd.DataFrame([metrics], columns=['Description', 'Accuracy', 'Precision', 'Recall', 'F1'])

        #if report folder already exists: append, else: create new
        if os.path.exists(self.report_file):
            report_file = pd.read_csv(self.report_file)
            report_file = report_file.append(df, sort=False, ignore_index=True)
        else:
            report_file = df

        report_file.to_csv(self.report_file, index=False)


        return accuracy, precision, recall, f1




    def visualize(self, always_unknown_train, always_unknown_test):
        if not self.__trained_model.history:
            raise Exception("You have to train the model first before visualizing")



        acc_plot = os.path.join(self.report_folder, 'acc.png')
        loss_plot = os.path.join(self.report_folder, 'loss.png')
        acc_loss = os.path.join(self.report_folder, 'acc_loss.csv')

        acc = self.__trained_model.history.history[self.__trained_model.config.model.metrics[0]]
        val_acc = self.__trained_model.history.history['val_'+self.__trained_model.config.model.metrics[0]]
        loss = self.__trained_model.history.history['loss']
        val_loss = self.__trained_model.history.history['val_loss']
        epochs = range(1, len(acc) + 1)


        #hack
        always_unknown_train_list = []
        for x in epochs:
            always_unknown_train_list.append(always_unknown_train)

        always_unknown_test_list = []
        for x in epochs:
            always_unknown_test_list.append(always_unknown_test)


        # save model data
        model_data = {'acc': acc,
                      'val_acc': val_acc,
                      'unk_acc': always_unknown_train_list,
                      'unk_val_acc': always_unknown_test_list,
                      'loss': loss,
                      'val_loss': val_loss}

        df = pd.DataFrame(model_data, columns=['acc', 'val_acc', 'unk_acc', 'unk_val_acc', 'loss', 'val_loss'])
        df.to_csv(acc_loss)


        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.plot(epochs, always_unknown_train_list, 'go', label='Unknown Training acc')
        plt.plot(epochs, always_unknown_test_list, 'g', label='Unknown Test Acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig(acc_plot)
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(loss_plot)


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
        return subtoken_list

