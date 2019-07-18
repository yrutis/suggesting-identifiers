import os
import matplotlib.pyplot as plt
import logging
import pandas as pd
from src.subtoken_approach.trainer.AbstractTrainSubtoken import AbstractTrainSubtoken


class Evaluator(object):
    def __init__(self, trainer:AbstractTrainSubtoken, report_folder):
        self.__trained_model = trainer
        self.report_folder = report_folder
        self.fig_folder = os.path.join(report_folder, 'figures')
        self.predictions_file = os.path.join(self.report_folder, 'predictions.csv')
        self.input = []
        self.prediction_k1 = []
        self.prediction_k100 = []
        self.ground_truth = []
        self.position = []

    def save_predictions(self):
        predictions = {'input': self.input,
                              'correct': self.ground_truth,
                              'k1': self.prediction_k1,
                              'k100': self.prediction_k100,
                              'i': self.position
                              }

        predictions = pd.DataFrame(predictions, columns=['input', 'prediction', 'correct', 'k1', 'k100', 'i'])
        df = predictions.to_csv(self.predictions_file, index=False)




    def evaluate(self, testX, testY, Vocabulary, tokenizer, trainer:AbstractTrainSubtoken, is_attention):
        logger = logging.getLogger(__name__)

        complete_true_k100 = 0
        true_positive_k100 = 0
        false_positive_k100 = 0
        false_negative_k100 = 0

        complete_true_k1 = 0
        true_positive_k1 = 0
        false_positive_k1 = 0
        false_negative_k1 = 0

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

            decoded_sentence_k100 = trainer.predict(tokenizer=tokenizer, input_seq=input_seq, k=1, return_top_n=1)
            decoded_sentence_k1 = trainer.predict(tokenizer=tokenizer, input_seq=input_seq, k=1, return_top_n=1)

            #results_for_attention_plot_only = decoded_sentence_k1 #just for attention

            decoded_sentence_k100 = self.filter_results(decoded_sentence_k100[0])
            decoded_sentence_k1 = self.filter_results(decoded_sentence_k1[0])
            decoded_correct_output_list = self.filter_results(decoded_correct_output_list)


            current_complete_true_k100, current_true_positive_k100, \
            current_false_positive_k100, current_false_negative_k100 = \
                self.get_subtoken_stats(decoded_correct_output_list, decoded_sentence_k100)

            current_complete_true_k1, current_true_positive_k1, \
            current_false_positive_k1, current_false_negative_k1 = \
                self.get_subtoken_stats(decoded_correct_output_list, decoded_sentence_k1)

            complete_true_k100 += current_complete_true_k100
            true_positive_k100 += current_true_positive_k100
            false_positive_k100 += current_false_positive_k100
            false_negative_k100 += current_false_negative_k100

            complete_true_k1 += current_complete_true_k1
            true_positive_k1 += current_true_positive_k1
            false_positive_k1 += current_false_positive_k1
            false_negative_k1 += current_false_negative_k1


            #if ((current_complete_true_k1 == 1) and (len(decoded_correct_output_list)>0) and is_attention): #not just unk

             #   if is_attention:
              #      attention_plot = results_for_attention_plot_only[1]
#
 #                   attention_plot = attention_plot[:len(results_for_attention_plot_only[0]), :len(input_seq_dec)]
  #                  self.plot_attention(attention_plot, input_seq_dec, results_for_attention_plot_only[0], i)

            self.input.append(input_seq_dec)
            self.prediction_k1.append(decoded_sentence_k1)
            self.prediction_k100.append(decoded_sentence_k100)
            self.ground_truth.append(decoded_correct_output_list)
            self.position.append(i)

            i += 1

        self.save_predictions()

        accuracy, precision, recall, f1 = self.calculate_results(complete_true_k100, testX.shape[0], true_positive_k100, false_positive_k100, false_negative_k100)
        accuracy_k1, precision_k1, recall_k1, f1_k1 = self.calculate_results(complete_true_k1, testX.shape[0], true_positive_k1, false_positive_k1, false_negative_k1)

        return accuracy, precision, recall, f1, accuracy_k1, precision_k1, recall_k1, f1_k1



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

