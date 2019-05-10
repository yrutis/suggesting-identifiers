import os
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
from sklearn import metrics
from src.trainer.AbstractTrain import AbstractTrain


class Evaluator(object):
    def __init__(self, trained_model:AbstractTrain, report_folder):
        self.__trained_model = trained_model
        self.report_folder = report_folder

    def evaluate(self):
        # get logger
        logger = logging.getLogger(__name__)


        if not self.__trained_model.history:
            raise Exception("You have to train the model first before evaluating")



        # Creating a reverse dictionary
        reverse_word_map = dict(map(reversed, self.__trained_model.tokenizer.word_index.items()))

        # Function takes a tokenized sentence and returns the words
        def sequence_to_text(list_of_indices):
            # Looking up words in dictionary
            words = [reverse_word_map.get(letter) for letter in list_of_indices]
            return (words)


        predictions = self.__trained_model.model.predict(self.__trained_model.valX)  # get all predictions
        predicted_classes = np.argmax(predictions, axis=1)
        #predicted_classes = list(map(lambda x: [x], predicted_classes))

        #get all possible target names

        target_names = np.unique(np.append(predicted_classes, self.__trained_model.valY)).tolist()
        target_names = list(map(lambda x: [x], target_names))
        target_names = list(map(sequence_to_text, target_names))
        target_names = list(map(lambda x: x[0], target_names))


        report = metrics.classification_report(self.__trained_model.valY, predicted_classes, target_names=target_names, output_dict=True)
        df = pd.DataFrame(report).transpose()


        sklearn_report = os.path.join(self.report_folder, "report.csv")
        df.to_csv(sklearn_report)



    def visualize(self, always_unknown_train, always_unknown_test):
        if not self.__trained_model.history:
            raise Exception("You have to train the model first before visualizing")



        acc_plot = os.path.join(self.report_folder, 'acc.png')
        loss_plot = os.path.join(self.report_folder, 'loss.png')
        acc_loss = os.path.join(self.report_folder, 'acc_loss.csv')

        acc = self.__trained_model.history.history[self.__trained_model.config.model.metrics[0]]
        val_acc = self.__trained_model.history.history[self.__trained_model.config.model.metrics[0]]
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

