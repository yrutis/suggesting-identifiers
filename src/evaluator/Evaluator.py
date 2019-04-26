import os
import matplotlib.pyplot as plt
import logging
import numpy as np
from sklearn import metrics

class Evaluator(object):
    def __init__(self, trained_model):
        self.__trained_model = trained_model

    def evaluate(self):
        if not self.__trained_model.history:
            raise Exception("You have to train the model first before evaluating")

        if not self.__trained_model.type:
            raise Exception("You need to assign a type to the model")

        predictions = self.__trained_model.model.predict(self.__trained_model.valX)  # get all predictions
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_prob = np.amax(predictions, axis=1)
        print(predicted_prob)
        print(predicted_classes)
        target_names = self.__trained_model.encoder.inverse_transform(
            np.unique(np.append(predicted_classes, self.__trained_model.valY))
        )

        report = metrics.classification_report(self.__trained_model.valY, predicted_classes, target_names=target_names)
        print(report)


    def visualize(self):
        if not self.__trained_model.history:
            raise Exception("You have to train the model first before visualizing")

        if not self.__trained_model.type:
            raise Exception("You need to assign a type to the model")

        model_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                    'models')
        acc_plot = os.path.join(model_folder, 'acc-' + self.__trained_model.type + '.png')
        loss_plot = os.path.join(model_folder, 'loss-' + self.__trained_model.type + '.png')

        acc = self.__trained_model.history.history['acc']
        val_acc = self.__trained_model.history.history['val_acc']
        loss = self.__trained_model.history.history['loss']
        val_loss = self.__trained_model.history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig(acc_plot)
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(loss_plot)
