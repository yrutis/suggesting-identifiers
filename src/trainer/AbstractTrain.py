import os
import logging
from keras.callbacks import EarlyStopping, ModelCheckpoint
from src.trainer.Callbacks.Callback import Histories

import pandas as pd
from matplotlib import pyplot as plt


class AbstractTrain(object):

    def __init__(self, model, data, tokenizer, config, report_folder):
        self.model = model
        self.config = config
        self.history = None
        self.type = None
        self.tokenizer = tokenizer
        self.trainX = data[0]
        self.trainY = data[1]
        self.valX = data[2]
        self.valY = data[3]
        self.testX = data[4]
        self.testY = data[5]
        self.histories = Histories(report_folder, tokenizer)
        self.es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        self.mc = ModelCheckpoint(os.path.join(report_folder, "best_model.h5"), monitor='val_acc', mode='max', verbose=1, save_best_only=True)
        self.report_folder = report_folder


    def train(self):
        logger = logging.getLogger(__name__)
        self.history = self.model.fit(self.trainX, self.trainY,
                            validation_data=[self.valX, self.valY],
                            batch_size=self.config.trainer.batch_size,
                            epochs=self.config.trainer.num_epochs,
                            verbose=0,
                            callbacks=[#self.histories,
                                       self.es,
                                       self.mc])

        val_score, val_acc = self.model.evaluate(self.valX, self.valY, verbose=0)
        logger.info('Validation accuracy: {}' .format(val_acc))
        test_score, test_acc = self.model.evaluate(self.testX, self.testY, verbose=0)
        logger.info('Test accuracy: {}' .format(test_acc))

        
    
    def visualize_training(self, perc_unk_train, perc_unk_val):
        if not self.history:
            raise Exception("You have to train the model first before visualizing")


        acc_plot = os.path.join(self.report_folder, 'acc.png')
        loss_plot = os.path.join(self.report_folder, 'loss.png')
        acc_loss = os.path.join(self.report_folder, 'acc_loss.csv')

        acc = self.history.history[self.config.model.metrics[0]]
        val_acc = self.history.history['val_'+self.config.model.metrics[0]]
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(acc) + 1)


        #hack
        always_unknown_train_list = []
        for x in epochs:
            always_unknown_train_list.append(perc_unk_train)

        always_unknown_test_list = []
        for x in epochs:
            always_unknown_test_list.append(perc_unk_val)


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



