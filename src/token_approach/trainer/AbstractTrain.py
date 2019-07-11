import os
import logging
from keras.callbacks import EarlyStopping
from src.token_approach.data.Datagenerator import DataGenerator

import pandas as pd
from matplotlib import pyplot as plt

class AbstractTrain(object):

    def __init__(self, model, config, report_folder):
        self.model = model
        self.config = config
        self.history = None
        self.type = None
        self.es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
        #self.mc = ModelCheckpoint(os.path.join(report_folder, "Model_weights-improvement-epoch-{epoch:02d}-val_acc-{val_acc:.4f}.hdf5"), monitor='val_acc', mode='max', verbose=1, save_best_only=True)
        self.report_folder = report_folder


    def train(self, all_train, all_val, data_storage, window_size):
        logger = logging.getLogger(__name__)

        params = {'dim': window_size,
                  'batch_size': self.config.trainer.batch_size,
                  'shuffle': False}

        # Generators
        training_generator = DataGenerator(all_train, data_storage, 'train', self.config.data_loader.partition, **params)
        validation_generator = DataGenerator(all_val, data_storage, 'val', self.config.data_loader.partition, **params)

        self.history = self.model.fit_generator(
                    generator=training_generator,
                    validation_data=validation_generator,
                    epochs=self.config.trainer.num_epochs,
                    verbose=2,
                    shuffle=False,
                    callbacks=[self.es])
        
    
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



