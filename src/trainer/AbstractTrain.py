import os
import logging
from keras.callbacks import EarlyStopping, ModelCheckpoint


from src.trainer.Callbacks.Callback import Histories


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
                            callbacks=[self.histories,
                                       self.es,
                                       self.mc])

        score, acc = self.model.evaluate(self.valX, self.valY, verbose=0)
        logger.info('Validation accuracy: {}' .format(acc))


