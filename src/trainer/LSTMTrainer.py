from src.trainer.AbstractTrain import AbstractTrain

class LSTMTrainer(AbstractTrain):
    def __init__(self, model, data, encoder, config):
        super(LSTMTrainer, self).__init__(model, data, encoder, config)
        self.type = "LSTM"


    def train(self):

        self.history = self.model.fit(self.trainX, self.trainY,
                  validation_split=self.config.trainer.validation_split,
                  epochs=self.config.trainer.num_epochs,
                  batch_size=self.config.trainer.batch_size)
        super().save()

