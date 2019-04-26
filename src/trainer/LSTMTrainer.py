from src.trainer.AbstractTrain import AbstractTrain

class LSTMTrainer(AbstractTrain):
    def __init__(self, model, data, encoder):
        super(LSTMTrainer, self).__init__(model, data, encoder)
        self.type = "LSTM"


    def train(self):

        self.history = self.model.fit(self.data[0], self.data[1],
                  validation_split=0.1,
                  epochs=3,
                  batch_size=100)
        super().save()

