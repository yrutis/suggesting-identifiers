from src.trainer.AbstractTrain import AbstractTrain

class SimpleNNTrainer(AbstractTrain):
    def __init__(self, model, data, encoder):
        super(SimpleNNTrainer, self).__init__(model, data, encoder)
        self.type = "SimpleNN"


    def train(self):
        trainX = []
        i = 0
        while i < self.data[0].shape[1]:
            trainX.append(self.data[0][:, i])
            i += 1

        self.history = self.model.fit(trainX, self.data[1],
                  validation_split=0.1,
                  epochs=3,
                  batch_size=100)
        super().save()

    def predict(self, x):
        x_new = []
        i = 0
        while i < x.shape[1]:
            x_new.append(x[:, i])
            i += 1
        super().predict(x_new)

