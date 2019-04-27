from src.trainer.AbstractTrain import AbstractTrain

class SimpleNNTrainer(AbstractTrain):
    def __init__(self, model, data, encoder, config):
        super(SimpleNNTrainer, self).__init__(model, data, encoder, config)
        self.type = "SimpleNN"
        self.trainX = self.__convert_x(data[0])
        self.valX = self.__convert_x(data[2])


    def __convert_x(self, input):
        x = []
        i = 0
        while i < input.shape[1]:
            x.append(input[:, i])
            i += 1
        return x


    def train(self):
        self.history = self.model.fit(self.trainX, self.trainY,
                  validation_split=self.config.trainer.validation_split,
                  epochs=self.config.trainer.num_epochs,
                  batch_size=self.config.trainer.batch_size)
        super().save()

    def predict(self, x):
        x = self.__convert_x(x)
        super().predict(x)

