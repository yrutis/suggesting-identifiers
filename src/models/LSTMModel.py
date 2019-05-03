from src.models.AbstractModel import AbstractModel

import logging

from keras import Input
from keras import layers
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Model
from keras.optimizers import Adam


class LSTMModel(AbstractModel):
    def __init__(self, context_vocab_size, windows_size, length_Y, config):
        super(LSTMModel, self).__init__(config) #call parent constractor
        self.__context_vocab_size = context_vocab_size
        self.__windows_size = windows_size
        self.__length_Y = length_Y
        self.type = "LSTM"
        self.build_model()

    def build_model(self):
        logger = logging.getLogger(__name__)

        logger.info("building model...")

        contextEmbedding = Embedding(output_dim=self.config.model.embedding_dim, input_dim=self.__context_vocab_size, input_length=self.__windows_size)

        tensor = Input(shape=(self.__windows_size,))
        c = contextEmbedding(tensor)
        c = LSTM(self.config.model.embedding_dim)(c)
        c = Dense(self.__context_vocab_size)(c)
        answer = layers.Dense(self.__length_Y, activation='softmax')(c)

        self.model = Model(tensor, answer)
        optimizer = Adam(lr=self.config.model.learning_rate)
        self.model.compile(optimizer=optimizer, loss=self.config.model.loss, metrics=self.config.model.metrics)
        print(self.model.summary())

        super().save_model_architecture() #save model architecture to disk

