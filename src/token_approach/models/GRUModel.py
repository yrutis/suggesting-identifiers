from src.token_approach.models.AbstractModel import AbstractModel

import logging

from keras import Input
from keras import layers
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import Dropout
from keras.models import Model
from keras.optimizers import Adam


class GRUModel(AbstractModel):
    def __init__(self, context_vocab_size, windows_size, config, report_folder):
        super(GRUModel, self).__init__(config, report_folder) #call parent constractor
        self.__context_vocab_size = context_vocab_size
        self.__windows_size = windows_size
        self.type = "GRU"
        self.build_model()

    def build_model(self):
        logger = logging.getLogger(__name__)

        logger.info("building model...")
        logger.info("Embedding of shape {}, {}, {}".format(self.__context_vocab_size, self.config.model.embedding_dim, self.__windows_size))


        contextEmbedding = Embedding(input_dim=self.__context_vocab_size, output_dim=self.config.model.embedding_dim, input_length=self.__windows_size)


        tensor = Input(shape=(self.__windows_size,))
        c = contextEmbedding(tensor)
        c = Dropout(self.config.model.dropout_1)(c)
        c = GRU(self.config.model.gru_dim, recurrent_dropout=self.config.model.recurrent_dropout)(c)
        c = Dropout(self.config.model.dropout_2)(c)
        c = Dense(self.config.model.dense_dim, activation=self.config.model.dense_activation_1)(c)
        c = Dropout(self.config.model.dropout_3)(c)
        answer = layers.Dense(self.__context_vocab_size, activation='softmax')(c)

        self.model = Model(tensor, answer)
        optimizer = Adam()
        self.model.compile(optimizer=optimizer, loss=self.config.model.loss, metrics=self.config.model.metrics)
        print(self.model.summary())

        super().save_model_architecture() #save model architecture to disk