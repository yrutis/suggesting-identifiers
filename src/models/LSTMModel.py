from src.models.AbstractModel import AbstractModel

from keras import Input
from keras import layers
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Model
from keras.optimizers import Adam

class LSTMModel(AbstractModel):
    def __init__(self, context_vocab_size, windows_size, length_Y):
        super(LSTMModel, self).__init__() #call parent constractor
        self.__context_vocab_size = context_vocab_size
        self.__windows_size = windows_size
        self.__length_Y = length_Y
        self.type = "LSTM"
        self.build_model()

    def build_model(self):

        print("building model...")

        contextEmbedding = Embedding(output_dim=50, input_dim=self.__context_vocab_size, input_length=self.__windows_size)

        tensor = Input(shape=(self.__windows_size,))
        c = contextEmbedding(tensor)
        c = LSTM(50)(c)
        c = Dense(self.__context_vocab_size)(c)
        answer = layers.Dense(self.__length_Y, activation='softmax')(c)

        self.model = Model(tensor, answer)
        optimizer = Adam(lr=0.007)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
        print(self.model.summary())

        super().save_model_architecture() #save model architecture to disk


