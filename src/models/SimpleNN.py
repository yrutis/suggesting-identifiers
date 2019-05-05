from src.models.AbstractModel import AbstractModel

from keras import Input
from keras import layers
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Add
from keras.models import Model
from keras.optimizers import Adam

class SimpleNN(AbstractModel):
    def __init__(self, context_vocab_size, windows_size, length_Y, config, report_folder):
        super(SimpleNN, self).__init__(config, report_folder) #call parent constractor
        self.__context_vocab_size = context_vocab_size
        self.__windows_size = windows_size
        self.__length_Y = length_Y
        self.type = "SimpleNN"
        self.build_model()

    def build_model(self):

        print("building model...")

        tensor_list = []
        embedded_list = []
        contextEmbedding = Embedding(output_dim=self.config.model.embedding_dim, input_dim=self.__context_vocab_size, input_length=1)

        i = 0
        while i < self.__windows_size:

            current_tensor = Input(shape=(1,))
            context = contextEmbedding(current_tensor)
            context = Flatten()(context)
            context = Dense(self.__context_vocab_size)(context)

            tensor_list.append(current_tensor) #add tensor to tensor list
            embedded_list.append(context) #add partial model to model list
            i += 1

        added = Add()(embedded_list)

        answer = layers.Dense(self.__length_Y, activation='softmax')(added)

        self.model = Model(tensor_list, answer)
        optimizer = Adam(lr=self.config.model.learning_rate)
        self.model.compile(optimizer=optimizer, loss=self.config.model.loss, metrics=self.config.model.metrics)
        print(self.model.summary())

        super().save_model_architecture() #save model architecture to disk


