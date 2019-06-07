from src.models.AbstractModel import AbstractModel

import logging

from keras import Input
from keras.layers import Embedding, Flatten, Dense, Dropout, LSTM
from keras.models import Model
from keras.optimizers import Adam


class Seq2SeqModel(AbstractModel):
    def __init__(self, context_vocab_size, windows_size, config, report_folder):
        super(Seq2SeqModel, self).__init__(config, report_folder) #call parent constractor
        self.__context_vocab_size = context_vocab_size
        self.__windows_size = windows_size
        self.type = "Seq2Seq"
        self.build_model()

    def build_model(self):
        logger = logging.getLogger(__name__)

        logger.info("building model...")
        logger.info("Embedding of shape {}, {}, {}".format(self.__context_vocab_size, 64, self.__windows_size))


        e = Embedding(self.__context_vocab_size, 64)
        encoder_inputs = Input(shape=(None,), name="encoder_input")
        en_x = e(encoder_inputs)
        encoder = LSTM(50, return_state=True)
        encoder_outputs, state_h, state_c = encoder(en_x)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,), name='decoder_input')
        dex = e
        final_dex = dex(decoder_inputs)

        decoder_lstm = LSTM(50, return_sequences=True, return_state=True)

        decoder_outputs, _, _ = decoder_lstm(final_dex,
                                             initial_state=encoder_states)

        decoder_dense = Dense(self.__context_vocab_size, activation='softmax')

        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

        print(self.model.summary())

        # Encode the input sequence to get the "thought vectors"
        self.encoder_model = Model(encoder_inputs, encoder_states)

        # Decoder setup
        # Below tensors will hold the states of the previous time step
        decoder_state_input_h = Input(shape=(50,))
        decoder_state_input_c = Input(shape=(50,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        dec_emb2 = dex(decoder_inputs)  # Get the embeddings of the decoder sequence

        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
        decoder_states2 = [state_h2, state_c2]
        decoder_outputs2 = decoder_dense(
            decoder_outputs2)  # A dense softmax layer to generate prob dist. over the target vocabulary

        # Final decoder model
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs2] + decoder_states2)

        super().save_model_architecture() #save model architecture to disk