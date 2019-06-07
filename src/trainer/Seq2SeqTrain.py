import os
import logging
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

from src.models.Seq2SeqModel import Seq2SeqModel
import numpy as np


from src.trainer.Callbacks.Callback import Histories
from matplotlib import pyplot as plt


class Seq2SeqTrain(object):

    def __init__(self, model, encoder_model, decoder_model, data, tokenizer, config, report_folder):
        self.model = model
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.config = config
        self.history = None
        self.type = None
        self.tokenizer = tokenizer
        self.trainX = data[0]
        self.trainY = data[1]
        self.valX = data[2]
        self.valY = data[3]
        self.report_folder = report_folder
        self.es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
        self.mc = ModelCheckpoint(os.path.join(report_folder, "best_model.h5"), monitor='val_acc', mode='max',
                                  verbose=1, save_best_only=True)

    def train(self):
        logger = logging.getLogger(__name__)
        self.history = self.model.fit(self.trainX, self.trainY,
                            batch_size=64,
                            epochs=2,
                            verbose=0,
                            validation_data=[self.valX, self.valY],
                            callbacks=[self.es, self.mc])

        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig(os.path.join(self.report_folder, "acc_plot.png"))
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(os.path.join(self.report_folder, "loss_plot.png"))

        # summarize model
        plot_model(self.encoder_model, to_file=os.path.join(self.report_folder, 'encoder_model.png'), show_shapes=True)
        plot_model(self.decoder_model, to_file=os.path.join(self.report_folder, 'decoder_model.png'), show_shapes=True)
        plot_model(self.model, to_file=os.path.join(self.report_folder, 'model.png'), show_shapes=True)


    def predict(self, input_seq):
        # idx2word
        # Creating a reverse dictionary
        reverse_word_map = dict(map(reversed, self.tokenizer.word_index.items()))

        # Function takes a tokenized sentence and returns the words
        def sequence_to_text(list_of_indices):
            # Looking up words in dictionary
            words = [reverse_word_map.get(letter) for letter in list_of_indices]
            return (words)

        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        # Populate the first character of target sequence with the start character.
        start_token_idx = self.tokenizer.texts_to_sequences(['starttoken'])
        start_token_idx_elem = start_token_idx[0][0]
        target_seq[0, 0] = start_token_idx_elem

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = sequence_to_text([sampled_token_index])
            sampled_char = str(sampled_char[0])  # in case of true which is oov

            decoded_sentence += ' ' + sampled_char

            # Exit condition: either hit max length
            # or find stop token
            # TODO figure out why stop condition
            if (sampled_char == 'endtoken' or
                    len(decoded_sentence) > self.config.data_loader.window_size_body):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update states
            states_value = [h, c]

        return decoded_sentence




