import os
import logging

from keras import Model, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
from keras.engine.saving import load_model

from matplotlib import pyplot as plt
import tensorflow as tf
from math import log


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
                            batch_size=self.config.trainer.batch_size,
                            epochs=self.config.trainer.num_epochs,
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

    def load_trained_model(self, path):
        logger = logging.getLogger(__name__)

        # load model from path
        self.model = load_model(os.path.join(path, 'best_model.h5'))

        logger.info(self.model.layers)
        print("first layer {}".format(self.model.layers[1]))
        print("second layer {}".format(self.model.layers[2]))

        latent_dim = self.config.model.lstm_decoder_dim
        encoder_inputs = self.model.input[0]  # input_1
        dex = self.model.layers[2]

        encoder_outputs, state_h_enc, state_c_enc = self.model.layers[3].output  # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_inputs = self.model.input[1]  # input_2
        decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
        decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        dec_emb2 = dex(decoder_inputs)  # Get the embeddings of the decoder sequence

        decoder_lstm = self.model.layers[4]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            dec_emb2, initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]

        decoder_dense = self.model.layers[5]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)


    def predict(self, input_seq):


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

        init_seq = [[list(), 1.0, False, target_seq, states_value]]
        sequences = self.run_beam_search(init_seq)
        return sequences


    def run_beam_search(self, sequences):

        #TODO add for any k

        # idx2word
        # Creating a reverse dictionary
        reverse_word_map = dict(map(reversed, self.tokenizer.word_index.items()))

        # Function takes a tokenized sentence and returns the words
        def sequence_to_text(list_of_indices):
            # Looking up words in dictionary
            words = [reverse_word_map.get(letter) for letter in list_of_indices]
            return (words)


        #checks if all sequences have come to and end
        stop_beam_search = True

        #all potential candidates
        all_candidates = list()

        #loop through the existing sequences
        for i in range(len(sequences)):
            length_seq = len(sequences)
            seq, score, stop_condition, target_seq, states_value = sequences[i]

            # if no stop condition: expand beam search tree, else just add the candidate again
            if not stop_condition:
                stop_beam_search = False

                output_tokens, h, c = self.decoder_model.predict(
                    [target_seq] + states_value)
                states_value = [h, c]

                # Sample a token
                prob_dist = output_tokens[0, -1, :]
                top_k_idx = np.argpartition(prob_dist, -4)[-4:]  # argpartition runs in O(n + k log k) time
                top_k_idx_sorted = top_k_idx[np.argsort(prob_dist[top_k_idx])]
                tok_k_probs = prob_dist[top_k_idx]
                tok_k_probs_sorted = prob_dist[top_k_idx_sorted]

                sampled_char = sequence_to_text(top_k_idx_sorted)
                sampled_char = list(map(lambda x: str(x), sampled_char))  # in case of true which is oov

                for i in range(top_k_idx_sorted.shape[0]):
                    if (sampled_char[i] == 'endtoken' or
                            len(seq) + 1 > self.config.data_loader.window_size_body):
                        stop_condition = True

                    else:
                        stop_condition = False

                    # Update the target sequence (of length 1).
                    target_seq = np.zeros((1, 1))
                    target_seq[0, 0] = top_k_idx_sorted[i]

                    candidate = [seq + [sampled_char[i]],
                                 score * -log(tok_k_probs_sorted[i]),
                                 stop_condition,
                                 target_seq,
                                 states_value]  # log probability because of very small values
                    all_candidates.append(candidate)

            else:
                all_candidates.append(sequences[i])  # add the candidate again to all candidates

        #get the top k ones
        ordered = sorted(all_candidates, key=lambda tup: tup[1])  # sort along the score
        sequences = ordered[:4]  # keep 4 best ones


        if stop_beam_search:
            sequences = np.array(sequences)
            sequences = sequences[:,0:2] #only return sequence and probability
            return sequences

        else:
            return self.run_beam_search(sequences)





