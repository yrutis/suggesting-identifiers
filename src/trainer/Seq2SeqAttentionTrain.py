import logging

import tensorflow as tf
import numpy as np
import os
import time

from src.Vocabulary.Vocabulary import Vocabulary
from src.trainer.AbstractTrainSubtoken import AbstractTrainSubtoken

tf.enable_eager_execution()


class Seq2SeqAttentionTrain(AbstractTrainSubtoken):
    def __init__(self, encoder, decoder, n_batches, val_n_batches, config, window_size, max_output_elements,
                 start_token, data_storage, report_folder):
        self.encoder = encoder
        self.decoder = decoder
        self.n_batches = n_batches
        self.val_n_batches = val_n_batches
        self.window_size = window_size
        self.config = config
        self.batch_size = config.trainer.batch_size
        self.epochs = config.trainer.num_epochs
        self.max_output_elements = max_output_elements
        self.start_token = start_token
        self.data_storage = data_storage
        self.optimizer = tf.train.AdamOptimizer()
        self.checkpoint_dir = report_folder
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              encoder=encoder,
                                              decoder=decoder)

    def loss_function(self, real, pred):
        mask = 1 - np.equal(real, 0)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
        return tf.reduce_mean(loss_)

    def train(self):
        logger = logging.getLogger(__name__)

        for epoch in range(self.epochs):
            start = time.time()

            hidden = self.encoder.initialize_hidden_state()
            total_loss = 0
            val_total_loss = 0

            for batch in range(0, self.n_batches):

                print(self.n_batches)

                trainX = np.empty([self.batch_size, self.window_size])
                trainY = np.empty([self.batch_size, self.max_output_elements])

                load_from = batch * self.batch_size
                load_until = (batch + 1) * self.batch_size

                print("loading from {} to {}".format(load_from, load_until))

                i = 0  # to always load in matrix place 0 to 64


                while load_from < load_until:

                    trainX[i,] = np.load(os.path.join(self.data_storage, 'trainX1-' + str(load_from) + '.npy'))
                    trainY[i,] = np.load(os.path.join(self.data_storage, 'trainX2-' + str(load_from) + '.npy'))
                    i += 1
                    load_from += 1

                trainX = trainX.astype(int)
                trainY = trainY.astype(int)

                trainX = tf.convert_to_tensor(trainX)
                trainY = tf.convert_to_tensor(trainY)



                loss = 0

                with tf.GradientTape() as tape:
                    enc_output, enc_hidden = self.encoder(trainX, hidden)

                    dec_hidden = enc_hidden

                    dec_input = tf.expand_dims([self.start_token] * self.batch_size, 1)

                    # Teacher forcing - feeding the target as the next input
                    for t in range(1, trainY.shape[1]):
                        # passing enc_output to the decoder
                        predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)

                        loss += self.loss_function(trainY[:, t], predictions)

                        # using teacher forcing
                        dec_input = tf.expand_dims(trainY[:, t], 1)

                batch_loss = (loss / int(trainY.shape[1]))

                total_loss += batch_loss

                variables = self.encoder.variables + self.decoder.variables #get all variables to update

                gradients = tape.gradient(loss, variables) # calc gradients

                self.optimizer.apply_gradients(zip(gradients, variables)) #apply gradients

                if batch % 5 == 0:
                    logger.info('Training Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 self.n_batches,
                                                                 batch_loss.numpy()))



            for batch in range(0, self.val_n_batches):

                valX = np.empty([self.batch_size, self.window_size])
                valY = np.empty([self.batch_size, self.max_output_elements])


                load_from = batch * self.batch_size
                load_until = (batch + 1) * self.batch_size

                i = 0  # to always load in matrix place 0 to 64
                while load_from < load_until:
                    valX[i,] = np.load(os.path.join(self.data_storage, 'valX1-' + str(load_from) + '.npy'))
                    valY[i,] = np.load(os.path.join(self.data_storage, 'valX2-' + str(load_from) + '.npy'))
                    i += 1
                    load_from += 1

                valX = valX.astype(int)
                valY = valY.astype(int)

                valX = tf.convert_to_tensor(valX)
                valY = tf.convert_to_tensor(valY)
                loss = 0

                with tf.GradientTape() as tape:
                    enc_output, enc_hidden = self.encoder(valX, hidden)

                    dec_hidden = enc_hidden

                    dec_input = tf.expand_dims([self.start_token] * self.batch_size, 1)

                    # Teacher forcing - feeding the target as the next input
                    for t in range(1, valY.shape[1]):
                        # passing enc_output to the decoder
                        predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)

                        loss += self.loss_function(valY[:, t], predictions)

                        # using teacher forcing
                        dec_input = tf.expand_dims(valY[:, t], 1)

                batch_loss = (loss / int(valY.shape[1]))

                val_total_loss += batch_loss

                if batch % 100 == 0:
                    logger.info('Validation! Epoch {} Batch {} of {} Loss {:.4f}'.format(epoch + 1,
                                                                       batch, self.val_n_batches,
                                                                       batch_loss.numpy()))
                                                                       


            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            logger.info('Epoch {} Loss {:.4f} Val Loss {:.4f}'.format(epoch + 1,
                                                total_loss / self.n_batches,
                                                val_total_loss / self.val_n_batches
                                                                          ))
            logger.info('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


    def predict(self, tokenizer, input_seq, k, return_top_n):
        logger = logging.getLogger(__name__)

        # restoring the latest checkpoint in checkpoint_dir
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

        result = []
        input_seq = input_seq.astype(int)

        input_seq = tf.convert_to_tensor(input_seq)

        hidden = [tf.zeros((1, self.config.model.gru_dim))]
        enc_out, enc_hidden = self.encoder(input_seq, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.start_token], 0)

        stop_condition = False
        while not stop_condition:

            predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, enc_out)

            # storing the attention weights to plot later on
            # attention_weights = tf.reshape(attention_weights, (-1,))
            # attention_plot[t] = attention_weights.numpy()

            predicted_id = int(tf.argmax(predictions[0]).numpy())

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

            result.append(Vocabulary.revert_back(tokenizer, predicted_id))

            if (Vocabulary.revert_back(tokenizer, predicted_id) == 'endtoken'
                    or len(result) >= self.config.data_loader.window_size_name):

                stop_condition = True


        return [result]
