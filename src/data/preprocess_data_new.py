import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
import keras
import matplotlib.pyplot as plt

from pickle import dump

import src.data.utils.helper_functions as helper_functions



def main():
    df = pd.read_json('Android-Universal-Image-Loader.json', orient='records')


    df['parameters'] = df['parameters'].apply(helper_functions.turn_all_to_lower)
    df['parameters'] = df['parameters'].apply(helper_functions.split_params)
    #print(df['parameters'].head())


    #some basic operations: preprocessing
    df['methodBody'] = df['methodBody'].apply(helper_functions.removeOptional)
    df["methodBody"] = df['methodBody'].apply(helper_functions.turn_strings_to_list)
    df["methodBody"] = df['methodBody'].apply(helper_functions.delete_certain_strings)
    df["methodBody"] = df['methodBody'].apply(helper_functions.turn_all_to_lower)

    #df['methodName']= df['methodName'].str.lower() should a function be all lower?



    #avg_mean = df['methodBody'].apply(compute_col_length).mean()
    df['methodBodyCleaned'] = df['methodBody'].apply(helper_functions.clean_from_function_structure)


    df["concatMethodBodyCleaned"] = df['Type'].map(lambda x: [x]) + df["parameters"] + df["methodBodyCleaned"]


    #avg_mean_cleaned = df['methodBodyCleaned'].apply(compute_col_length).mean()

    x_train, x_test, y_train, y_test = train_test_split(df['concatMethodBodyCleaned'], df['methodName'], test_size = 0.2)
    method_body_cleaned_list_x = list(x_train)
    method_name_x = list(y_train)


    training_vocab_x = helper_functions.get_training_vocab(method_body_cleaned_list_x, is_for_x=True)
    training_vocab_y = helper_functions.get_training_vocab(method_name_x, is_for_x=False)

    x_train = list(map(helper_functions.get_into_tokenizer_format, method_body_cleaned_list_x))
    print(x_train[:10])


    #fit on text the most common words from trainX and trainY
    tokenizer = Tokenizer(oov_token=True)
    tokenizer.fit_on_texts(training_vocab_x) #actual training data gets mapped on text
    tokenizer.fit_on_texts(training_vocab_y) #actual training data gets mapped on text

    word_index = tokenizer.word_index
    print('Found {} unique tokens.'.format(len(word_index)+1))

    #tokenize just trainX
    vocab_size = len(word_index) +1
    sequences = tokenizer.texts_to_sequences(x_train)
    print(sequences[:10])
    trainX = pad_sequences(sequences, maxlen=8, value=0)
    print(trainX[:10])

    #tokenize just trainY
    y_train = list(y_train)
    y_train_tokenized = tokenizer.texts_to_sequences(y_train)
    print(y_train_tokenized[0:10])
    print(y_train[0:10])
    y_train_tokenized = list(map(lambda x: x[0], y_train_tokenized))


    counter = 0
    for x in y_train_tokenized:
        if x == 1:
            counter += 1
    print("has this amount of UNK functions in Y Train {}, percentage of total {}".format(counter, counter/len(y_train_tokenized)))

    always_unknown_train = counter/len(y_train_tokenized)

    trainY = np.array(y_train_tokenized)

    # tokenize just valX
    x_test_seq = tokenizer.texts_to_sequences(x_test)
    valX = pad_sequences(x_test_seq, maxlen=8, value=0)

    # tokenize just testY
    y_test = list(y_test)
    y_test_tokenized = tokenizer.texts_to_sequences(y_test)
    print(y_test_tokenized[0:10])
    print(y_test[0:10])
    y_test_tokenized = list(map(lambda x: x[0], y_test_tokenized))
    valY = np.array(y_test_tokenized)

    counter = 0
    for x in y_test_tokenized:
        if x == 1:
            counter += 1
    print("has this amount of UNK functions in Y Val {} percentage of total {}".format(counter, counter/len(y_test_tokenized)))
    always_unknown_test = counter/len(y_test_tokenized)



    #trainY = to_categorical(trainY, num_classes=vocab_size)
    #valY = to_categorical(valY, num_classes=vocab_size)

    class Histories(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.currentPredictions = []

        def on_train_end(self, logs={}):
            return

        def on_epoch_begin(self, epoch, logs={}):
            return

        def on_epoch_end(self, epoch, logs={}):
            y_pred = self.model.predict(self.validation_data[0])
            current_dict = {}
            current_dict["X"] = self.validation_data[0]
            current_dict["Y_hat"] = np.argmax(y_pred, axis=1)
            current_dict["Y"] = self.validation_data[1]
            self.currentPredictions.append(current_dict)
            return

        def on_batch_begin(self, batch, logs={}):
            return

        def on_batch_end(self, batch, logs={}):
            return

    histories = Histories()
    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=8))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    # compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    history = model.fit(trainX, trainY,
              validation_data=[valX, valY],
              batch_size=128, epochs=50,
              callbacks=[histories])

    # save the model to file
    model.save('model.h5')
    # save the tokenizer
    dump(tokenizer, open('tokenizer.pkl', 'wb'))


    #print(histories.currentPredictions)

    score = model.evaluate(valX, valY, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])



    history_dict = history.history
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    epochs = range(1, len(acc) + 1)

    always_unknown_train_list = []
    for x in epochs:
        always_unknown_train_list.append(always_unknown_train)

    always_unknown_test_list = []
    for x in epochs:
        always_unknown_test_list.append(always_unknown_test)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')

    plt.plot(epochs, always_unknown_train_list, 'y', label='Unk acc')
    plt.plot(epochs, always_unknown_test_list, 'g', label='Unk acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.savefig("acc_plot.png")


    # Creating a reverse dictionary
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

    # Function takes a tokenized sentence and returns the words
    def sequence_to_text(list_of_indices):
        # Looking up words in dictionary
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        return (words)

    i = 0
    while i < len( histories.currentPredictions):

        #every 5 epochs
        if i % 5 == 0:

            first_x = histories.currentPredictions[i]['X'].tolist()
            first_y = histories.currentPredictions[i]['Y'].tolist()
            first_y_hat = histories.currentPredictions[i]['Y_hat'].tolist()
            first_y_hat = list(map(lambda x:[x], first_y_hat))


            # Creating texts
            first_x_reversed = list(map(sequence_to_text, first_x))
            first_y_reversed = list(map(sequence_to_text, first_y))
            first_y_hat_reversed = list(map(sequence_to_text, first_y_hat))

            df = pd.DataFrame(
                {"X": first_x_reversed,
                 "Y": first_y_reversed,
                 "Y_hat": first_y_hat_reversed})

            df.to_csv('myPred_epoch-'+str(i)+'.csv')
        i += 1



if __name__ == '__main__':
    main()