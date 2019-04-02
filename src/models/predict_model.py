# -*- coding: utf-8 -*-
import ast
import logging
from keras import preprocessing
from keras.utils import to_categorical
from keras.models import model_from_json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import pickle

def main(filename):
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")

    processed_decoded_full_path = '../../data/processed/decoded/' + filename + '.csv'
    processedDf = pd.read_csv(processed_decoded_full_path)

    # loading
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    context = processedDf['x'].apply(ast.literal_eval) #saves all context x as list in list

    contextVocabSize = len(tokenizer.word_index) + 1
    print('Found %s unique tokens.' % contextVocabSize)

    tokenizer.fit_on_texts(context)
    sequences = tokenizer.texts_to_sequences(context)
    padded_sequences = preprocessing.sequence.pad_sequences(sequences, maxlen=None,
                                                                  value=0)

    contextVocabSize = len(tokenizer.word_index) + 1
    print('Found %s unique tokens.' % contextVocabSize)

    valW1 = padded_sequences[int(0.9 * padded_sequences.shape[0]): padded_sequences.shape[0], 0]
    valW2 = padded_sequences[int(0.9 * padded_sequences.shape[0]): padded_sequences.shape[0], 1]

    # load Y's
    y = processedDf['y']  # get all Y
    Y = y.values  # convert to numpy
    print("first load y: {}".format(Y[0]))

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    lenY = len(np.unique(Y))  # amount of unique Y's

    valYEnc = encoded_Y[int(0.9 * Y.shape[0]): Y.shape[0]]

    valY = to_categorical(valYEnc, num_classes=lenY)

    prediction1 = model.predict([valW1[1:2], valW2[1:2]]) #predict for 1 pair
    print("prediction1.shape {}".format(prediction1.shape)) #get numpy array with each prob
    print(prediction1) #get a prob for each label
    # sorting the predictions in descending order
    sorting = (-prediction1).argsort() #sorts by index
    print("after sorting it is {}".format(sorting))


    print("just top5 suggestion (index) {}".format((-prediction1).argsort()[0][0:4]))
    idx = (-prediction1).argsort()[0][0:4]
    idx2 = idx.tolist()
    pr2 = prediction1[0]
    probs = np.take(pr2, idx2)
    decoded = encoder.inverse_transform(idx)
    print("just top5 suggestion (prob) {} ".format(probs))
    print("decoded top5 suggestion {}".format(decoded))




    # getting the top 5 predictions
    sorted_ = sorting[0][:5]
    sorted_ = sorted_.tolist()
    print("sorted is {}".format(sorted_))
    predicted_label = encoder.inverse_transform(sorted_)
    predicted_label = predicted_label.tolist() #convert numpy to list

    for idx, value in enumerate(sorted_):
        # just some rounding steps
        prob = (prediction1[0][value]) * 100
        prob = "%.2f" % round(prob, 2)
        print("Number {} prob is {} for {}".format(idx+1, prob, predicted_label[idx]))

    predictions = model.predict([valW1, valW2])  # get all predictions
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_prob = np.amax(predictions, axis=1)
    print(predicted_prob)
    print(predicted_classes)
    target_names = encoder.inverse_transform(np.unique(np.append(predicted_classes, valYEnc)))

    report = metrics.classification_report(valYEnc, predicted_classes, target_names=target_names)
    print(report)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    filename = 'Android-Universal-Image-Loader_methoddeclarations_train'
    main(filename)