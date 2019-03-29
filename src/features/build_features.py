# -*- coding: utf-8 -*-
import click
import logging
import os
import json
import zipfile
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np
import ast


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main(filename):
    """ Runs data processing scripts to turn decoded data from (../raw) into
        encoded data ready to be analyzed (saved in ../processed).
    """



    # TODO fix customTokenizer func to receive df and return df
    def customTokenizer(tokenIndex, df):
        """
        transforms words (x and y) to tokens
        """
        x = df.loc[:, 'x']
        y = df.loc[:, 'y']
        features = df.loc[:, 'features']

        encodedX = []
        print("Encoding X...")
        for index, row in df.iterrows():
            currentEncoded = []
            x = ast.literal_eval(row['x']) #convert to list
            print("first x {}" .format(x[0]))
            for idx, word in enumerate(x):
                print("current word {}" .format(word))
                # print("current word: {}" .format(word))
                if word not in tokenIndex:
                    tokenIndex[word] = len(tokenIndex) + 1
                currentEncoded.append(tokenIndex[word])

            encodedX.append(currentEncoded)
        print("this is first encoded x {} ".format(encodedX[0]))

        encodedY = []
        print("Encoding Y...")
        for index, row in df.iterrows():
            target = row['y']
            print("first target {}" .format(target))
            if target not in tokenIndex:
                tokenIndex[target] = len(tokenIndex) + 1
            encodedY.append(tokenIndex[target])

        encodedFeatures = []

        print("Encoding features...")
        #TODO fix here
        for index, row in df.iterrows():
            currentFeatureSetEncoded = []
            featureSet = ast.literal_eval(row['features'])  # convert to list
            print("length of featureSet ".format(len(featureSet)))
            for feature in featureSet:
                if feature not in tokenIndex:
                    tokenIndex[feature] = len(tokenIndex) + 1
                currentFeatureSetEncoded.append(tokenIndex[feature])

            encodedFeatures.append(currentFeatureSetEncoded)


        d = {'x': encodedX, 'y': encodedY, 'features': encodedFeatures}
        encodedProcessedDf = pd.DataFrame(data=d)
        return tokenIndex, encodedProcessedDf

    logger = logging.getLogger(__name__)
    logger.info('turning raw data in something useful')

    full_path = "../../data/raw/json/" + filename + ".json"
    processed_encoded_full_path = '../../data/processed/encoded/' + filename + '.csv'
    processed_decoded_full_path = '../../data/processed/decoded/' + filename + '.csv'

    if not os.path.exists(full_path):
        print("raw data does not exist!")
        return False


    if not os.path.exists('../../data/processed/decoded'):  # check if path exists
        print("creating decoded folder...")
        os.mkdir('../../data/processed/decoded')

    if not os.path.exists(processed_decoded_full_path):
        print("encoded file does not exist, exiting...")
        return False

    else:
        processedDf = pd.read_csv('../../data/processed/decoded/'+filename+'.csv')

    if not os.path.exists(processed_encoded_full_path):

        if not os.path.exists('../../data/processed/encoded'):  # check if path exists
            print("creating encoded folder...")
            os.mkdir('../../data/processed/encoded')


        # tokenize data and make ready for model
        print("init tokenizer...")
        tokenIndex = {}

        tokenIndex, encodedProcessedDf = customTokenizer(tokenIndex, processedDf)
        encodedProcessedDf.to_csv(processed_encoded_full_path)

        print('length of vocab size: {}'.format(len(tokenIndex) + 1))



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automatically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    filename = 'Bukkit_types_test'
    main(filename)
