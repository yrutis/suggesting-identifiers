# -*- coding: utf-8 -*-
import click
import logging
import os
import json
import zipfile
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv

import json
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
            for idx, word in enumerate(x):
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
        for index, row in df.iterrows():
            currentFeatureSetEncoded = []
            featureSet = ast.literal_eval(row['features'])  # convert to list
            for feature in featureSet:
                if feature not in tokenIndex:
                    tokenIndex[feature] = len(tokenIndex) + 1
                currentFeatureSetEncoded.append(tokenIndex[feature])

            encodedFeatures.append(currentFeatureSetEncoded)



        d = {'x': encodedX, 'y': encodedY, 'features': encodedFeatures}
        encodedProcessedDf = pd.DataFrame(data=d)
        return tokenIndex, encodedProcessedDf

    logger = logging.getLogger(__name__)
    logger.info('building encoded version...')

    full_path = "../../data/raw/json/" + filename + ".json"
    processed_decoded_full_path = '../../data/processed/decoded/' + filename + '.csv'
    processed_encoded_full_path = '../../data/processed/encoded/' + filename + '.json'

    if not os.path.exists(full_path):
        print("raw data does not exist!")
        return False

    if not os.path.exists(processed_decoded_full_path):
        print("decoded file does not exist!")
        return False

    else:
        processedDf = pd.read_csv(processed_decoded_full_path)

    #only encode if encoded file doesn't exist yet
    if not os.path.exists(processed_encoded_full_path):

        # check if encoded folder exists
        if not os.path.exists('../../data/processed/encoded'):
            print("creating encoded folder...")
            os.mkdir('../../data/processed/encoded')

        # tokenize data and make ready for model
        print("init tokenizer...")
        tokenIndex = {}
        tokenIndex, encodedProcessedDf = customTokenizer(tokenIndex, processedDf)

        #save the encoded df to json
        encodedProcessedDf.to_json('../../data/processed/encoded/' + filename + '.json')

        print('length of vocab size: {}'.format(len(tokenIndex) + 1))

        #save tokenizer
        with open('../../data/processed/encoded/'+filename+'-tokenizer.json', 'w') as fp:
            json.dump(tokenIndex, fp)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automatically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    filename = 'Bukkit_types_test'
    main(filename)
