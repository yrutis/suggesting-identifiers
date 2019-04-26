import pandas as pd
import numpy as np
import logging
import os


def main(filename, window_size):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    def getPredictand(df):
        """
        load predictands from raw data (df)
        """

        print("getting predictands...")
        df = df.loc[:, ['boundVariables', 'codeTokens']]
        y = []
        for index, row in df.iterrows():
            for idx, val in enumerate(row['boundVariables']):
                y.append(row['codeTokens'][val[0]])
        y = np.asarray(y)  # TODO check if needed
        print("predictands loaded in y")
        return y

    def getPredictor(df, windowSize=8):
        """
        load predictors from raw data (df)
        """

        print("getting predictors...")
        df = df.loc[:, ['boundVariableFeatures', 'boundVariables', 'codeTokens']]
        x = []
        for index, row in df.iterrows():
            for idx, val in enumerate(row['boundVariables']):
                currentX = []
                j = -2

                # add all context to currentX
                while j < windowSize - 1:
                    if j != 0:
                        try:
                            currentX.append(((row['codeTokens'][val[0] + j])))
                        except IndexError:
                            print('There is an index error')
                            currentX.append('IndexError')
                    j += 1

                # add currentX to x
                x.append((currentX))

        print("predictors loaded in x")
        return x

    def getFeatures(df):
        """
        load features from raw data
        """
        print("getting features...")
        features = df.loc[:, ['boundVariableFeatures', 'provenance']]
        featuresList = []  # get all features (list of all features is in all elem)
        for index, row in df.iterrows():
            for idx, val in enumerate(row['boundVariableFeatures']):
                featuresList.append(features['boundVariableFeatures'][idx][0]) #save features in right format

        return featuresList

    def rareIdentifiersToUnk(df):
        """
        transforms rare identifiers (y <= 2) to %UNK%
        """
        print("substituting rare identifiers with %UNK%...")
        cleanedDf = df.copy()
        cleanedDf.loc[cleanedDf.groupby('y').y.transform(len) <= 2, 'y'] = "%UNK%"
        print("Check if still same lenght as before substituting: {}".format(len(df) == len(cleanedDf)))
        print(
            "amount of unique y after substitution: {}, check this number with excel".format(len(cleanedDf.y.unique())))
        return cleanedDf

    def shuffleDf(df):
        print("shuffling...")
        df = df.sample(frac=1)
        return df


    logger = logging.getLogger(__name__)
    logger.info('turning raw data in something useful')

    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    full_path = os.path.join(os.path.join(os.path.join(data_folder, 'raw'), 'json'), filename + '.json') #get decoded path
    processed_decoded_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'processed'), 'decoded'), filename + '.csv') #get decoded path

    if not os.path.exists(full_path):
        print("raw data does not exist!")
        return False

    if not os.path.exists(os.path.join(data_folder, 'processed')):  # check if path exists
        print("creating processed folder...")
        os.mkdir(os.path.join(data_folder, 'processed'))

    if not os.path.exists(os.path.join(os.path.join(data_folder, 'processed'), 'decoded')):  # check if path exists
        print("creating decoded folder...")
        os.mkdir(os.path.join(os.path.join(data_folder, 'processed'), 'decoded'))

    if not os.path.exists(processed_decoded_full_path):
        print("processing data..")

        df = pd.read_json(full_path, orient='columns')  # Dataset is now stored in a Pandas Dataframe
        print('df <- {}'.format(filename))
        #print(df.head(5))

        x = getPredictor(df, windowSize=window_size)
        y = getPredictand(df)
        features = getFeatures(df)

        d = {'x': x, 'y': y, 'features': features}
        processedDf = pd.DataFrame(data=d)
        processedDf = rareIdentifiersToUnk(processedDf)
        processedDf = shuffleDf(processedDf)
        print(processedDf.head())

        print("saving decoded version of processed data...")
        processedDf.to_csv(processed_decoded_full_path)




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automatically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    filename = 'bigbluebutton_methoddeclarations_train'
    main(filename, 8)
