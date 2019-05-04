import pandas as pd
import logging
import os
import re
import ast
import numpy as np


def main(filename, window_size):

    logger = logging.getLogger(__name__)


    '''
    TODO get better at retrieving whole function
    def getMethodBody(code_tokens, method_index):
        i = method_index
        max_length = len(code_tokens)
        counter = 0
        while i < max_length:
            if code_tokens[i] == '{':
                counter += 1
    
            if code_tokens[i] == '}':
                counter -= 1
    
                if counter == 0:
                    end = i
                    return end
            i += 1
    
        return "NotAnIndex"
    '''

    def getFilteredList(df):
        x = []
        for index, row in df.iterrows():
            filter_elem = lambda x: re.match(r'^\w+$', x)
            filtered_whole_method_body = list(filter(filter_elem, row['x']))
            uniq = []
            [uniq.append(x) for x in filtered_whole_method_body if x not in uniq]
            x.append(uniq)
        data = {'x': x}
        df_cleaned = pd.DataFrame(data, columns=['x'])
        df_cleaned['y'] = df['y'].values
        return df_cleaned



    def getClearedList(df):
        logger.info("getting predictands...")
        y = []
        x = []

        for index, row in df.iterrows():

            # retrieve element in list
            get_elem = lambda x: x[0]
            list_of_methods = list(map(get_elem, row['boundVariables']))
            list_of_methods.sort()

            length_list_of_methods = len(list_of_methods)
            length_of_code_tokens = len(row['codeTokens'])

            for idx, val in enumerate(list_of_methods):
                begin = list_of_methods[idx] - 1  # begin is one word before
                method = list_of_methods[idx]
                if idx < (length_list_of_methods - 1):
                    end = list_of_methods[idx + 1]  # end is where next method starts
                else:
                    end = length_of_code_tokens

                x.append([row['codeTokens'][begin]] + row['codeTokens'][method + 1:end])
                y.append(row['codeTokens'][method])


        data = {'x': x, 'y': y}
        df_cleaned = pd.DataFrame(data, columns=['x', 'y'])
        return df_cleaned

    def rareIdentifiersToUnk(df):
        """
        transforms rare identifiers (y <= 2) to %UNK%
        """
        logger.info("substituting rare identifiers with %UNK%...")
        cleanedDf = df.copy()
        cleanedDf.loc[cleanedDf.groupby('y').y.transform(len) <= 2, 'y'] = "%UNK%"
        logger.info("Check if still same lenght as before substituting: {}".format(len(df) == len(cleanedDf)))
        logger.info(
            "amount of unique y after substitution: {}, check this number with excel".format(len(cleanedDf.y.unique())))
        return cleanedDf


    logger.info('turning raw data in something useful')

    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    full_path = os.path.join(os.path.join(os.path.join(data_folder, 'raw'), 'json'), filename + '.json') #get decoded path
    intermediate_decoded_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'processed'), 'intermediate'), filename + '.csv') #get decoded path
    processed_decoded_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'processed'), 'decoded'), filename +'-'+str(window_size)+ '.csv') #get decoded path

    if not os.path.exists(full_path):
        logger.info("raw data does not exist!")
        return False

    if not os.path.exists(os.path.join(data_folder, 'processed')):  # check if path exists
        logger.info("creating processed folder...")
        os.mkdir(os.path.join(data_folder, 'processed'))

    if not os.path.exists(os.path.join(os.path.join(data_folder, 'processed'), 'intermediate')):  # check if path exists
        logger.info("creating intermediate folder...")
        os.mkdir(os.path.join(os.path.join(data_folder, 'processed'), 'intermediate'))

    if not os.path.exists(os.path.join(os.path.join(data_folder, 'processed'), 'decoded')):  # check if path exists
        logger.info("creating decoded folder...")
        os.mkdir(os.path.join(os.path.join(data_folder, 'processed'), 'decoded'))

    if not os.path.exists(intermediate_decoded_full_path):
        logger.info("preprocessing data..")

        logger.info("getting method body")
        df = pd.read_json(full_path)
        df_cleaned = getClearedList(df)
        df_filtered = getFilteredList(df_cleaned)
        df_filtered = rareIdentifiersToUnk(df_filtered)
        #print(df_cleaned.head())
        #print(df_filtered.head())
        df_filtered.to_csv(intermediate_decoded_full_path)

    if not os.path.exists(processed_decoded_full_path):
        logger.info("creating decoded version of data..")

        processedDf = pd.read_csv(intermediate_decoded_full_path)
        context = processedDf['x'].apply(ast.literal_eval)  # saves all context x as list in list
        f = lambda x: x[0:min(len(x), window_size)]
        r = context.apply(f)
        df = pd.DataFrame(columns=['x', 'y'])
        df['x'] = r
        df['y'] = processedDf['y']
        df.to_csv(processed_decoded_full_path)


        '''
        splitted = np.array_split(df, 3)
        i = 0
        print(splitted)
        while i < len(splitted):
            #save partition
            splitted[i].to_csv("number-" + str(i) +".csv")
            i += 1

        '''
        #ToDo refactor!

    logger.info("done preparing data")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    filename = 'bigbluebutton_methoddeclarations_train'

    window_size = 8
    main(filename, window_size)