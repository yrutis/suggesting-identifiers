import pandas as pd
import src.data.utils.helper_functions as helper_functions
import logging
import os
import re


def split_camel_case_and_snake_case_target(y):

    regex = "(?!^)([A-Z][a-z]+)|_"  # split by camelCase and snake_case
    splitted_target = re.sub(regex, r' \1', y).split()
    splitted_target_lower = [x.lower() for x in splitted_target]  # make all lowercase
    return splitted_target_lower

def split_camel_case_and_snake_case_body(y):

    splitted_list = []
    for target in y:
        regex = "(?!^)([A-Z][a-z]+)|_"  # split by camelCase and snake_case
        splitted_target = re.sub(regex, r' \1', target).split()
        splitted_target_lower = [x.lower() for x in splitted_target]  # make all lowercase
        splitted_list += splitted_target_lower
    splitted_list = splitted_list
    return splitted_list


def main(filename):
    # get logger
    logger = logging.getLogger(__name__)

    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    intermediate_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'processed'), 'intermediate'),
                                               filename + '.json')

    filename += '-subtoken'

    processed_decoded_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'processed'), 'decoded'),
                                               filename + '.json')  # get decoded path


    df = pd.read_json(intermediate_full_path, orient='records')

    # some basic operations: preprocessing parameters
    df['parameters'] = df['parameters'].apply(helper_functions.split_params)

    # some basic operations: preprocessing method body
    df['methodBody'] = df['methodBody'].apply(helper_functions.removeOptional)
    df["methodBody"] = df['methodBody'].apply(helper_functions.turn_strings_to_list)
    df["methodBody"] = df['methodBody'].apply(helper_functions.delete_certain_strings)

    #clean from function structure
    df['methodBodyCleaned'] = df['methodBody'].apply(helper_functions.clean_from_function_structure)


    df['methodNameSplitted'] = df['methodName'].apply(split_camel_case_and_snake_case_target)
    df['methodBodySplitted'] = df['methodBodyCleaned'].apply(split_camel_case_and_snake_case_body)
    df['parametersSplitted'] = df['parameters'].apply(split_camel_case_and_snake_case_body)

    df["methodBodySplitted"] = df['methodBodySplitted'].apply(helper_functions.turn_all_to_lower)
    df["parametersSplitted"] = df['parametersSplitted'].apply(helper_functions.turn_all_to_lower)

    df["methodBody"] = df['methodBody'].apply(helper_functions.turn_all_to_lower)
    df['parameters'] = df['parameters'].apply(helper_functions.turn_all_to_lower)

    # delete any rows where there is no method name for some reason...
    df['methodName'] = df['methodName'].apply(lambda x: " ".join(x))

    df = df[df['methodName'] != ' ']
    df = df[df['methodName'] != '']
    # back to list
    df['methodName'] = df['methodName'].apply(lambda x: x.split())


    #remove all abstract methods
    # create copy of df
    # turn list in rows to strings
    df['methodBody'] = df['methodBody'].apply(lambda x: " ".join(x))

    # remove some rows
    df = df[df['methodBody'] != 'empt']
    df = df[df['methodBody'] != '{ }']

    # turn back to list
    df['methodBody'] = df['methodBody'].apply(lambda x: x.split())

    # split list in rows to strings
    df['methodBodyCleaned'] = df['methodBodyCleaned'].apply(lambda x: " ".join(x))

    df = df[df['methodBodyCleaned'] != ' ']
    df = df[df['methodBodyCleaned'] != '']
    # back to list
    df['methodBodyCleaned'] = df['methodBodyCleaned'].apply(lambda x: x.split())

    export = df.to_json(processed_decoded_full_path, orient='records')
    logger.info('finished')
    logger.info(processed_decoded_full_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main("jav_med_train")
