import pandas as pd
import src.data.utils.helper_functions as helper_functions
import logging
import os


def main(filename):
    # get logger
    logger = logging.getLogger(__name__)



    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    intermediate_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'processed'), 'intermediate'),
                                               filename + '.json')


    filename = filename + '-without-abstract-methods'
    processed_decoded_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'processed'), 'decoded'),
                                               filename + '.json')  # get decoded path

    df = pd.read_json(intermediate_full_path, orient='records')

    # some basic operations: preprocessing parameters
    df['parameters'] = df['parameters'].apply(helper_functions.turn_all_to_lower)
    df['parameters'] = df['parameters'].apply(helper_functions.split_params)

    # some basic operations: preprocessing method body
    df['methodBody'] = df['methodBody'].apply(helper_functions.removeOptional)
    df['methodBody'] = df['methodBody'].apply(helper_functions.replace_string_values)
    df["methodBody"] = df['methodBody'].apply(helper_functions.turn_strings_to_list)
    df["methodBody"] = df['methodBody'].apply(helper_functions.delete_certain_strings)
    df["methodBody"] = df['methodBody'].apply(helper_functions.turn_all_to_lower)

    #clean from function structure
    df['methodBodyCleaned'] = df['methodBody'].apply(helper_functions.clean_from_function_structure)

    #concat type, params, method body
    df["concatMethodBodyCleaned"] = df['Type'].map(lambda x: [x]) + df["parameters"] + df["methodBodyCleaned"]

    # create copy of df
    df_mod = df.copy()
    # turn list in rows to strings
    df_mod['methodBody'] = df_mod['methodBody'].apply(lambda x: " ".join(x))

    # remove some rows
    df_mod = df_mod[df_mod['methodBody'] != 'empt']
    df_mod = df_mod[df_mod['methodBody'] != '{ }']

    # turn back to list
    df_mod['methodBody'] = df_mod['methodBody'].apply(lambda x: x.split())


    # split list in rows to strings
    df_mod['methodBodyCleaned'] = df_mod['methodBodyCleaned'].apply(lambda x: " ".join(x))

    df_mod = df_mod[df_mod['methodBodyCleaned'] != ' ']
    df_mod = df_mod[df_mod['methodBodyCleaned'] != '']
    # back to list
    df_mod['methodBodyCleaned'] = df_mod['methodBodyCleaned'].apply(lambda x: x.split())

    export = df_mod.to_json(processed_decoded_full_path, orient='records')
    logger.info('finished')
    logger.info(processed_decoded_full_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main("Android-Universal-Image-Loader")