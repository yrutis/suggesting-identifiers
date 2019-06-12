import pandas as pd
import src.data.utils.helper_functions as helper_functions
import logging
import os
import tensorflow as tf


def main():

    def delete_abstract_methods(df):
        all_methods_cnt = df.shape[0]
        # remove all abstract methods
        # turn list in rows to strings
        df['methodBody'] = df['methodBody'].apply(lambda x: " ".join(x))

        # remove some rows
        logger.info("before df[df['methodBody'] != 'empt'] {}".format(df.shape))
        df = df[df['methodBody'] != 'empt']
        logger.info("after df[df['methodBody'] != 'empt'] {}".format(df.shape))
        logger.info("before df[df['methodBody'] != 'curlybrackets'] {}".format(df.shape))
        df = df[df['methodBody'] != '{ }']
        logger.info("after df[df['methodBody'] != 'curlybrackets'] {}".format(df.shape))

        # turn back to list
        df['methodBody'] = df['methodBody'].apply(lambda x: x.split())

        # split list in rows to strings
        df['methodBodyCleaned'] = df['methodBodyCleaned'].apply(lambda x: " ".join(x))

        logger.info("before df[df['methodBodyCleaned'] != ' '] {}".format(df.shape))
        df = df[df['methodBodyCleaned'] != ' ']
        logger.info("after df[df['methodBodyCleaned'] != ' '] {}".format(df.shape))

        logger.info("before df[df['methodBodyCleaned'] != ''] {}".format(df.shape))
        df = df[df['methodBodyCleaned'] != '']
        logger.info("after df[df['methodBodyCleaned'] != ''] {}".format(df.shape))

        # back to list
        df['methodBodyCleaned'] = df['methodBodyCleaned'].apply(lambda x: x.split())
        impl_methods_cnt = df.shape[0]
        abstract_methods_cnt = all_methods_cnt - impl_methods_cnt
        logger.info("{} Total Methods, {} Abstract Methods, {} Implemented Methods"
                    .format(all_methods_cnt, abstract_methods_cnt, impl_methods_cnt))
        return df

    # load some flags
    FLAGS = tf.app.flags.FLAGS

    tf.app.flags.DEFINE_string('data', 'java-small-project-split',
                               'must be in processed / intermediate')

    tf.app.flags.DEFINE_string('type', 'training',
                               'must be either training/ validation/ test')

    data_processed_intermediate = FLAGS.data + '-processed'
    type = FLAGS.type

    # get logger
    logger = logging.getLogger(__name__)

    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    intermediate_full_path = os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(data_folder, 'processed'),
                                                'intermediate'), data_processed_intermediate), type), data_processed_intermediate + '.json')


    processed_decoded_project_path = os.path.join(os.path.join(os.path.join(data_folder, 'processed'),
                                               'decoded'), data_processed_intermediate)

    processed_decoded_type_path = os.path.join(processed_decoded_project_path, type)

    data_processed_intermediate += '-token'

    processed_decoded_full_path = os.path.join(processed_decoded_type_path, data_processed_intermediate + '.json')



    # create folder if it doesn't exist
    if not os.path.exists(processed_decoded_project_path):
        os.mkdir(processed_decoded_project_path)

    # create folder if it doesn't exist
    if not os.path.exists(processed_decoded_type_path):
        os.mkdir(processed_decoded_type_path)


    df = pd.read_json(intermediate_full_path, orient='records')

    # some basic operations: preprocessing parameters
    df['parameters'] = df['parameters'].apply(helper_functions.split_params)

    # some basic operations: preprocessing method body
    df['methodBody'] = df['methodBody'].apply(helper_functions.removeOptional)
    df["methodBody"] = df['methodBody'].apply(helper_functions.turn_strings_to_list)
    df["methodBody"] = df['methodBody'].apply(helper_functions.delete_certain_strings)

    #clean from function structure
    df['methodBodyCleaned'] = df['methodBody'].apply(helper_functions.clean_from_function_structure)

    # each entry is a list
    df["methodBodyCleaned"] = df['methodBodyCleaned'].apply(helper_functions.turn_all_to_lower)
    df['parameters'] = df['parameters'].apply(helper_functions.turn_all_to_lower)


    df['Type'] = df['Type'].apply(lambda x: x.lower())
    df['methodName'] = df['methodName'].apply(lambda x: x.lower())

    #delete abstract methods and empty methods
    df = delete_abstract_methods(df)

    #remove bad naming
    df = helper_functions.remove_bad_naming_methods(df)

    # for consistency each method now has a methodName, methodBody, parameters, type
    df['methodBody'] = df['methodBodyCleaned']
    df = df.drop(['methodBodyCleaned'], axis=1)


    export = df.to_json(processed_decoded_full_path, orient='records')
    logger.info('finished')
    logger.info(processed_decoded_full_path)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
