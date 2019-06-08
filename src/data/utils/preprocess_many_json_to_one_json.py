import pandas as pd
import os
import tensorflow as tf

def main():

    #load some flags
    FLAGS = tf.app.flags.FLAGS

    tf.app.flags.DEFINE_string('data', 'java-med',
                               'must be a parsed project')

    tf.app.flags.DEFINE_string('type', 'training',
                               'must be either training/ validation/ test')

    data_parsed = FLAGS.data + '-parsed'
    data_processed = FLAGS.data + '-processed'
    type = FLAGS.type

    #get data folder
    data_folder = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'data')


    #get folder to be concat
    raw_train_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'parsed'), data_parsed), type)

    #proc/intermed path
    intermediate_path = os.path.join(os.path.join(
                            os.path.join(data_folder, 'processed'), 'intermediate'), data_processed)

    #safe in this path
    intermediate_type_path = os.path.join(os.path.join(os.path.join(
                            os.path.join(data_folder, 'processed'), 'intermediate'), data_processed),
                            type)


    #create folder if it doesn't exist
    if not os.path.exists(intermediate_path):
        os.mkdir(intermediate_path)

    #create folder if it doesn't exist
    if not os.path.exists(intermediate_type_path):
        os.mkdir(intermediate_type_path)


    #get all files
    files = os.listdir(raw_train_full_path)
    complete_path_files = [os.path.join(raw_train_full_path, f) for f in files]


    #concat
    df = pd.concat([pd.read_json(f, orient='records') for f in complete_path_files], ignore_index=True, keys=files)


    #get shape
    print("amount of files {}".format(len(files)))
    print(df.head())
    print(df.shape)

    #safe
    df.to_json(os.path.join(intermediate_type_path, data_processed+".json"), orient='records')


if __name__ == '__main__':
    main()

