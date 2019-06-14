import os

import pandas as pd

def main(filename):

    # get data folder
    data_folder = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'data')

    intermediate = os.path.join(os.path.join(os.path.join(os.path.join(
        os.path.join(data_folder, 'processed'), 'intermediate'), filename),
        'training'),
        filename+'.json')


    intermediate_train = os.path.join(os.path.join(os.path.join(os.path.join(
        os.path.join(data_folder, 'processed'), 'intermediate'), filename),
        'training'),
        filename+'.json')

    intermediate_test = os.path.join(os.path.join(os.path.join(os.path.join(
        os.path.join(data_folder, 'processed'), 'intermediate'), filename),
        'training'),
        filename+'-validation.json')

    df = pd.read_json(intermediate, orient='records')

    train = df.sample(frac=0.9, random_state=200)
    test = df.drop(train.index)

    print(df.shape)
    print(train.shape)
    print(test.shape)

    train.to_json(intermediate_train, orient='records')
    test.to_json(intermediate_test, orient='records')


if __name__ == '__main__':
    filename = 'java-small-project-split-processed'
    main(filename)
