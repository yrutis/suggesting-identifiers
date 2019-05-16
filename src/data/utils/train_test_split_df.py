import os

import pandas as pd

def main():
    filename = 'androidTest'

    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'data')

    print(data_folder)
    raw_original_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'raw'), 'original'),
                                               filename + '.json')

    print(raw_original_full_path)

    raw_train_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'raw'), 'train'),
                                               filename + '.json')

    raw_test_full_path = os.path.join(os.path.join(os.path.join(data_folder, 'raw'), 'test'),
                                               filename + '.json')

    df = pd.read_json(raw_original_full_path, orient='records')

    train = df.sample(frac=0.8, random_state=200)
    test = df.drop(train.index)

    print(df.shape)
    print(train.shape)
    print(test.shape)

    train.to_json(raw_train_full_path, orient='records')
    test.to_json(raw_test_full_path, orient='records')


if __name__ == '__main__':
    main()
