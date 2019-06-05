import pandas as pd
import os

def main():
    data_folder = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'data')


    raw_train_full_path = os.path.join(os.path.join(data_folder, 'raw'), 'jav_med_parsed_train_new')
    intermediate_full_path = os.path.join(os.path.join(data_folder, 'processed'), 'intermediate')

    files = os.listdir(raw_train_full_path)
    complete_path_files = [os.path.join(raw_train_full_path, f) for f in files]


    df = pd.concat([pd.read_json(f, orient='records') for f in complete_path_files], ignore_index=True, keys=files)


    print(df.head())
    print(df.shape)


    df.to_json(os.path.join(intermediate_full_path, "jav_med_train.json"), orient='records')


if __name__ == '__main__':
    main()

