import os
import glob
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  

DATASET_DIRECTORY = os.path.join(os.path.dirname(__file__), '../../src/data/dataset')

def load_data(dataset_directory=DATASET_DIRECTORY):
    csv_files = glob.glob(os.path.join(dataset_directory, '*.csv'))
    return csv_files

def clean_data(df):
    year_columns = [col for col in df.columns if col.strip().isdigit()]

    # convert year columns to nuemric
    df[year_columns] = df[year_columns].apply(pd.to_numeric, errors='coerce')

    # fill missing value with mean 
    df[year_columns] = df[year_columns].apply(lambda x: x.fillna(x.mean()))

    # normilize year column
    if df[year_columns].notnull().any().any():
        scaler = MinMaxScaler()
        df[year_columns] = scaler.fit_transform(df[year_columns])

    return df

def preprocess_df(dataset_directory=DATASET_DIRECTORY):
    csv_files = load_data(dataset_directory)
    cleaned_files = []
    for file_path in csv_files:
        df = pd.read_csv(file_path, skiprows=1)
        cleaned_df = clean_data(df)

        # save the cleaned data
        base_name = os.path.basename(file_path)
        cleaned_file_path = os.path.join(dataset_directory, f'cleaned_{base_name}')
        cleaned_df.to_csv(cleaned_file_path, index=False)
        cleaned_files.append(cleaned_file_path)
    return cleaned_files

if __name__ == "__main__":
    preprocess_df()