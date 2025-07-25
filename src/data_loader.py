import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    print('Dataset Info:')
    print(data.info())
    print('\nFirst 5 rows:')
    print(data.head())
    return data
