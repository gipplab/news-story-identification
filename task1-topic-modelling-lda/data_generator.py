import pandas as pd
from utils import preprocess, DATA_FOLDER, OUTPUT_FOLDER

class DataGenerator():
    def __init__(self, data_dir=DATA_FOLDER):
        self.files = [
            '2017_1.csv',
            '2017_2.csv',
            '2018_1.csv',
            '2018_2.csv',
            '2019_1.csv',
            '2019_2.csv'
        ]
        self.data_dir = data_dir

    def __iter__(self):
        for f in self.files:
            print(f'Reading documents from ./data/{self.data_dir}/{f}')
            for df in pd.read_csv(f'./data/{self.data_dir}/{f}', sep=',', iterator=True, chunksize=10000):
                for index, row in df.iterrows():
                    content = preprocess(row['body'])
                    if index > 0 and index % 10000 == 0:
                        print(f'{index} documents read... ')
                    yield content
            print(f'Done with ./data/{self.data_dir}/{f}')
