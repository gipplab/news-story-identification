from gensim import corpora
from data_generator import DataGenerator
import pandas as pd
from utils import process_text, DATA_FOLDER, OUTPUT_FOLDER

class TokenGenerator:

    def __init__(self, data_dir=DATA_FOLDER):
        self.dictfilename = 'Task1_corpus_full.gensim'
        self.data = DataGenerator(data_dir)
        self.data_dir = data_dir
        self.files = [
            '2017_1.csv',
            '2017_2.csv',
            '2018_1.csv',
            '2018_2.csv',
            '2019_1.csv',
            '2019_2.csv'
        ]

    def __iter__(self):
        for f in self.files:
            print(f'Building tokens from {self.data_dir}{f}')
            for df in pd.read_csv(f'{self.data_dir}{f}', sep=',', iterator=True, chunksize=10000):
                for index, row in df.iterrows():
                    content = process_text(row['body'])
                    if index > 0 and index % 10000 == 0:
                        print(f'{index} corpuses built... ')
                    yield content
            # print(f'Done with ./data/{self.data_dir}/{f}')