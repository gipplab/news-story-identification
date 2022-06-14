from gensim import corpora
from data_generator import DataGenerator

from utils import DATA_FOLDER, OUTPUT_FOLDER

class Dictionary():
    def __init__(self):
        self.dictfilename = 'Task1_dictionary_full.gensim'

    def save(self, dict, filename=''):
        if len(filename) > 0:
            self.dictfilename = filename
        dict.save(self.dictfilename)
        print(f'Dictionary saved into file {self.dictfilename}')

    def build(self, data_dir=DATA_FOLDER, out_dir=OUTPUT_FOLDER, save=True, filename='Task1_dictionary_full.gensim'):
        print(f'Start building dictionary')
        self.dictfilename = filename
        data = DataGenerator(data_dir)
        dictionary = corpora.Dictionary(data)
        print(f'Dictionary built finished! {len(dictionary)} words mapped')
        if save:
            self.save(dictionary, filename=f'./output/{out_dir}/{self.dictfilename}')
        return dictionary