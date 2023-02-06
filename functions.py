import code
from os.path import join

import nltk
import pandas

from modules.preprocessing import io


def read_data(folder, files):
    li = []
    for filename in files:
        print(f'=== Reading {filename}')
        df = pandas.read_csv(join(folder, filename), index_col=None, header=0)
        li.append(df)

    df = pandas.concat(li, axis=0, ignore_index=True)

    return df

def split_sentences(input_text=""):
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sentence_tokenizer.tokenize(input_text)        
    return sentences

def read_features(folder, file):
    features = io.read_json(join(folder, file))
    return features

def get_feature(features, id_1, id_2):
    for f in features:
        if int(f['article_1_id']) == int(id_1) and int(f['article_2_id']) == int(id_2):
            return f, False
        if int(f['article_1_id']) == int(id_2) and int(f['article_2_id']) == int(id_1):
            return f, True
    return None, None

def id_synced_with_feature(features, article_1_id, article_2_id):
    for f in features:
        if int(f['article_1_id']) == int(article_1_id) and int(f['article_2_id']) == int(article_2_id):
            return True
    return False

def openShell():
    code.interact(local=locals())