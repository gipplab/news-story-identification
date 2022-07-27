from os import listdir
from os.path import isfile, join

# Pandas
import pandas as pd

# Spacy
import spacy
from spacy.lang.en import English

# NLTK
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

# Gensim
import gensim
from gensim import corpora

# Numpy
import numpy as np

# Pickle
import pickle

# Util
import random
spacy.load('en_core_web_sm')
parser = English()

# nltk.download('wordnet')

wnLemmatizer = WordNetLemmatizer()

# nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

parser = English()

DATA_FOLDER = 'polusa_balanced'
OUTPUT_FOLDER = 'polusa_balanced'

def read_data(file_name, nrows=None):
    if isinstance(file_name, list):
        data = pd.DataFrame()
        for file in file_name:
            chunk = pd.read_csv(file, nrows=nrows)
            data = data.append(chunk, ignore_index=True)
    else:
        data = pd.read_csv(file_name, nrows=nrows)
    return data

def extract_document_body(data):
    bodys = [row['body'] for index, row in data.iterrows()]
    return bodys

def extract_dataframe_content(dataframe, column):
    content = [row[column] for index, row in dataframe.iterrows()]
    return content

def tokenize(text, parser):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def lemmatize_1(word, wn):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def lemmatize_2(word, lemmatizer):
    return lemmatizer.lemmatize(word)

"""
    Last update:
        changed function name from `preprocess` into `process_text`
"""
def process_text(sentence):
    # Tokenize
    tokens = tokenize(sentence, parser)

    # Remove short words
    tokens = [token for token in tokens if len(token) > 4]

    # Remove stop words
    tokens = [token for token in tokens if token not in en_stop]

    # Lemmatize
    tokens = [lemmatize_1(token, wn) for token in tokens]
    return tokens

def save_model(model, filename='Task1_LDA.gensim'):
    model.save(filename)
    print(f'Model is saved `{filename}`')

def save_corpus(corpus, filename='Task1_corpus.pkl'):
    pickle.dump(corpus, open(filename, 'wb'))
    print(f'Corpus saved as `{filename}`')

def save_dictionary(dictionary, filename='Task1_dictionary.gensim'):
    dictionary.save(filename)
    print(f'Dictionary saved as `{filename}`')


def train_lda(corpus, num_topics=15, filename='task1_lda.gensim', save=True, print_topics=True, print_num_topics=15,
              print_num_words=4):
    print(f'Start modelling topics with Gensim\'s LDA, corpus length = {len(corpus)}')
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    if save == True:
        ldamodel.save(filename)
        print(f'LDA built with 15 topics and saved as `{filename}`')

    if print_topics == True:
        topics = ldamodel.print_topics(num_topics=print_num_topics, num_words=print_num_words)
        print(f'Here are the top {print_num_topics} with #{print_num_words} words per topic')
        for topic in topics:
            print(topic)

    return ldamodel


def print_topics(model, print_num_topics=15, print_num_words=4):
    topics = model.print_topics(num_topics=print_num_topics, num_words=print_num_words)
    print(f'Here are the top {print_num_topics} with #{print_num_words} words per topic')
    for topic in topics:
        print(topic)

def pipeline(save=True):
    # Step 1: read data
    data = read_data(["data/polusa_balanced/2017_1.csv"], nrows=1000000)

    # Step 2 - extract main content from each article
    documents = extract_document_body(data)

    # Step 3 - preprocessing
    text_data = [process_text(body) for body in documents]

    # Step 4 - build dictionary
    dictionary = corpora.Dictionary(text_data)

    # Step 5 - build corpus
    corpus = [dictionary.doc2bow(text) for text in text_data]

    # Step 6 - run LDA
    model = train_lda(corpus)

    if save == True:
        save_corpus(corpus)
        save_dictionary(dictionary)
        save_model(model)

    return model


def list_files(folder):
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    return files


def pickle_save(object, folder, file_name):
    with open(f'{join(folder, file_name)}', 'wb') as f:
        pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def pickle_load(folder, file_name):
    with open(f'{join(folder, file_name)}', 'rb') as f:
        object = pickle.load(f)
    f.close()
    return object