from transformers import BertTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup

def clean_html(text: str):
    text = BeautifulSoup(text, features="lxml").get_text()
    text = text.replace('\n', '.')
    text = text.replace('  ', '. ')
    text = text.replace('..', '. ')
    text = text.replace(' .', '')
    return text

def lemmatizer(word, wn):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def tokenize(text: str, tokenizer=None, min_word_length=4, stopwords_set=None, wordnet=None):    
    if not tokenizer:
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    if not stopwords_set:
        stopwords_set = set(stopwords.words('english'))
    if not wordnet:
        wordnet = wn
        
    text = clean_html(text)

    tokens = tokenizer.tokenize(text)
    # Remove short words
    tokens = [token for token in tokens if len(token) > min_word_length]
    # Remove stop words
    tokens = [token for token in tokens if token not in stopwords_set]
    # Lemmatize
    tokens = [lemmatizer(token, wn) for token in tokens]
    
    return tokens