from gensim import similarities
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from transformers import BertTokenizer
from os.path import join
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from base.cache import has_cache, save_cache, load_cache

class Embedder():
    def __init__(self, input_dir, output_dir, method='tfidf', use_cache=True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.method = method
        self.use_cache = use_cache

    def lemmatize(word, wn):
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma
    def clean(self, tokens):
        en_stop = set(stopwords.words('english'))

        tokens = [t for t in tokens if len(t) > 4]
        tokens = [w for w in tokens if w not in en_stop]
        tokens = [self.lemmatize(t, wordnet) for t in tokens]
        return tokens 
        
    def preprocess(self, docs):
        if self.use_cache and has_cache([ join(self.output_dir, '_cache_corpus.pkl') ]):
            self.corpus = load_cache([ join(self.output_dir, '_cache_corpus.pkl') ])
        else:
            tk = BertTokenizer.from_pretrained("bert-base-cased")

            
            self.corpus = [self.clean(tk.tokenize(doc['body'])) for i, doc in docs.iterrows()]
            
            if self.use_cache:
                save_cache([self.corpus], [ join(self.output_dir, '_cache_corpus.pkl') ])
        return self.corpus

    def process(self, corpus):
        # corpus = self.preprocess(self.read_docs())

        if self.method == 'tfidf':
            corpus_tfidf_dense = self.tfidf(corpus)
        elif self.method == 'doc2vec':
            pass
        elif self.method == 'sbert':
            pass

        return corpus_tfidf_dense

    def tfidf(self, corpus):
        cache_files = [
            join(self.output_dir, '_cache_dictionary.pkl'),
            join(self.output_dir, '_cache_corpus_tfidf.pkl'),
            join(self.output_dir, '_cache_distance_matrix.pkl')
        ]
        if self.use_cache and has_cache(cache_files):
            caches_objs = load_cache(cache_files)
            self.dictionary = caches_objs[0]
            self.corpus_tfidf = caches_objs[1]
            self.distance_matrix = caches_objs[2]
        else:
            self.dictionary = Dictionary(corpus)
            corpus_bow = [self.dictionary.doc2bow(doc) for doc in corpus]
            tfidf_model = TfidfModel(corpus_bow)
            self.corpus_tfidf = tfidf_model[corpus_bow]
            self.distance_matrix = similarities.MatrixSimilarity(tfidf_model[self.corpus_tfidf])

            if self.use_cache:
                save_cache([self.dictionary, self.corpus_tfidf, self.distance_matrix], cache_files)

        return self.dictionary, self.corpus_tfidf, self.distance_matrix