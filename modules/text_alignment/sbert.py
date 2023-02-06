import codecs
import pickle
import os
from os import listdir
from os.path import join, isfile, exists
import string
import xml.dom.minidom
# import codecs
from datetime import datetime
from modules.base.cache import has_cache, load_cache, save_cache
from pydantic import validate_model

#   Spacy
from spacy.lang.en import English

#   NLTK
import nltk
from nltk.corpus import wordnet as wn

import re

#   Gensim
# from gensim.utils import tokenize as tokenizer

#   SBert
from sentence_transformers import SentenceTransformer, util
from torch import threshold

from preprocessing import sentence_splitter
# Const
# =====

DELETECHARS = ''.join([string.punctuation, string.whitespace])
LENGTH = 50

def serialize_features(susp, src, features, outdir):
    """ Serialze a feature list into a xml file.
    The xml is structured as described in the readme file of the 
    PAN plagiarism corpus 2012. The filename will follow the naming scheme
    {susp}-{src}.xml and is located in the current directory.
    Existing files will be overwritten.

    Keyword arguments:
    susp     -- the filename of the suspicious document
    src      -- the filename of the source document
    features -- a list containing feature-tuples of the form
                ((start_pos_susp, end_pos_susp),
                 (start_pos_src, end_pos_src))
    """
    impl = xml.dom.minidom.getDOMImplementation()
    doc = impl.createDocument(None, 'document', None)
    root = doc.documentElement
    root.setAttribute('reference', susp)
    doc.createElement('feature')

    for f in features:
        feature = doc.createElement('feature')
        feature.setAttribute('name', 'detected-plagiarism')
        feature.setAttribute('this_offset', str(f[1][0]))
        feature.setAttribute('this_length', str(f[1][1] - f[1][0]))
        feature.setAttribute('source_reference', src)
        feature.setAttribute('source_offset', str(f[0][0]))
        feature.setAttribute('source_length', str(f[0][1] - f[0][0]))
        root.appendChild(feature)

    doc.writexml(open(join(outdir, susp.split('.')[0] + '-'
                      + src.split('.')[0] + '.xml'), 'w'),
                 encoding='utf-8')

# Plagiarism pipeline
# ===================
class SBertWrapper:
    def __init__(self, pairs, srcdir, suspdir, outdir, threshold=0.99, threshold_length=10, model_name='all-MiniLM-L7-v2', verbose=False):
        """
            Initialize SBertWrapper class

            :param str pairs: 
            :param str srcdir: 
            :param str suspdir: 
            :param outdir: 
            :param threshold:
            :param threshold_length: 
            :param model_name:
            :param verbose:
        """
        self.pairs = pairs
        self.srcdir = srcdir
        self.suspdir = suspdir
        self.outdir = outdir 
        self.threshold = threshold
        self.threshold_length = threshold_length
        self.model_name = model_name        
        self.model = SentenceTransformer(self.model_name)
        self.verbose = verbose
        self.src_text = {}
        self.src_sentences = {}
        self.susp_text = {}
        self.susp_sentences = {}

    def preprocess(self):
        cache_files = [ join(self.outdir, f) for f in [
            '_cache_src_text.pkl',
            '_cache_src_sentences.pkl',
            '_cache_src_embeddings.pkl',
            '_cache_susp_text.pkl',
            '_cache_susp_sentences.pkl',
            '_cache_susp_embeddings.pkl'
        ]]
        if has_cache(cache_files):
            objs = load_cache(cache_files)
            self.src_text = objs[0]
            self.src_sentences = objs[1]
            self.src_embeddings = objs[2]
            self.susp_text = objs[3]
            self.susp_sentences = objs[4]
            self.susp_embeddings = objs[5]
        else:
            """ Reading sentences """
            src_files = [f for f in listdir(self.srcdir) if isfile(join(self.srcdir, f))]
            susp_files = [f for f in listdir(self.suspdir) if isfile(join(self.suspdir, f))]
            print(f'[PREPROCESSING] Reading src documents...')
            for i, file in enumerate(src_files):
                src_fp = codecs.open(os.path.join(self.srcdir, file), 'r', encoding='utf-8')
                self.src_text[file] = src_fp.read()
                self.src_sentences[file] = sentence_splitter.split(self.src_text[file])
            save_cache(self.src_sentences, join(self.outdir, '_cache_src_sentences.pkl'))
            save_cache(self.src_text, join(self.outdir, '_cache_src_text.pkl'))

            print(f'[PREPROCESSING] Reading susp documents...')
            for i, file in enumerate(susp_files):
                susp_fp = codecs.open(os.path.join(self.suspdir, file), 'r', encoding='utf-8')
                self.susp_text[file] = susp_fp.read()
                # self.susp_sentences[file] = self.split_sentences(self.susp_text[file])
                self.susp_sentences[file] = sentence_splitter.split(self.susp_text[file])
            save_cache(self.susp_sentences, join(self.outdir, '_cache_susp_sentences.pkl'))
            save_cache(self.susp_text, join(self.outdir, '_cache_susp_text.pkl'))

            """ Creating embeddings """
            print(f'[PREPROCESSING] Creating susp-embeddings...')
            self.susp_embeddings = {}
            for file in susp_files:
                if self.verbose == True:
                    print(f'[PREPROCESSING] {file} converted')
                self.susp_embeddings[file] = self.model.encode(self.susp_sentences[file])
            save_cache(self.susp_embeddings, join(self.outdir, '_cache_susp_embeddings.pkl'))
            
            print(f'[PREPROCESSING] Creating src-embeddings...')
            self.src_embeddings = {}
            for file in src_files:
                if self.verbose == True:
                    print(f'[PREPROCESSING] {file} converted')
                self.src_embeddings[file] = self.model.encode(self.src_sentences[file])
            save_cache(self.src_embeddings, join(self.outdir, '_cache_src_embeddings.pkl'))

    def process(self):
        self.preprocess()
        #   Loop all pairs
        lines = open(self.pairs, 'r').readlines()
        for i, line in enumerate(lines):
            start_time = datetime.now()
            susp, src = line.split()
            if exists(join(self.outdir, f'{susp[:-4]}-{src[:-4]}.xml')):
                print(f'[PROCESSING] {"{0:0.2f}".format(i / len(lines) * 100)}% Already Computed! Skipped pair {src}-{susp}... ', end='')
            else:
                print(f'[PROCESSING] {"{0:0.2f}".format(i / len(lines) * 100)}% Processing pair {src}-{susp}... ', end='')   
                detections = self.detect(
                    src_text=self.src_text[src],
                    src_sentences=self.src_sentences[src], 
                    src_embeddings=self.src_embeddings[src], 
                    susp_text=self.susp_text[susp],
                    susp_sentences=self.susp_sentences[susp], 
                    susp_embeddings=self.susp_embeddings[susp]
                )   

                self.postprocess(src, susp, detections)

            print(f'done! {datetime.now() - start_time} elapsed')
    
    def postprocess(self, src_file, susp_file, detections):
        """ Postprocess the results. """
        serialize_features(susp_file, src_file, detections, self.outdir)

    def detect(self, src_text, src_sentences, src_embeddings, susp_text, susp_sentences, susp_embeddings):
        pairs = []
        for i, src_emb in enumerate(src_embeddings):
            for j, susp_emb in enumerate(susp_embeddings):
                cos_sim = util.cos_sim(src_emb, susp_emb)
                if cos_sim >= self.threshold:
                    pairs.append((i, j))

        detections = []
        srcIdx = 0
        suspIdx = 0
        if len(pairs) > 0:
            srcS = src_sentences[pairs[0][0]]
            srcIdx = src_text.find(srcS)
            suspS = susp_sentences[pairs[0][1]]
            suspIdx = susp_text.find(suspS)
            new_detection = [[srcIdx, srcIdx + len(srcS)], [suspIdx, suspIdx + len(suspS)]]                  
            for i in range(len(pairs)):
                # the last pair
                if i == len(pairs) - 1:  
                    if self.verbose:
                        print(f'[DETECTION-end] Pairs: {pairs[i]}')
                        print(f'[DETECTION-end] Pairs-srcS: {src_sentences[pairs[i][0]]}')
                        print(f'[DETECTION-end] Pairs-suspS: {susp_sentences[pairs[i][1]]}')
                        print(f'[DETECTION-end] {new_detection}:')      
                        print(f'[DETECTION-end] Source:\n {src_text[new_detection[0][0]:new_detection[0][1]]}')
                        print(f'[DETECTION-end] Suspicious:\n {susp_text[new_detection[1][0]:new_detection[1][1]]}')
                    if self.validate_detection(new_detection):
                        detections.append(new_detection)                    
                else:
                    if pairs[i + 1][0] == pairs[i][0] + 1:
                        srcS = src_sentences[pairs[i + 1][0]]
                        srcIdx = src_text.find(srcS)
                        suspS = susp_sentences[pairs[i + 1][1]]
                        suspIdx = susp_text.find(suspS)
                        new_detection[0][1] = srcIdx + len(srcS)
                        new_detection[1][1] = suspIdx + len(suspS)                         
                    else:
                        if self.verbose:
                            print(f'[DETECTION] {new_detection}:')      
                            print(f'[DETECTION] Source:\n {src_text[new_detection[0][0]:new_detection[0][1]]}')
                            print(f'[DETECTION] Suspicious:\n {susp_text[new_detection[1][0]:new_detection[1][1]]}')
                        if self.validate_detection(new_detection):
                            detections.append(new_detection)
                        srcS = src_sentences[pairs[i + 1][0]]
                        srcIdx = src_text.find(srcS)
                        suspS = susp_sentences[pairs[i + 1][1]]
                        suspIdx = susp_text.find(suspS)
                        new_detection = [[srcIdx, srcIdx + len(srcS)], [suspIdx, suspIdx + len(suspS)]]                            
        return detections    

    ##########################################################
    #                                                        #
    #  Source code from https://stackoverflow.com/a/4576110  #
    #                                                        #
    ##########################################################
    def split_sentences(self, input_text=""):
        sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sentence_tokenizer.tokenize(input_text)        
        return sentences

    def validate_detection(self, detection):
        valid = False
        if (detection[0][1] - detection[0][0] >= self.threshold_length) or (detection[1][1] - detection[1][0] >= self.threshold_length):
            valid = True
        if (detection[0][1] > detection[0][0]) and (detection[1][1] > detection[1][0]):
            valid = True
        return valid
""" 
    Sentence Transformer pipeline
    Source: https://www.sbert.net/docs/usage/semantic_textual_similarity.html
"""
class SBert:

    def __init__(self, susp, src, outdir, threshold, model_name='all-MiniLM-L6-v2', verbose=False):
        
        self.susp = susp
        self.src = src
        self.susp_file = os.path.split(susp)[1]
        self.src_file = os.path.split(src)[1]
        self.susp_id = os.path.splitext(susp)[0]
        self.src_id = os.path.splitext(src)[0]
        self.src_embeddings = []
        self.susp_embeddings = []
        self.output = self.susp_id + '-' + self.src_id + '.xml'
        self.detections = None
        self.outdir=outdir
        self.paragraphs = []
        self.threshold = threshold
        self.model_name = model_name
        self.verbose = verbose        
        self.threshold_length = 10

    def process(self):
        """ Process the plagiarism pipeline. """
        self.preprocess()
        self.detections = self.detect()
        self.postprocess()

    """ 
        Preprocess the suspicious and source document. 
        Source and suspicious documents are splitted into sentences and tokenized.
    """
    def preprocess(self):
        # Preprocess Suspicious docs
        susp_fp = codecs.open(self.susp, 'r', encoding='utf-8')
        self.susp_text = susp_fp.read()
        self.susp_sentences = self.split_sentences(self.susp_text)      
        
        # Preprocess sources docs
        src_fp = codecs.open(self.src, 'r', encoding='utf-8')
        self.src_text = src_fp.read()   
        self.src_sentences = self.split_sentences(self.src_text)
    
    """ 
        Test a suspicious document for near-duplicate plagiarism with regards to
        a source document and return a feature list.
    """ 
    def detect(self):  
        """ SBert Text similarity https://www.sbert.net/docs/quickstart.html """
        model = SentenceTransformer(self.model_name)

        src_embeddings = model.encode(self.src_sentences)
        susp_embeddings = model.encode(self.susp_sentences)
        pairs = []
        for i, src_emb in enumerate(src_embeddings):
            for j, susp_emb in enumerate(susp_embeddings):
                cos_sim = util.cos_sim(src_emb, susp_emb)
                if cos_sim >= self.threshold:
                    pairs.append((i, j))

        detections = []
        srcIdx = 0
        suspIdx = 0
        if len(pairs) > 0:
            srcS = self.src_sentences[pairs[0][0]]
            srcIdx = self.src_text.find(srcS)
            suspS = self.susp_sentences[pairs[0][1]]
            suspIdx = self.susp_text.find(suspS)
            new_detection = [[srcIdx, srcIdx + len(srcS)], [suspIdx, suspIdx + len(suspS)]]                  
            for i in range(len(pairs)):
                if i == len(pairs) - 1:  
                    if self.verbose:
                        print(f'[DETECTION-end] Pairs: {pairs[i]}')
                        print(f'[DETECTION-end] Pairs-srcS: {self.src_sentences[pairs[i][0]]}')
                        print(f'[DETECTION-end] Pairs-suspS: {self.susp_sentences[pairs[i][1]]}')
                        print(f'[DETECTION-end] {new_detection}:')      
                        print(f'[DETECTION-end] Source:\n {self.src_text[new_detection[0][0]:new_detection[0][1]]}')
                        print(f'[DETECTION-end] Suspicious:\n {self.susp_text[new_detection[1][0]:new_detection[1][1]]}')
                    if self.validate_detection(new_detection):
                        detections.append(new_detection)                    
                else:
                    if pairs[i + 1][0] == pairs[i][0] + 1:
                        srcS = self.src_sentences[pairs[i + 1][0]]
                        srcIdx = self.src_text.find(srcS)
                        suspS = self.susp_sentences[pairs[i + 1][1]]
                        suspIdx = self.susp_text.find(suspS)
                        new_detection[0][1] = srcIdx + len(srcS)
                        new_detection[1][1] = suspIdx + len(suspS)                         
                    else:
                        if self.verbose:
                            print(f'[DETECTION] {new_detection}:')      
                            print(f'[DETECTION] Source:\n {self.src_text[new_detection[0][0]:new_detection[0][1]]}')
                            print(f'[DETECTION] Suspicious:\n {self.susp_text[new_detection[1][0]:new_detection[1][1]]}')
                        if self.validate_detection(new_detection):
                            detections.append(new_detection)
                        srcS = self.src_sentences[pairs[i + 1][0]]
                        srcIdx = self.src_text.find(srcS)
                        suspS = self.susp_sentences[pairs[i + 1][1]]
                        suspIdx = self.susp_text.find(suspS)
                        new_detection = [[srcIdx, srcIdx + len(srcS)], [suspIdx, suspIdx + len(suspS)]]                            
        return detections       

    def postprocess(self):
        """ Postprocess the results. """
        serialize_features(self.susp_file, self.src_file, self.detections, self.outdir)

    ##########################################################
    #                                                        #
    #  Source code from https://stackoverflow.com/a/64863601 #
    #                                                        #
    ##########################################################
    def split_paragraphs(self, input_text=""):
        NEWLINES_RE = re.compile(r"\n{2,}")  # two or more "\n" characters
        no_newlines = input_text.strip("\n")  # remove leading and trailing "\n"
        split_text = NEWLINES_RE.split(no_newlines)  # regex splitting

        paragraphs = [p + "\n" for p in split_text if p.strip()]
        # p + "\n" ensures that all lines in the paragraph end with a newline
        # p.strip() == True if paragraph has other characters than whitespace

        return paragraphs

    def validate_detection(self, detection):
        return (detection[0][1] > detection[0][0]) and (detection[1][1] > detection[1][0])

    ##########################################################
    #                                                        #
    #  Source code from https://stackoverflow.com/a/4576110  #
    #                                                        #
    ##########################################################
    def split_sentences(self, input_text=""):
        sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sentence_tokenizer.tokenize(input_text)        
        return sentences

    # def tokenize(self, text):
    #     tokens = tokenizer(text)

    #     #   Keep only token whose length > 4
    #     tokens = [token for token in tokens if len(token) > 4]

    #     #   Remove stop words
    #     en_stop = set(nltk.corpus.stopwords.words('english'))        
    #     tokens = [token for token in tokens if token not in en_stop]

    #     #   Lemmatization
    #     tokens = [self.lemmatize_1(token) for token in tokens]
    #     return tokens

    # def lemmatize_1(self, word):
    #     lemma = wn.morphy(word)
    #     if lemma is None:
    #         return word
    #     else:
    #         return lemma