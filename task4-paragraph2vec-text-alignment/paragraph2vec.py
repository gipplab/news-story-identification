import os
import string
import xml.dom.minidom
import codecs
from datetime import datetime

#   Spacy
from spacy.lang.en import English

#   NLTK
import nltk
from nltk.corpus import wordnet as wn

import re

#   Gensim
from gensim.utils import tokenize as tokenizer

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

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

    doc.writexml(open(outdir + susp.split('.')[0] + '-'
                      + src.split('.')[0] + '.xml', 'w'),
                 encoding='utf-8')

# Plagiarism pipeline
# ===================

""" The following class implement a very basic baseline comparison, which
aims at near duplicate plagiarism. It is only intended to show a simple
pipeline your plagiarism detector can follow.
Replace the single steps with your implementation to get started.
"""
class Paragraph2Vec:
    def __init__(self, susp, src, outdir, threshold, vector_size, alpha, debug=False):
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
        self.vector_size = vector_size
        self.alpha = alpha 
        self.debug = debug
        
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
        self.susp_tokens = []
        for s in self.susp_sentences:
            self.susp_tokens.append(self.tokenize(s))
        susp_fp.close()
        self.susp_taggedDoc = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(self.susp_tokens)]        

        # Preprocess sources docs
        src_fp = open(self.src, 'r', encoding='utf-8')
        self.src_text = src_fp.read()   
        self.src_sentences = self.split_sentences(self.src_text)
        self.src_tokens = []
        for s in self.src_sentences:
            self.src_tokens.append(self.tokenize(s))    
        src_fp.close()
        self.src_taggedDoc = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(self.src_tokens)]

        self.mark_sentences()

        # print('=============== SOURCE ===============')
        # for s in self.src_sentences:
        #     print(s)
        #     print()
        # print('===============')

        # print('=============== SUSP ===============')
        # for s in self.susp_sentences:
        #     print(s)
        #     print()
        # print('===============')

    def detect(self):  
        """ 
            Test a suspicious document for near-duplicate plagiarism with regards to
            a source document and return a feature list.
        """     
        max_epochs = 10
        # alpha = 0.25
        # vec_size = 30
        # threshold = 0.75

        # dm=1 means ‘distributed memory’ (PV-DM) and dm =0 means ‘distributed bag of words’ (PV-DBOW).
        model = Doc2Vec(vector_size=self.vector_size,
                        alpha=self.alpha, 
                        min_alpha=0.00025,
                        min_count=1,
                        dm =1)
        splitAt = len(self.src_tokens)
        """
            Since src sentences and susp sentences are merged together and stored in `full_tokens`,
            full_tokens contains sentences in tokens format for src_text at position from [0, splitAt-1]
            and for susp_text from [splitAt, len(full_tokens)-1]
            so if a sentence's index is < splitAt-1 then it belongs to the src document, otherwise it belongs to the suspicious doc
        """
        full_tokens = self.src_tokens + self.susp_tokens
        tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(full_tokens)]
        model.build_vocab(tagged_data)
        for epoch in range(max_epochs):
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=max_epochs)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha

        sentences_pairs = []

        for i in range(splitAt):
            similar_doc = model.docvecs.most_similar(str(i))
            # print('=============================')
            # print(f'Source sentence #{i}:\n{self.src_sentences[i]}')
            """ Take all potential suspicious sentences which scores higher than `threshold` (default to 0.99) """
            similarTag = -1
            bestSim = (0, 0)
            for sim in similar_doc:
                if int(sim[0]) >= splitAt:
                    if sim[1] >= self.threshold:
                        similarTag = int(sim[0])                        
                        bestSim = sim
                        break
            
            """ SimilarTag > -1 means we accept a sentence as suspicious based on similarity score """
            if similarTag != -1:
                if self.debug:
                    print('=============================')
                    print(f'::: Source sentence #{i}:\n{self.src_sentences[i]}\n')
                    print(f'::: Potential suspicious sentence #{similarTag-splitAt} :::')
                    print(f'{self.susp_sentences[similarTag-splitAt]}')
                    print(f'Similarity score: {bestSim[1]}')
                    print('=============================')
                    print('\n')
                sentences_pairs.append(
                    {
                        "src": (i, self.src_sentences[i]),
                        "susp": (similarTag-splitAt, self.susp_sentences[similarTag-splitAt])
                    }
                )
        
        # Build paragraphs by merging 
        # continous sentences into a bigger paragraph
        paragraphs = []
        if len(sentences_pairs) > 0:
            textSrc = sentences_pairs[0]["src"][1]
            textSusp = sentences_pairs[0]["susp"][1]
            for i in range(len(sentences_pairs)):
                if i == len(sentences_pairs) - 1:
                    paragraphs.append((textSrc, textSusp))
                else:
                    if sentences_pairs[i+1]["src"][0] == sentences_pairs[i]["src"][0] + 1:
                        textSrc += sentences_pairs[i+1]["src"][1]
                        textSusp += sentences_pairs[i+1]["susp"][1]
                    else:
                        paragraphs.append((textSrc, textSusp))
                        textSrc = sentences_pairs[i+1]["src"][1]
                        textSusp = sentences_pairs[i+1]["susp"][1]

        # Build detections list followed by the baseline format
        # Where each item is a 
        detections = []
        for p in paragraphs:
            # print('=============================')
            # print(f"Source: {p[0]} \n\nSuspicious: {p[1]}")
            # print('=============================')            
            srcIdx = self.src_text.find(p[0] if len(p[0]) <= 20 else p[0][:20])
            suspIdx = self.susp_text.find(p[1] if len(p[1]) <= 20 else p[1][:20])
            detectPairs = ((srcIdx, srcIdx + len(p[0])), (suspIdx, suspIdx + len(p[1])))
            # print(detectPairs, "\n\n")            
            detections.append(detectPairs)
        print(len(detections), "detections ", end='')
        return detections

    def postprocess(self):
        """ Postprocess the results. """
        # TODO: Implement your postprocessing steps here.
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

    ##########################################################
    #                                                        #
    #  Source code from https://stackoverflow.com/a/4576110  #
    #                                                        #
    ##########################################################
    def split_sentences(self, input_text=""):
        sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        # print("????", len(input_text), type(input_text), type(input_text.decode("windows-1252")), len(input_text.decode("windows-1252")))
        sentences = sentence_tokenizer.tokenize(input_text)        
        return sentences

    """
        Mark the start position of the sentence in the document and its length.
        Results are 2 list, `susp_sentence_marks` and `src_sentence_marks` which denote suspicious and source documents, respectively
        Element i-th is a dictionary of 2 keys: 
            'offset' - denotes the start position
            'length' - the length of the sentence
    """
    def mark_sentences(self):
        self.src_sentences_marks = []
        for sentence in self.src_sentences:
            length = len(sentence)
            offset = self.src_text.find(sentence)
            self.src_sentences_marks.append({'offset': offset, 'length': length})

        self.susp_sentences_marks = []
        for sentence in self.susp_sentences:
            length = len(sentence)
            offset = self.susp_text.find(sentence)
            self.susp_sentences_marks.append({'offset': offset, 'length': length})

    def tokenize(self, text):
        tokens = tokenizer(text)

        #   Keep only token whose length > 4
        tokens = [token for token in tokens if len(token) > 4]

        #   Remove stop words
        en_stop = set(nltk.corpus.stopwords.words('english'))        
        tokens = [token for token in tokens if token not in en_stop]

        #   Lemmatization
        tokens = [self.lemmatize_1(token) for token in tokens]
        return tokens

    def lemmatize_1(self, word):
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma