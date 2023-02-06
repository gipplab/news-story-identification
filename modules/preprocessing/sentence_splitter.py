##########################################################
#                                                        #
#  Source code from https://stackoverflow.com/a/4576110  #
#                                                        #
##########################################################
import nltk
def split(text: str):
    if (len(text) > 0):
        sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sentence_tokenizer.tokenize(text)        
        return sentences
    return []