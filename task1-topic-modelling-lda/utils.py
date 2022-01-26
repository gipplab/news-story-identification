
"""
    Convert a string into a list of token
"""
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

"""
    Get the basic form of a word
    Params:
        word: word to be converted
        wn: lemmatizer
    E.g: 
        Input: learning 
        Output: learn
"""
def get_lemma(word, wn):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

"""
    Get the basic form of a word
    Params:
        word: word to be converted
        wn: lemmatizer
    E.g: 
        Input: learning 
        Output: learn
"""
def get_lemma2(word, lemmatizer):
    return lemmatizer.lemmatize(word)

"""
    Preprocessing texts
    1.  Tokenize string into tokens
    2.  Keep only token of length > 4
    3.  Remove stopwords
    4.  Lemmatizing the rest
"""
def prepare_text_for_lda(text, stop_words):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [get_lemma(token) for token in tokens]
    return tokens