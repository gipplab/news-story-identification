#source https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21
import task1


#
# import spacy
# spacy.load('en_core_web_sm')
# from spacy.lang.en import English
# parser = English()
#
# def tokenize(text):
#     lda_tokens = []
#     tokens = parser(text)
#     for token in tokens:
#         if token.orth_.isspace():
#             continue
#         elif token.like_url:
#             lda_tokens.append('URL')
#         elif token.orth_.startswith('@'):
#             lda_tokens.append('SCREEN_NAME')
#         else:
#             lda_tokens.append(token.lower_)
#     return lda_tokens
#
# import nltk
# nltk.download('wordnet')
#
# from nltk.corpus import wordnet as wn
#
#
# def get_lemma(word):
#     lemma = wn.morphy(word)
#     if lemma is None:
#         return word
#     else:
#         return lemma
#
#
# from nltk.stem.wordnet import WordNetLemmatizer
#
#
# def get_lemma2(word):
#     return WordNetLemmatizer().lemmatize(word)
#
# for w in ['dogs', 'ran', 'discouraged']:
#     print(w, get_lemma(w), get_lemma2(w))
#
# nltk.download('stopwords')
# en_stop = set(nltk.corpus.stopwords.words('english'))
#
# def prepare_text_for_lda(text):
#     tokens = tokenize(text)
#     tokens = [token for token in tokens if len(token) > 4]
#     tokens = [token for token in tokens if token not in en_stop]
#     tokens = [get_lemma(token) for token in tokens]
#     return tokens
#
# import random
# text_data = []
# with open('task1-topic-modelling-lda/data/dataset.csv') as f:
#     for line in f:
#         tokens = prepare_text_for_lda(line)
#         if random.random() > .99:
#             print(tokens)
#             text_data.append(tokens)
#
# from gensim import corpora
# dictionary = corpora.Dictionary(text_data)
#
# corpus = [dictionary.doc2bow(text) for text in text_data]
#
# import pickle
# pickle.dump(corpus, open('corpus.pkl', 'wb'))
# dictionary.save('dictionary.gensim')
#
# import gensim
# NUM_TOPICS = 5
# ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
# ldamodel.save('model5.gensim')
#
# topics = ldamodel.print_topics(num_words=4)
# for topic in topics:
#     print(topic)
#
# new_doc = 'Practical Bayesian Optimization of Machine Learning Algorithms'
# new_doc = prepare_text_for_lda(new_doc)
# new_doc_bow = dictionary.doc2bow(new_doc)
# print(new_doc_bow)
# print(ldamodel.get_document_topics(new_doc_bow))
#
# ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 3, id2word=dictionary, passes=15)
# ldamodel.save('model3.gensim')
# topics = ldamodel.print_topics(num_words=4)
# for topic in topics:
#     print(topic)
#
# ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 10, id2word=dictionary, passes=15)
# ldamodel.save('model10.gensim')
# topics = ldamodel.print_topics(num_words=4)
# for topic in topics:
#     print(topic)
#
# dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
# corpus = pickle.load(open('corpus.pkl', 'rb'))
# lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')
#
# import pyLDAvis
# import pyLDAvis.gensim_models as gensimvis
# lda_display = gensimvis.prepare(lda, corpus, dictionary, sort_topics=False)
# pyLDAvis.save_html(lda_display, 'LDA_Visualization.html')
# # pyLDAvis.display(lda_display)
#
# lda3 = gensim.models.ldamodel.LdaModel.load('model3.gensim')
# lda_display3 = gensimvis.prepare(lda3, corpus, dictionary, sort_topics=False)
# pyLDAvis.save_html(lda_display, 'LDA_Visualization3.html')
# # pyLDAvis.display(lda_display3)
#
#
# lda10 = gensim.models.ldamodel.LdaModel.load('model10.gensim')
# lda_display10 = gensimvis.prepare(lda10, corpus, dictionary, sort_topics=False)
# pyLDAvis.save_html(lda_display, 'LDA_Visualization10.html')
# # pyLDAvis.display(lda_display10)