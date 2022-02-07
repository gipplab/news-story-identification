#source https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21

#   Spacy
# import spacy
# from spacy.lang.en import English

#   NLTK
# import nltk
# from nltk.corpus import wordnet as wn
# from nltk.stem.wordnet import WordNetLemmatizer

#   Gensim
# from gensim import corpora

#   Numpy
# import numpy as np
#
# #   Util
# import random
#
# #   Additional packages
from utils import pipeline

model = pipeline()

# spacy.load('en_core_web_sm')
# parser = English()

# nltk.download('wordnet')

# wnLemmatizer = WordNetLemmatizer()
# print("######## TESTING LEMMATIZATION #######")
# for w in ['dogs', 'ran', 'discouraged']:
#     print(w, get_lemma(w, wn), get_lemma2(w, wnLemmatizer))
#
# nltk.download('stopwords')
# en_stop = set(nltk.corpus.stopwords.words('english'))
# print("######## READ SOME DATA FROM ./data/dataset.csv #######")
# #   Read sample data
# text_data = []
# with open('./data/dataset.csv') as f:
#     for line in f:
#         tokens = prepare_text_for_lda(line, en_stop, parser, wn)
#         if random.random() > .99:
#             print(line, tokens)
#             text_data.append(tokens)
#
# dictionary = corpora.Dictionary(text_data)
#
# print("Total lines read:")
# print(len(text_data))
#
# corpus = [dictionary.doc2bow(text) for text in text_data]
#
# print("######## TAKE A LOOK AT THE 'dictionary' #######")
# for k in dictionary:
#     print(k, dictionary[k])
#
# print("######## TAKE A LOOK AT THE 'corpus' #######")
# for anything, i in enumerate(corpus):
#     print(anything, text_data[anything], i)
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
# print(f'######## HERE ARE THE {NUM_TOPICS} TOPICS: #######')
# topics = ldamodel.print_topics(num_words=4)
# for topic in topics:
#     print(topic)
#
# new_doc = 'Practical Bayesian Optimization of Machine Learning Algorithms'
# new_doc = prepare_text_for_lda(new_doc, en_stop, parser, wn)
# new_doc_bow = dictionary.doc2bow(new_doc)
# print(f'####### NOW TEST THIS SENTENCE {new_doc_bow}')
# potential_topic = ldamodel.get_document_topics(new_doc_bow)
# highest_among_them = max(range(len(potential_topic)), key=lambda i: potential_topic[i][1])
# print(f'The topics distribution is: ', potential_topic)
# print(f'So the possible topic is: ', max(potential_topic, key=lambda x: x[1]), f' index {highest_among_them}')
# print(f'Equivalent to {topics[highest_among_them]}')

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