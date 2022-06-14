import numpy as np
import pandas as pd
import re, nltk
import spacy
import gensim

import sys

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

"""

"""
def import_data(data_dir, limit=-1):
    doc_cnt = 1
    data = []
    for item in listdir(data_dir):
        if isfile(join(data_dir, item)):
            doc_cnt = doc_cnt + 1
            if doc_cnt == limit:
                break
            with open(f'{data_dir}{item}', 'r', encoding='utf-8') as f:
                data.append(f.read())
                if doc_cnt % 10000 == 0:
                    print(f'{doc_cnt} document(s) read...')
                f.close()
    print(f'=== Total {len(data)} documents read')
    return data

def clean_data(data):
    print(f'=== Cleaning data... ', end='')
    # Convert to list
    # data = df.content.values.tolist()

    # Remove Emails
    print(f'removing emails... ', end='')
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    print(f'removing new line characters... ', end='')
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    print(f'removing single quotes... ', end='')
    data = [re.sub("\'", "", sent) for sent in data]
    print('finished !')
    return data

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # Run in terminal: python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

def preprocess(data, vectorizer):

    data = clean_data(data)

    data_words = list(sent_to_words(data))

    # Do lemmatization keeping only Noun, Adj, Verb, Adverb
    data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    data_vectorized = vectorizer.fit_transform(data_lemmatized)

    return data_vectorized

def check_sparsicity(data_vectorized):
    # Materialize the sparse data
    data_dense = data_vectorized.todense()

    # Compute Sparsicity = Percentage of Non-Zero cells
    print("=== Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")

def process(data_vectorized):
    # Define Search Param
    search_params = {'n_components': [5, 10, 15, 20], 
                    'learning_decay': [.5, .7, .9],
                    'learning_method': ['online']}

    # Init the Model
    lda = LatentDirichletAllocation()

    # Init Grid Search Class
    model = GridSearchCV(lda, param_grid=search_params, verbose=3)

    # Do the Grid Search
    model.fit(data_vectorized)

    return model

def postprocess(model, data_vectorized, vectorizer, output_dir=''):
    # Best Model
    best_lda_model = model.best_estimator_

    # Model Parameters
    print("=== Best Model's Params: ", model.best_params_)

    # Log Likelihood Score
    print("=== Best Log Likelihood Score: ", model.best_score_)

    # Perplexity
    print("=== Model Perplexity: ", best_lda_model.perplexity(data_vectorized))

    n_components = [5, 10, 15, 20]

    # Get Log Likelyhoods from Grid Search Output
    log_likelyhoods_5 = []
    log_likelyhoods_7 = []
    log_likelyhoods_9 = []
    for i, learning_decay in enumerate(model.cv_results_['param_learning_decay']):
        if learning_decay == 0.5:
            log_likelyhoods_5.append(model.cv_results_['mean_test_score'][i])
        elif learning_decay == 0.7:
            log_likelyhoods_7.append(model.cv_results_['mean_test_score'][i])
        elif learning_decay == 0.9:
            log_likelyhoods_9.append(model.cv_results_['mean_test_score'][i])
            
    # Show graph
    plt.figure(figsize=(12, 8))
    plt.plot(n_components, log_likelyhoods_5, label='0.5')
    plt.plot(n_components, log_likelyhoods_7, label='0.7')
    plt.plot(n_components, log_likelyhoods_9, label='0.9')
    plt.title("Choosing Optimal LDA Model")
    plt.xlabel("Num Topics")
    plt.ylabel("Log Likelyhood Scores")
    plt.legend(title='Learning decay', loc='best')
    plt.show()

    # Create Document - Topic Matrix
    lda_output = best_lda_model.transform(data_vectorized)

    # column names
    topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]

    # index names
    docnames = ["Doc" + str(i) for i in range(len(data))]

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic

    # Apply Style
    df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
    df_document_topics

    df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
    df_topic_distribution.columns = ['Topic Num', 'Num Documents']
    df_topic_distribution

    panel = pyLDAvis.sklearn.prepare(best_lda_model, data_vectorized, vectorizer, mds='tsne')
    if len(output_dir) > 0:
        pyLDAvis.save_html(panel, f'{output_dir}/pyLDAvis.html')

    # Topic-Keyword Matrix
    df_topic_keywords = pd.DataFrame(best_lda_model.components_)

    # Assign Column and Index
    df_topic_keywords.columns = vectorizer.get_feature_names()
    df_topic_keywords.index = topicnames

    # View
    df_topic_keywords.head()
 
    topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)        

    # Topic - Keywords Dataframe
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
    df_topic_keywords

    # Construct the k-means clusters
    from sklearn.cluster import KMeans
    clusters = KMeans(n_clusters=15, random_state=100).fit_predict(lda_output)

    # Build the Singular Value Decomposition(SVD) model
    svd_model = TruncatedSVD(n_components=2)  # 2 components
    lda_output_svd = svd_model.fit_transform(lda_output)

    # X and Y axes of the plot using SVD decomposition
    x = lda_output_svd[:, 0]
    y = lda_output_svd[:, 1]

    # Weights for the 15 columns of lda_output, for each component
    print("=== Component's weights: \n", np.round(svd_model.components_, 2))

    # Percentage of total information in 'lda_output' explained by the two components
    print("=== Perc of Variance Explained: \n", np.round(svd_model.explained_variance_ratio_, 2))

    if len(output_dir) > 0:
        # Optional 
        import joblib
        joblib.dump(best_lda_model, f'{output_dir}/model.pkl')
        with open(f'{output_dir}/results.txt', 'w', encoding='utf-8') as f:
            f.write(f"Gridsearch's parameters: {model.get_params()}\n")
            f.write(f"=== Best Model's Params: {model.best_params_}\n")
            f.write(f"=== Best Log Likelihood Score: {model.best_score_}\n")
            f.write(f"=== Model Perplexity: {best_lda_model.perplexity(data_vectorized)}\n")
            f.write(f"=== Component's weights: \n {np.round(svd_model.components_, 2)}\n")
            f.write(f"=== Perc of Variance Explained: \n {np.round(svd_model.explained_variance_ratio_, 2)}\n")
            plt.savefig(f'{output_dir}/figure.png')
            
# Styling
def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)

def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)

# Show top n keywords for each topic
def show_topics(vectorizer, lda_model=None, n_words=20):
    topic_keywords = []
    if lda_model != None:
        keywords = np.array(vectorizer.get_feature_names())
        for topic_weights in lda_model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
    else:
        print(f'=== WARNING === lda_model is "None"')
    return topic_keywords

# Main
# ====
from datetime import datetime

if __name__ == "__main__":
    """ Process the commandline arguments. We expect two arguments: The path
    pointing to the directories of raw text documents and the path pointing to
    the folder where outputs are to be written.
    """
    if len(sys.argv) == 3:
        beginning = datetime.now()

        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        if output_dir[-1] != "/":
            output_dir+="/"

        data_dir = './data/polusa_raw/'
        data = import_data(data_dir, limit=5000)
        vectorizer = CountVectorizer(
            analyzer='word',       
            min_df=10,                        # minimum reqd occurences of a word 
            stop_words='english',             # remove stop words
            lowercase=True,                   # convert all words to lowercase
            token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
            # max_features=50000,             # max number of uniq words
        )

        data_vectorized = preprocess(data, vectorizer)
        model = process(data_vectorized)
        postprocess(model, data_vectorized, vectorizer, output_dir=output_dir)

        # Optional
        # check_sparsicity(data_vectorized)

        print(f"=== DONE ! Total times for Task 4 is {datetime.now() - beginning}")
    else:
        print('\n'.join(["Unexpected number of commandline arguments.",
                         "Usage: python ./task1_automatic_gridsearch.py {input-dir} {output-dir}"]))