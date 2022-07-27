import pickle
import json
import pyLDAvis
import pyLDAvis.gensim_models

# from tokenize import String
import numpy as np
import pandas as pd
from codecs import open
from os.path import exists
from math import isclose

from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim import corpora

from datetime import datetime

from sqlalchemy import Float

from utils import OUTPUT_FOLDER, DATA_FOLDER, process_text, extract_document_body, extract_dataframe_content
from tokens import TokenGenerator
from corpus import Corpus
from dictionary import Dictionary

# https://github.com/RaRe-Technologies/gensim/issues/3040
# direct_confirmation_measure.log_ratio_measure = custom_log_ratio_measure
def get_coherence_score(model, num_topic=0, data_dir=DATA_FOLDER, out_dir=OUTPUT_FOLDER, coherence='c_v'):
    if num_topic == 0:
        raise Exception(f'Num of topics in invalid (0). Exit')

    if coherence in ['c_v', 'c_uci', 'c_npmi']:
        texts = TokenGenerator(data_dir=data_dir)
        coherence_model = CoherenceModel(
            model=model,
            texts=texts,
            coherence=coherence)
    elif coherence in ['u_mass']:
        dictionary = Dictionary().build(data_dir=data_dir, out_dir=out_dir)
        corpus = Corpus(dictionary=dictionary, data_dir=data_dir)
        coherence_model = CoherenceModel(
            model=model,
            corpus=corpus,
            coherence=coherence)
    else:
        raise Exception(f'Unknown {coherence}')

    return coherence_model.get_coherence()

def load_result(output_dir):
    gridsearch_results = {'n_topics': [], 'learning_decay': [], 'alpha': [], 'beta': [], 'coherence': []}

    cache_results = pd.read_csv(f'{output_dir}gridsearch_result.csv')
    for i, row in cache_results.iterrows():
        gridsearch_results['coherence'].append(float(row['coherence']))
        gridsearch_results['n_topics'].append(int(row['n_topics']))
        gridsearch_results['learning_decay'].append(float(row['learning_decay']))
        gridsearch_results['alpha'].append(row['alpha'])
        gridsearch_results['beta'].append(None if not row['beta'] else row['beta'])        
    return gridsearch_results

"""
    Check if a gridsearch combination's score is already learnt
"""
def check_cache_result(gridsearch_results, params):
    num_topics = params['num_topics']
    learning_decay = params['learning_decay']
    alpha = params['alpha']
    beta = params['beta']
    for i in range(len(gridsearch_results['n_topics'])):
        g_n_topics = int(gridsearch_results['n_topics'][i])
        g_learning_decay = float(gridsearch_results['learning_decay'][i])
        g_alpha = gridsearch_results['alpha'][i]
        g_beta = gridsearch_results['beta'][i]
        try:
            g_alpha = np.float64(g_alpha)
        except ValueError:
            pass
        try:
            g_beta = np.float64(g_beta)
        except ValueError:
            pass
        if g_n_topics == num_topics and g_learning_decay == learning_decay:
            if (type(g_alpha) == type(alpha)) and (type(g_beta) == type(beta)):
                same_alpha = False
                if isinstance(g_alpha, str):
                    same_alpha = g_alpha == alpha
                else:
                    same_alpha = isclose(g_alpha, alpha)

                same_beta = False            
                if isinstance(g_beta, str):
                    same_beta = g_beta == beta
                else:
                    same_beta = isclose(g_beta, beta)
                
                if (same_alpha and same_beta):
                    return True
    return False

"""
    Load best parameters from cache result
"""
def get_best_params(gridsearch_results):
    best_score = -9 * 10**9
    best_params = None
    for i in range(len(gridsearch_results['n_topics'])):
        score = np.float64(gridsearch_results['coherence'][i])
        if score > best_score:
            best_params = {
                'num_topics': gridsearch_results['n_topics'][i], 
                'learning_decay': gridsearch_results['learning_decay'][i], 
                'alpha': gridsearch_results['alpha'][i], 
                'beta': gridsearch_results['beta'][i]
            }
            best_score = score
    return best_params, best_score

"""
    Gridsearch Gensim's LDA on 4 parameters:
        num_topics
        learning_decay
        alpha
        beta
"""
def gridsearch(input_dir, output_dir, params, coherence='c_uci', lda_components=None): 
    if not lda_components:
        raise Exception(f'[LEARNING] No lda_components provided. Exit!')
        
    (texts, corpus, dictionary, doc_ids) = lda_components
    
    #   Checking gridsearch params
    required_hyperparameters = ['num_topics', 'learning_decay', 'alpha', 'beta']
    for p in required_hyperparameters:
        if p not in params:
            raise Exception(f'Require hyperparameter `{p}`')
        else:
            if isinstance(params[p], list) == False:
                raise Exception(f'Type of hyperparameter `{p}` must be a list/array')

    best_model = None
    best_score = -9*10**7
    best_param = {}
    gridsearch_results = {'n_topics': [], 'learning_decay': [], 'alpha': [], 'beta': [], f'coherence': []}

    #   Preload calculated results
    if exists(f'{output_dir}gridsearch_result.csv'):
        gridsearch_results = load_result(output_dir=output_dir)
        best_param, best_score = get_best_params(gridsearch_results)
        print(f'[LEARNING] Preloaded best parameters: {best_param}, score = {best_score}')

    total_loop = len(params['num_topics']) * len(params['learning_decay']) * len(params['alpha']) * len(params['beta'])
    loop = 0
    #   Gridsearch starts
    for num_topics in params['num_topics']:      
        for learning_decay in params['learning_decay']:
            for alpha in params['alpha']:
                for beta in params['beta']:
                    loop = loop + 1
                    if check_cache_result(gridsearch_results, {'num_topics': num_topics, 'learning_decay': learning_decay, 'alpha': alpha, 'beta': beta}):
                        print(f'[LEARNING] {"{0:0.2f}".format((loop / total_loop) * 100)}% {loop}/{total_loop} Skipped num_topics: {num_topics}, learning_decay: {learning_decay}, alpha: {alpha}, beta: {beta}, already computed')
                        continue
                    lda_model = LdaModel(corpus=corpus,
                        id2word=dictionary,
                        num_topics=num_topics,
                        decay=learning_decay,
                        alpha=alpha,
                        eta=beta,
                        update_every=1,
                        chunksize=10000,
                        passes=0)

                    coherence_model = CoherenceModel(model=lda_model,
                        texts=texts,
                        coherence=coherence)

                    score = coherence_model.get_coherence()
                    if score > best_score:
                        best_model = lda_model
                        best_score = score
                        best_param = {'num_topics': num_topics, 'learning_decay': learning_decay, 'alpha': alpha, 'beta': beta}
                    gridsearch_results['coherence'].append(score)
                    gridsearch_results['n_topics'].append(num_topics)
                    gridsearch_results['learning_decay'].append(learning_decay)
                    gridsearch_results['alpha'].append(alpha)
                    gridsearch_results['beta'].append(beta)
                    pd.DataFrame({'n_topics': [num_topics], 'learning_decay': [learning_decay], 'alpha': [alpha], 'beta': [beta], f'coherence': [score]}).to_csv(f'{output_dir}gridsearch_result.csv', mode='a', index=False, header=not exists(f'{output_dir}gridsearch_result.csv'))
                    print(f'[LEARNING] {"{0:0.2f}".format((loop / total_loop) * 100)}% {loop}/{total_loop} Model num_topics: {num_topics}, learning_decay: {learning_decay}, alpha: {alpha}, beta: {beta}, score: {score}')
    print(f'[LEARNING] Best param {best_param} with score {best_score}')
    
    if best_model:
        best_model.save(f'{output_dir}gridsearch_result_best_model.gensim')
        with open(f'{output_dir}gridsearch_report_short.txt', 'w', 'utf-8') as f:
            f.write(f'Gridsearch run on {datetime.now()} with the following parameters:\n')
            f.write(f'{params}\n')
            f.write(f'Best model is {best_param} with score {best_score}\n')
    elif not exists(f'{output_dir}gridsearch_result_best_model.gensim'):
            raise Exception(f'[POSTPROCESSING] No model learnt or provided. Exit!')
    else:
        best_model = LdaModel.load(f'{output_dir}gridsearch_result_best_model.gensim')
    return best_model

def preprocess(input_dir, output_dir, memory_intense=True):
    # DATA PREPARATION
    if memory_intense:
        """
            GENERATOR VERSION == USE WHEN MEMORY IS SENSITIVE
        """
        dictionary =  Dictionary().build(data_dir=input_dir, out_dir=output_dir)
        corpus = Corpus(dictionary=dictionary, data_dir=input_dir)
        texts = TokenGenerator(data_dir=input_dir)
    else:        
        """
            FULL MEMORY VERSION
        """
        #   Read preprocessed cache
        if exists(f'{output_dir}_cache_tokens.pkl') and exists(f'{output_dir}_cache_corpus.pkl') and exists(f'{output_dir}_cache_dictionary.pkl'):
            print(f'[PREPROCESSING] Load tokens from cache...')
            with open(f'{output_dir}_cache_tokens.pkl', 'rb') as f:
                texts = pickle.load(f)
            print(f'[PREPROCESSING] Load corpus from cache...')
            with open(f'{output_dir}_cache_corpus.pkl', 'rb') as f:
                corpus = pickle.load(f)
            print(f'[PREPROCESSING] Load dictionary from cache...')
            with open(f'{output_dir}_cache_dictionary.pkl', 'rb') as f:
                dictionary = pickle.load(f)
            print(f'[POSTPROCESSING] Load doc_ids from cache...')
            with open(f'{output_dir}_cache_doc_ids.pkl', 'rb') as f:
                doc_ids = pickle.load(f)
        else:
            files = ['2017_1.csv','2017_2.csv', '2018_1.csv', '2018_2.csv', '2019_1.csv', '2019_2.csv']
            data = pd.DataFrame()
            for file in files:
                print(f'[PREPROCESSING] Read {input_dir}{file}...')
                chunk = pd.read_csv(f'{input_dir}{file}')
                data = data.append(chunk, ignore_index=True)
            print(f'[PREPROCESSING] Extracting main content...', end=' ')
            documents = extract_dataframe_content(data, 'body')
            print(f'preprocessing...', end=' ')
            texts = [process_text(body) for body in documents]
            print(f'building dictionary...', end=' ')
            dictionary = corpora.Dictionary(texts)
            print(f'buiding corpus...', end=' ')
            corpus = [dictionary.doc2bow(text) for text in texts]
            print(f'saving ids...')
            doc_ids = extract_dataframe_content(data, 'id')
            print(f'completed!')

            with open(f'{output_dir}_cache_tokens.pkl', 'wb') as f:
                pickle.dump(texts, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'{output_dir}_cache_corpus.pkl', 'wb') as f:
                pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'{output_dir}_cache_dictionary.pkl', 'wb') as f:
                pickle.dump(dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'{output_dir}_cache_doc_ids.pkl', 'wb') as f:
                pickle.dump(doc_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
    return (texts, corpus, dictionary, doc_ids)

def postprocess(input_dir, output_dir, best_model=None, lda_components=None):
    if not lda_components:
        raise Exception(f'[POSTPROCESSING] No lda_components provided. Exit!')
        
    (texts, corpus, dictionary, doc_ids) = lda_components

    if best_model == None:
        if not exists(f'{output_dir}gridsearch_result_best_model.gensim'):
            raise Exception(f'[POSTPROCESSING] No model learnt or provided. Exit!')
        else:
            best_model = LdaModel.load(f'{output_dir}gridsearch_result_best_model.gensim')

    print(f'[POSTPROCESSING] Creating reports...')

    topics = best_model.get_topics()
    reports = {}
    for i in range(len(topics)):
        topic_ith = best_model.show_topic(i)
        reports[i] = {
            'topic_id': i,
            'keywords': [dist[0] for dist in topic_ith],
            'document_ids': []
        }
    outliers = 0
    for i, row in enumerate(best_model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        if len(row):
            doc_id = doc_ids[i]
            topic_id = row[0][0]
            reports[topic_id]['document_ids'].append(doc_id)
        else:
            outliers = outliers + 1
    # print("num of outliers", outliers, "len(row)=", len(row))

    with open(f'{output_dir}gridsearch_report_full.json', 'w') as f:
        json.dump(reports, f, indent=4)

    if len(output_dir) > 0:
        panel = pyLDAvis.gensim_models.prepare(best_model, corpus, dictionary=best_model.id2word)
        pyLDAvis.save_html(panel, f'{output_dir}/pyLDAvis.html')

def start_gridsearch(input_dir, output_dir):
    # Alpha parameter
    alpha = list(np.arange(0.01, 1, 0.1))
    alpha.append('symmetric')
    alpha.append('asymmetric')
    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.1))
    beta.append('symmetric')

    """ 
        Gensim's default: {num=topics=100, learning_decay=0.5, alpha=symmetric, beta=None} 
    """
    params = {
        'num_topics': [118],
        'learning_decay': [1.0],    
        'alpha': [0.31000000000000005],
        'beta': [0.91]
    }

    """
        Memory intense:
            True: use online learning, slower but can be used on weak computer
            False: does not use online learning, faster but requires high resources
    """
    memory_intense = False

    lda_components = preprocess(
        input_dir=input_dir, 
        output_dir=output_dir, 
        memory_intense=memory_intense
    )

    best_model = gridsearch(
        input_dir=input_dir, 
        output_dir=output_dir, 
        params=params, 
        coherence='c_uci',
        lda_components=lda_components
    )
    
    postprocess(
        input_dir=input_dir, 
        output_dir=output_dir, 
        best_model=best_model, 
        lda_components=lda_components
    )
  