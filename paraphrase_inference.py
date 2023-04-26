from datetime import datetime
from os.path import join

import matplotlib as mpl
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

import consts
import functions as f
from consts import dataset
from modules.preprocessing.io import write_json


def split_sentences(input_text=""):
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sentence_tokenizer.tokenize(input_text)        
    return sentences

def read_data(folder, files, header=0, encoding='utf-8'):
    li = []
    for filename in files:
        print(f'=== Reading {join(folder, filename)}')
        if filename.endswith('csv'):
            df = pd.read_csv(join(folder, filename), index_col=None, header=header, encoding=encoding)
        if filename.endswith('tsv'):
            df = pd.read_csv(join(folder, filename), index_col=None, header=header, encoding=encoding, sep='\t')
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)

    return df

def detect_paraphrases(sentences_1, sentences_2):
    results = []
    for i in range(len(sentences_1)):
        for j in range(len(sentences_2)):
            if len(sentences_1[i]) >= min_sentence_length and len(sentences_2[j]) >= min_sentence_length:
                emb1 = model.encode(sentences_1[i])
                emb2 = model.encode(sentences_2[j])

                cos_sim = util.cos_sim(emb1, emb2)
                if cos_sim >= cosine_threshold:
                    results.append({
                        'article_1_sentence': sentences_1[i],
                        'article_1_sentence_index': i,
                        'article_1_sentence_length': len(sentences_1[i]),
                        'article_2_sentence': sentences_2[j],
                        'article_2_sentence_index': j,
                        'article_2_sentence_length': len(sentences_2[j]),
                        'score': float(cos_sim)
                    })
    return results

def build_chart(data, filename=''):
    articles = []
    for d in data:
        if len([a for a in articles if a['id'] == d['article_1_id']]) == 0:
            articles.append({ 'id': d['article_1_id'], 'publish_date': d['article_1_publish_date'], 'length': d['article_1_paragraph_length'] })
        if len([a for a in articles if a['id'] == d['article_2_id']]) == 0:
            articles.append({ 'id': d['article_2_id'], 'publish_date': d['article_2_publish_date'], 'length': d['article_2_paragraph_length'] })
    
    # sort by publish date
    articles = sorted(articles, key=lambda e: e['publish_date'])
    
    labels = [f'{a["id"]}' if i == 1 else '' for a in articles for i in range(1, a['length'] + 1)]
    chart_labels = [f'a{a["id"]}p{i}' for a in articles for i in range(1, a['length'] + 1)]
    d = len(chart_labels)
    chart_values = np.zeros((d, d), dtype=float)
    
    chart_values_dict = {}
    for d in data:
        features = d['features']
        for i in range(len(features)):
            for j in range(len(features[i])):              
                key = f'a{d["article_1_id"]}p{i}_a{d["article_2_id"]}p{j}'                
                chart_values_dict[key] = features[i][j] if features[i][j] > 0 else 0
                # chart_values_dict[key] = features[i][j]
    
    for i, labeli in enumerate(chart_labels):
        for j, labelj in enumerate(chart_labels):
            # if i < j:
            key = f'{labeli}_{labelj}'
            if key in chart_values_dict:
                if i < j:
                    chart_values[j][i] = chart_values_dict[key]
                else:
                    chart_values[i][j] = chart_values_dict[key]
    fig, ax = plt.subplots()
    ax.matshow(chart_values,  cmap=mpl.colormaps['Oranges'], vmin=0)
    ax.set(xticks=np.arange(len(labels)), xticklabels=labels,
           yticks=np.arange(len(labels)), yticklabels=labels)
    ax.set_xlabel('Article ids')
    ax.set_ylabel('Article ids')
    # Legends
    heatmap = ax.pcolor(chart_values, cmap=mpl.colormaps['Oranges'])
    plt.colorbar(heatmap)
    
    ax.set_title("Heatmap of Similarity scores between all sentences of all articles")
    
    if len(filename) > 0:
        plt.savefig(filename)
    else:
        plt.show()
    
    return chart_values

def build_features(sentences_1, sentences_2):
    m = len(sentences_1) 
    n = len(sentences_2)
    sim = np.zeros( (m, n) , dtype=np.float64)
    for i, s1 in enumerate(sentences_1):
        for j, s2 in enumerate(sentences_2):
            # if i < j:
            cos_sim = util.cos_sim(model.encode(s1), model.encode(s2))
            sim[i][j] = cos_sim
    return sim.tolist()

def start(df, buildFeature=True):
    
    articles = [(row['text'], row['id'], row['datetime'], row['label']) for i, row in df.iterrows()]
    results = []
    for i in range(len(articles) - 1):
        for j in range(i + 1, len(articles)):      
            print(f'=== Inferencing article id {articles[i][1]} and {articles[j][1]} / Total {len(articles)}')
            sentences_1 = split_sentences(articles[i][0])
            sentences_2 = split_sentences(articles[j][0])
            res = {
                'article_1_id': articles[i][1],
                'article_1_publish_date': articles[i][2],
                'article_1_label': articles[i][3],
                'article_1_paragraph_length': len(sentences_1),
                'article_1_sentences': sentences_1,
                'article_2_id': articles[j][1],
                'article_2_publish_date': articles[j][2],
                'article_2_label': articles[j][3],
                'article_2_sentences': sentences_2,
                'article_2_paragraph_length': len(sentences_2),
                'feature_built_at': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            }
            if buildFeature:
                features = build_features(sentences_1, sentences_2)
                res['description'] = f'Features (or Similartiy scores) between all sentences between article {articles[i][1]} and {articles[j][1]}'
                res['features'] = features
            else:
                paraphrases = detect_paraphrases(sentences_1, sentences_2)
                res['description'] = f'Paraphrases detected between article {articles[i][1]} as article_1, and {articles[j][1]} as article_2'
                res['paraphrases'] = paraphrases
            results.append(res)

    results = { 'results': results  }
    return results

#   Sample run
if __name__ == "__main__":
    DATASET = 'GROUNDNEWS'
    DATASET_VERSION = 'Full'
    FOLDER = dataset[DATASET][DATASET_VERSION]['FOLDER']
    FILES = dataset[DATASET][DATASET_VERSION]['FILES']

    if consts.inference:
        for i, file in enumerate(FILES):
            data = read_data(FOLDER, [file], encoding='ISO-8859-1')
            data = data.dropna() 

            # model_name = 'all-MiniLM-L6-v2'    
            # model_name = './model/training_OnlineConstrativeLoss-2022-12-28_23-06-03'   # Pretrained 'stsb-distilbert-base', fined-tune with QuoraQA dataset
            # model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
            model_name = consts.paraphrase_identifier_modelname
            min_sentence_length = 10
            model = SentenceTransformer(model_name)
            cosine_threshold = 0.6    

            results_filename = f"./model/features_{model_name}_{DATASET}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_results.json"    
            results = start(data)
            write_json(results_filename, results['results'])

            if consts.visualize:
                build_chart(results['results'], f'{results_filename}.png')
    if consts.openShell:
        f.showToast("Features built completed")
        f.openShell()        



