from copy import copy
from datetime import datetime
from os.path import join
from pathlib import Path

import numpy as np
from transformers import pipeline
import consts
from consts import dataset
from functions import openShell, read_data
from modules.preprocessing import io

def read_features(filename):
    features = io.read_json(filename)
    return features

def read_features(folder, file):
    features = io.read_json(join(folder, file))
    return features

def get_feature(features, article_1_id, article_2_id):
    for f in features:
        if int(f['article_1_id']) == int(article_1_id) and int(f['article_2_id']) == int(article_2_id):
            return f
    return None

def id_synced_with_feature(features, article_1_id, article_2_id):
    for f in features:
        if int(f['article_1_id']) == int(article_1_id) and int(f['article_2_id']) == int(article_2_id):
            return True
    return False

def find_highest_source_index(sim_scores, reused_index):
    highest_id = -1
    highest = -1 * 10 ** 7
    for i in range(len(sim_scores)):
        if sim_scores[i][reused_index] > highest:
            highest = sim_scores[i][reused_index]
            highest_id = i
    return highest_id, highest

def analyze_polarity(df, features_collection, polarity_classifier, folder=None):
    df = df.sort_values(by='datetime', ascending=True)
    
    for i in range(len(df) -1):    
        for j in range(i + 1, len(df)):        
            org = df.iloc[i]                    # article_1_id
            reused = df.iloc[j]                 # article_2_id

            if id_synced_with_feature(features_collection, org['id'], reused['id']) == False:
                org, reused = reused, org
            
            feature = get_feature(features_collection, org['id'], reused['id'])
            org_sentences = feature['article_1_sentences']
            reused_sentences = feature['article_2_sentences']

            sim_scores = np.asarray(feature['features'])
            print("Analyzing:", feature['description'])
    
            org_label = df.iloc[i]['label']
            reused_label = df.iloc[j]['label']
            results = copy(feature)
            results['analysis'] = []
            results['is_source'] = True # TODO - should be judge programmatically
            results.pop('features', None)

            for reused_index, p in enumerate(reused_sentences):
                print(f'Sim_score shape {np.shape(sim_scores)}, i: {i}, reused_index: {reused_index}')
                potential_org_id, highest_score = find_highest_source_index(sim_scores, reused_index)
                pred = polarity_classifier(p)[0]
                # print(f'Reused-{reused_index}: potential at {potential_org_id}')

                results['analysis'].append({
                    'original_sentence': org_sentences[potential_org_id],
                    'original_sentence_assigned_polarity': org_label,
                    'reused_sentence': p,
                    'reused_sentence_assigned_polarity': reused_label,
                    'sentence_similarity': highest_score,
                    'reused_sentence_classified_polarity': pred['label'],
                    'reused_sentence_classified_score': pred['score'],
                    'transform_flow': f'from `{org_label}` to `{reused_label}`, it should be `{pred["label"]}`'                    
                })
                                               
            results_folder = folder if folder != None else f'./{FOLDER}/polarity_analysis' 
            Path(results_folder).mkdir(parents=True, exist_ok=True)
            results_filename = f"./{results_folder}/by_source_selection_{DATASET}_articles_{org['id']}_{reused['id']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"    
            io.write_json(results_filename, results)
            
def by_source_selection():
    pass

if __name__ == "__main__":
    DATASET = 'GROUNDNEWS'
    DATASET_VERSION = 'Full'
    FOLDER = dataset[DATASET]['Full']['FOLDER']
    FILES = dataset[DATASET]['Full']['FILES']

    for i, file in enumerate(FILES):
        df = read_data(FOLDER, [file])
        df = df.dropna()
        try:
            features = read_features(FOLDER, f'./{file.split("/")[1]}/features.json')
        except:
            print(f'File {file} does not have features extracted. Skipped')
            continue
        classifier = pipeline("text-classification", model="./model/checkpoint-225")
        analyze_polarity(df, features, classifier, f'./{FOLDER}/{file.split("/")[1]}/')

    if consts.openShell:
        openShell()
        