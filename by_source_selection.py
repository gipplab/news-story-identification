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

def get_outlet_polarity(df, id):
    for _, row in df.iterrows():
        if int(row['id']) == id:
            return row['label']
    return None

def analyze_source_selection(df, features_collection, polarity_classifier, threshold=0.5, folder=None):
    for _, row in df.iterrows():
        id_1 = int(row['id'])
        results = {
            'reused_id': row['id'],
            'reused_datetime': row['datetime'],
            'reused_label': row['label'],
            'potential_sources': []
        }
        for feature in features_collection:
            if id_1 != int(feature['article_1_id']) and id_1 != int(feature['article_2_id']):
                continue

            datetime_1 = datetime.strptime(feature['article_1_publish_date'], '%d/%m/%Y %H:%M:%S')
            datetime_2 = datetime.strptime(feature['article_2_publish_date'], '%d/%m/%Y %H:%M:%S')

            reversed = False

            if int(feature['article_1_id']) == id_1:        # article_1 = source, article_2 = reused
                if datetime_2 > datetime_1:
                    continue

            if int(feature['article_2_id']) == id_1:        # article_1 = reused, article_2 = source
                reversed = True
                if datetime_2 < datetime_1:
                    continue
            
            sim_scores = np.asarray(feature['features'])

            if not reversed:
                reused = { 'id': feature['article_1_id'], 'datetime': feature['article_1_publish_date'], 'paragraphs_length': feature['article_1_paragraph_length'], 'sentences': feature['article_1_sentences'] }
                source = { 'id': feature['article_2_id'], 'datetime': feature['article_2_publish_date'], 'paragraphs_length': feature['article_2_paragraph_length'], 'sentences': feature['article_2_sentences'] }
            else:
                sim_scores = sim_scores.transpose()
                reused = { 'id': feature['article_2_id'], 'datetime': feature['article_2_publish_date'], 'paragraphs_length': feature['article_2_paragraph_length'], 'sentences': feature['article_2_sentences'] }
                source = { 'id': feature['article_1_id'], 'datetime': feature['article_1_publish_date'], 'paragraphs_length': feature['article_1_paragraph_length'], 'sentences': feature['article_1_sentences'] }
            
            reused['label'] = get_outlet_polarity(df, reused['id'])
            source['label'] = get_outlet_polarity(df, source['id'])

            new_source = {
                'source_id': source['id'],
                'source_datetime': source['datetime'],
                'source_label': source['label'],
                'details': [],
            }

            # save to reused_id
            for i in range(reused['paragraphs_length']):
                classified_label = polarity_classifier(reused['sentences'][i])[0]
                for j in range(source['paragraphs_length']):
                    if sim_scores[i][j] < threshold:
                        continue

                    new_source['details'].append({
                        'source_text': source['sentences'][j],
                        'source_label_from_outlet': source['label'],
                        'reused_text': reused['sentences'][i],
                        'reused_label_from_outlet': reused['label'],
                        'reused_label_from_classifier': classified_label,
                        'transform_flow': f'from `{source["label"]}` to `{reused["label"]}`, it should be `{classified_label["label"]}`'  
                    })
            results['potential_sources'].append(new_source)
        results_folder = folder if folder != None else f'./{FOLDER}/by_source_selection' 
        Path(results_folder).mkdir(parents=True, exist_ok=True)
        results_filename = f"./{results_folder}/by_source_selection_{DATASET}_of_article_{id_1}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"    
        io.write_json(results_filename, results)

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
        # analyze_polarity(df, features, classifier, f'./{FOLDER}/{file.split("/")[1]}/')
        analyze_source_selection(df, features, classifier)

    if consts.openShell:
        openShell()
        