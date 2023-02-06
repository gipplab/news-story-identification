from copy import copy
from datetime import datetime
from pathlib import Path

import numpy as np
from transformers import pipeline

import consts
import functions as f
from modules.preprocessing import io

def get_outlet_polarity(df, id):
    for _, row in df.iterrows():
        if int(row['id']) == id:
            return row['label']
    return None

def analyze_omission(df, features_collection, polarity_classifier, threshold=0.5, folder=None):
    for _, row in df.iterrows():
        id_1 = int(row['id'])
        results = {}
        results = {
            'this_article_id': row['id'],
            'this_article_datetime': row['datetime'],
            'this_article_label': row['label'],
            'omissions': []
        }
        print("=== ", id_1)
        for feature in features_collection:
            if id_1 != int(feature['article_1_id']) and id_1 != int(feature['article_2_id']):
                continue

            datetime_1 = datetime.strptime(feature['article_1_publish_date'], '%d/%m/%Y %H:%M:%S')
            datetime_2 = datetime.strptime(feature['article_2_publish_date'], '%d/%m/%Y %H:%M:%S')

            reversed = False

            if int(feature['article_1_id']) == id_1:
                if datetime_2 > datetime_1:
                    continue

            if int(feature['article_2_id']) == id_1:
                reversed = True
                if datetime_2 < datetime_1:
                    continue
            
            sim_scores = np.asarray(feature['features'])

            if not reversed:
                source = { 'id': feature['article_2_id'], 'datetime': feature['article_2_publish_date'], 'paragraphs_length': feature['article_2_paragraph_length'], 'sentences': feature['article_2_sentences'] }
                reused = { 'id': feature['article_1_id'], 'datetime': feature['article_1_publish_date'], 'paragraphs_length': feature['article_1_paragraph_length'], 'sentences': feature['article_1_sentences'] }
            else:
                sim_scores = sim_scores.transpose()
                source = { 'id': feature['article_1_id'], 'datetime': feature['article_1_publish_date'], 'paragraphs_length': feature['article_1_paragraph_length'], 'sentences': feature['article_1_sentences'] }
                reused = { 'id': feature['article_2_id'], 'datetime': feature['article_2_publish_date'], 'paragraphs_length': feature['article_2_paragraph_length'], 'sentences': feature['article_2_sentences'] }
            source['label'] = get_outlet_polarity(df, source['id'])
            reused['label'] = get_outlet_polarity(df, reused['id'])
            
            new_omission = {
                'source_id': source['id'],
                'source_datetime': source['datetime'],
                'source_label': source['label'],
                'details': [],
                'polarity_changes': {
                    'to_LEFT': 0,
                    'to_CENTER': 0,
                    'to_RIGHT': 0
                }
            }

            for i in range(reused['paragraphs_length']):
                for j in range(source['paragraphs_length']):
                    # consider only sentences that are NOT reused
                    if sim_scores[i][j] >= threshold:
                        continue
                    
                    classified_label = polarity_classifier(source['sentences'][j])[0]

                    if classified_label['label'] == 'LEFT':
                        new_omission['polarity_changes']['to_LEFT'] = new_omission['polarity_changes']['to_LEFT'] + 1
                    if classified_label['label'] == 'CENTER':
                        new_omission['polarity_changes']['to_CENTER'] = new_omission['polarity_changes']['to_CENTER'] + 1
                    if classified_label['label'] == 'RIGHT':
                        new_omission['polarity_changes']['to_RIGHT'] = new_omission['polarity_changes']['to_RIGHT'] + 1

                    new_omission['details'].append({
                        'reused_text': source['sentences'][j],
                        'polarity_from_outlet': source['label'],
                        'polarity_from_classifier': classified_label
                    })
            
            results['omissions'].append(new_omission)

        results_folder = folder if folder != None else f'./{FOLDER}/by_omission' 
        Path(results_folder).mkdir(parents=True, exist_ok=True)
        results_filename = f"./{results_folder}/by_commission_{DATASET}_of_article_{reused['id']}_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.json"    
        io.write_json(results_filename, results)

if __name__ == "__main__":
    DATASET = 'GROUNDNEWS'
    DATASET_VERSION = 'Full'
    FOLDER = consts.dataset[DATASET]['Full']['FOLDER']
    FILES = consts.dataset[DATASET]['Full']['FILES']

    for i, file in enumerate(FILES):
        df = f.read_data(FOLDER, [file])
        df = df.dropna()
        try:
            features = f.read_features(FOLDER, f'./{file.split("/")[1]}/features.json')
        except Exception as e:
            print(e)
            continue
        classifier = pipeline("text-classification", model="./model/checkpoint-5138")
        analyze_omission(df=df, features_collection=features, polarity_classifier=classifier)

    if consts.openShell:
        f.openShell()
        