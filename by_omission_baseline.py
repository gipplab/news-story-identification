from datetime import datetime
from pathlib import Path

import numpy as np
from transformers import pipeline

import consts
import functions as f
from modules.preprocessing import io


#   'aic' is short for 'article in consideration'
#   'ea' is short for 'earlier article'
#
#   paraphrase threshold: limitation to check if 2 paragraphs/sentences are plagirized
#   reused_threshold: e.g 0.8, means more than 80% paragraphs/sentences from aic are paraphrased in ea
#   omission_bias_threshold: e.g 0.5, means more than 50% of non-plagiarized paragraphs from ea are not slanted
def analyze_omission(df, features_collection, polarity_classifier, paraphrase_threshold=0.75, omission_bias_threshold=0.5, reused_threshold=0.8, folder=None):
    for _, row in df.iterrows():
        aic_id = int(row['id'])
        results = {            
            'article_id': row['id'],
            'article_datetime': row['datetime'],
            'article_label': row['label'],
            'article_total_paragraphs': 0,
            'earlier_articles': [],
        }
        is_earliest = True
        for feature in features_collection:            
            if aic_id != int(feature['article_1_id']) and aic_id != int(feature['article_2_id']):
                continue

            datetime_1 = datetime.strptime(feature['article_1_publish_date'], '%d/%m/%Y %H:%M:%S')
            datetime_2 = datetime.strptime(feature['article_2_publish_date'], '%d/%m/%Y %H:%M:%S')

            reversed = False

            # map to correct order from features
            if int(feature['article_1_id']) == aic_id:
                if datetime_2 > datetime_1:
                    continue

            if int(feature['article_2_id']) == aic_id:
                reversed = True
                if datetime_2 < datetime_1:
                    continue

            is_earliest = False
            sim_scores = np.asarray(feature['features'])

            if reversed:
                ea = { 'id': feature['article_1_id'], 'label': feature['article_1_label'], 'datetime': feature['article_1_publish_date'], 'paragraphs_length': feature['article_1_paragraph_length'], 'sentences': feature['article_1_sentences'] }
                aic = { 'id': feature['article_2_id'], 'label': feature['article_2_label'], 'datetime': feature['article_2_publish_date'], 'paragraphs_length': feature['article_2_paragraph_length'], 'sentences': feature['article_2_sentences'] }
            else:
                sim_scores = sim_scores.transpose()
                ea = { 'id': feature['article_2_id'], 'label': feature['article_2_label'], 'datetime': feature['article_2_publish_date'], 'paragraphs_length': feature['article_2_paragraph_length'], 'sentences': feature['article_2_sentences'] }
                aic = { 'id': feature['article_1_id'], 'label': feature['article_1_label'], 'datetime': feature['article_1_publish_date'], 'paragraphs_length': feature['article_1_paragraph_length'], 'sentences': feature['article_1_sentences'] }
            results['article_total_paragraphs'] = aic['paragraphs_length']
            # print('aic:', aic['id'], aic['datetime'], aic['paragraphs_length'])
            # print('ea:', ea['id'], ea['datetime'], ea['paragraphs_length'])
            analyzed = {
                'article_id': ea['id'],
                'label': ea['label'],
                'datetime': ea['datetime'],                
                'total_paragraphs': ea['paragraphs_length'],
                'total_reused_paragraphs': 0,    
                'reused_ratio': 0,
                'is_biased_by_source_selection': 'No',
                'reused_paragraphs_label': {
                    'LEFT': 0,
                    'CENTER': 0,
                    'RIGHT': 0
                },
                'reused_percentage': {
                    'LEFT': 0,
                    'CENTER': 0,
                    'RIGHT': 0
                },
                'is_biased_by_omission': 'No',
                'total_non-reused_paragraphs': 0,
                'non-reused_paragraphs_label': {
                    'LEFT': 0,
                    'CENTER': 0,
                    'RIGHT': 0
                },                
                'non-reused_percentage': {
                    'LEFT': 0,
                    'CENTER': 0,
                    'RIGHT': 0
                }
            }

            for i in range(ea['paragraphs_length']):
                for j in range(aic['paragraphs_length']):
                    classified_label = polarity_classifier(ea['sentences'][i])[0]
                    # sim_scores[i][j] = similarity score between earlier_article's paragraph i-th and article_in_consideration's paragraph j-th
                    # if sim_scores > threshold, that means aic has reused paragraph i-th in its paragraph j-th
                    if sim_scores[i][j] > paraphrase_threshold:
                        analyzed['total_reused_paragraphs'] += 1
                        analyzed['reused_paragraphs_label'][classified_label["label"]] += 1
                    else:
                        analyzed['total_non-reused_paragraphs'] += 1
                        analyzed['non-reused_paragraphs_label'][classified_label["label"]] += 1
            if analyzed['total_reused_paragraphs'] > 0:
                for label in consts.Labels:
                    analyzed['reused_percentage'][label] = analyzed['reused_paragraphs_label'][label] / analyzed['total_reused_paragraphs']

            if analyzed['total_non-reused_paragraphs'] > 0:
                for label in consts.Labels:
                    analyzed['non-reused_percentage'][label] = analyzed['non-reused_paragraphs_label'][label] / analyzed['total_non-reused_paragraphs']

            is_biased_by_source_selection = ''

            analyzed['reused_ratio'] = round(analyzed['total_reused_paragraphs'] / results['article_total_paragraphs'], 2)

            if analyzed['reused_ratio'] > reused_threshold: 
                highest_percent = 0
                highest_label = None      
                for label in consts.Labels:
                    if label :
                        if analyzed['reused_percentage'][label] > highest_percent:
                            highest_percent = analyzed['reused_percentage'][label]
                            highest_label = label
                if highest_label:
                    is_biased_by_source_selection = f'Yes, to the {highest_label}. Percentage: {"{:0.2%}".format(highest_percent)}'
                analyzed['is_biased_by_source_selection'] = is_biased_by_source_selection
            if len(is_biased_by_source_selection) == 0:
                analyzed['is_biased_by_source_selection'] = "No"
            else:
                # start checking bias by commission
                is_biased_by_omission = "No"                
                non_reused_percentage = analyzed['non-reused_percentage']['CENTER']
                if analyzed['label'] == 'LEFT':
                    non_reused_percentage += analyzed['non-reused_percentage']['RIGHT']
                if analyzed['label'] == 'RIGHT':
                    non_reused_percentage += analyzed['non-reused_percentage']['LEFT']
                if non_reused_percentage > omission_bias_threshold:
                    is_biased_by_omission = f'Yes, by more than {"{:0.2%}".format(non_reused_percentage)}'
                analyzed['is_biased_by_omission'] = is_biased_by_omission
            
            results['earlier_articles'].append(analyzed)

        if is_earliest:
            results['is_biased'] = 'This is the earliest article'
            
        results_folder = folder if folder != None else f'./{FOLDER}/by_omission' 
        Path(results_folder).mkdir(parents=True, exist_ok=True)
        results_filename = f"./{results_folder}/by_omission_{DATASET}_of_article_{aic_id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"    
        io.write_json(results_filename, results)

if __name__ == "__main__":
    DATASET = 'GROUNDNEWS'
    DATASET_VERSION = 'Full'
    FOLDER = consts.dataset[DATASET][DATASET_VERSION]['FOLDER']
    FILES = consts.dataset[DATASET][DATASET_VERSION]['FILES']
    
    for i, file in enumerate(FILES):
        df = f.read_data(FOLDER, [file])
        df = df.dropna()
        try:
            features = f.read_features(FOLDER, f'./{file.split("/")[1]}/features.json')
        except Exception as e:
            print(e)
            continue
        classifier = pipeline("text-classification", model=f'./model/{consts.polarity_classifier_path}')
        analyze_omission(
            df=df, 
            features_collection=features, 
            polarity_classifier=classifier, 
            paraphrase_threshold=consts.paraphrase_threshold
        )

    if consts.openShell:
        f.showToast("Bias by omission - Baseline")
        f.openShell()
        