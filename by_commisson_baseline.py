
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
#   paraphrase threshold: e.g 0.75, means if simscore of a sentence-pair is > 0.75 then this pair is plagiarized
#   reused_threshold: e.g 0.8, means if more than 80% paragraphs/sentences from aic are plagizarized, then aic is plagiarized
#   comission_bias_threshold: e.g 0.8, means if more than 80% of plagiarized paragraphs are slanted, than aic is bias by commission (to the slanted labal (L/R))
def analyze_commission(df, features_collection, polarity_classifier, paraphrase_threshold=0.75, commission_bias_threshold=0.8, reused_threshold=0.5, folder=None):
    for _, row in df.iterrows():
        aic_id = int(row['id'])
        results = {            
            'article_id': row['id'],
            'article_datetime': row['datetime'],
            'article_label': row['label'],
            'total_paragraphs': 0,
            'total_reused_paragraphs': 0,
            'reused_percentage': 0,
            'is_biased': "No",
            'ea_label': {
                'LEFT': 0,
                'CENTER': 0,
                'RIGHT': 0
            },
            'ea_label_ratio': {
                'LEFT': 0,
                'CENTER': 0,
                'RIGHT': 0
            },
            'earlier_articles': []
        }
        is_earliest = True
        for feature in features_collection:            
            if aic_id != int(feature['article_1_id']) and aic_id != int(feature['article_2_id']):
                continue

            datetime_1 = datetime.strptime(feature['article_1_publish_date'], '%d/%m/%Y %H:%M:%S')
            datetime_2 = datetime.strptime(feature['article_2_publish_date'], '%d/%m/%Y %H:%M:%S')

            reversed = False

            if int(feature['article_1_id']) == aic_id:
                results['total_paragraphs'] = feature['article_1_paragraph_length']
                if datetime_2 > datetime_1:
                    continue

            if int(feature['article_2_id']) == aic_id:
                results['total_paragraphs'] = feature['article_2_paragraph_length']
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
            
            for i in range(aic['paragraphs_length']):
                for j in range(ea['paragraphs_length']):
                    # sim_scores[i][j] = similarity score between article_in_consideration's paragraph i-th and earlier_article's paragraph j-th
                    # if sim_scores > threshold, that means aic has reused paragraph i-th in its paragraph j-th
                    if sim_scores[j][i] > paraphrase_threshold:
                        # classified_label = polarity_classifier(aic['sentences'][i])[0]
                        if ea['id'] not in results['earlier_articles']:
                            results['earlier_articles'].append({
                                ea['id']: ea['label']    
                            })

        if is_earliest:
            results['is_biased'] = 'This is the earliest article'
        if len(results['earlier_articles']) > 0:
            for ea in results['earlier_articles']:
                for id, label in ea.items():
                    results['ea_label'][label] += 1
                    results['ea_label_ratio'][label] = results['ea_label'][label] / len(results['earlier_articles'])
        
        if results['total_paragraphs'] > 0:
            results['reused_percentage'] = results['total_reused_paragraphs'] / results['total_paragraphs']
            if results['total_reused_paragraphs'] > 0:
                for label in consts.Labels:
                    results['reused_label_ratio'][label] = results['reused_label'][label] / results['total_reused_paragraphs']
            
        is_bias = ''
        max_label_ratio = 0
        max_label = ''
        for label in consts.Labels:
            if results['ea_label_ratio'][label] > max_label_ratio:
                max_label_ratio = results['ea_label_ratio'][label]
                max_label = label
        if max_label != 'CENTER':
            is_bias += f'Yes, to the {max_label}. Percentage: {max_label_ratio}'
        if len(is_bias) > 0:
            results['is_biased'] = is_bias
        results_folder = folder if folder != None else f'./{FOLDER}/by_commission' 
        Path(results_folder).mkdir(parents=True, exist_ok=True)
        results_filename = f"./{results_folder}/by_commission_{DATASET}_of_article_{aic_id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"    
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
        analyze_commission(
            df=df, 
            features_collection=features, 
            polarity_classifier=classifier, 
            paraphrase_threshold=consts.paraphrase_threshold
        )

    if consts.openShell:
        f.showToast("Bias by omission - Baseline")
        f.openShell()
        