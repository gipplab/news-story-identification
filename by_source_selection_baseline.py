from datetime import datetime
from os.path import join
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import pipeline

import consts
import functions as f
from consts import dataset
from modules.preprocessing import io


#   'aic' is short for 'article in consideration'
#   'ea' is short for 'earlier article'
#
#   paraphrase threshold: if simscore between 2 paragraphs/sentences are higher than this value, then they are plagiarized
#   reused_threshold: e.g 0.5, means more than 50% paragraphs/sentences from aic are paraphrased in ea
def analyze_source_selection(df, features_collection, polarity_classifier, paraphrase_threshold=0.75, reused_threshold=0.8, folder=None):    
    for _, row in df.iterrows():
        aic_id = int(row['id'])
        results = {            
            'article_id': row['id'],
            'article_datetime': row['datetime'],
            'article_label': row['label'],
            'article_total_paragraphs': 0,
            'biased_by_these_sources': [],
            'biased_labels': {
                'LEFT': 0,
                'CENTER': 0,
                'RIGHT': 0
            },
            'is_biased': "No",
            'earlier_articles': [],
        }
        
        is_earliest = True

        # compared to other articles called earlier-article or ea
        for feature in features_collection:            
            if aic_id != int(feature['article_1_id']) and aic_id != int(feature['article_2_id']):
                continue

            datetime_1 = datetime.strptime(feature['article_1_publish_date'], '%d/%m/%Y %H:%M:%S')
            datetime_2 = datetime.strptime(feature['article_2_publish_date'], '%d/%m/%Y %H:%M:%S')

            reversed = False

            if int(feature['article_1_id']) == aic_id:        # article_1 = source, article_2 = reused
                if datetime_2 > datetime_1:
                    continue

            if int(feature['article_2_id']) == aic_id:        # article_1 = reused, article_2 = source
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
            
            analyzed = {
                'article_id': ea['id'],
                'article_label': ea['label'],
                'datetime': ea['datetime'],
                'total_paragraphs': ea['paragraphs_length'],
                'total_reused_paragraphs': 0,    
                'reused_ratio': 0,
                'is_biased_to': 'No',
                'reused_paragraphs_label': {
                    'LEFT': 0,
                    'CENTER': 0,
                    'RIGHT': 0
                },
                'reused_percentage': {
                    'LEFT': 0,
                    'CENTER': 0,
                    'RIGHT': 0
                }
            }

            for i in range(ea['paragraphs_length']):
                for j in range(aic['paragraphs_length']):
                    # sim_scores[i][j] = similarity score between earlier_article's paragraph i-th and article_in_consideration's paragraph j-th
                    # if sim_scores > threshold, that means aic has reused paragraph i-th in its paragraph j-th
                    if sim_scores[i][j] > paraphrase_threshold:
                        analyzed['total_reused_paragraphs'] += 1
                        classified_label = polarity_classifier(ea['sentences'][i])[0]
                        analyzed['reused_paragraphs_label'][ea['label']] += 1
            if analyzed['total_reused_paragraphs'] > 0:
                for label in consts.Labels:
                    analyzed['reused_percentage'][label] = analyzed['reused_paragraphs_label'][label] / analyzed['total_reused_paragraphs']

            is_biased_to = ''

            analyzed['reused_ratio'] = round(analyzed['total_reused_paragraphs'] / results['article_total_paragraphs'], 2)

            if analyzed['reused_ratio'] > reused_threshold: 
                highest_percent = 0
                highest_label = None      
                for label in consts.Labels:
                    if analyzed['reused_percentage'][label] > highest_percent:
                        highest_percent = analyzed['reused_percentage'][label]
                        highest_label = label
                if highest_label:
                    is_biased_to = f'Yes, to the {highest_label}. Percentage: {"{:0.2%}".format(highest_percent)}'
                analyzed['is_biased_to'] = is_biased_to
            if len(is_biased_to) == 0:
                analyzed['is_biased_to'] = "No"
            else:
                results['biased_by_these_sources'].append(ea['id'])
                results['biased_labels'][highest_label] += 1
            
            results['earlier_articles'].append(analyzed)

        if is_earliest:
            results['is_biased'] = 'This is the earliest article'
        if len(results['biased_by_these_sources']) > 0:
            max_l = 0
            max_label = None
            for label in consts.Labels:
                if results['biased_labels'][label] > max_l:
                    max_l = results['biased_labels'][label]
                    max_label = label
            if max_label != 'CENTER':
                results['is_biased'] = f"Yes, to the {max_label}"

        results_folder = folder if folder != None else f'./{FOLDER}/by_source_selection' 
        Path(results_folder).mkdir(parents=True, exist_ok=True)
        results_filename = f"./{results_folder}/by_source_selection_{DATASET}_of_article_{aic_id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"    
        io.write_json(results_filename, results)
        if len(results['biased_by_these_sources']) > 0:
            results_filename = f"./{results_folder}/by_source_selection_{DATASET}_of_article_{aic_id}.png"
            build_chart(results, results_filename)

def build_chart(results, filename):
    
    L = []
    R = []
    C = []
    ea_ids = []
    for ea in results['earlier_articles']:
        L.append(ea['reused_paragraphs_label']['LEFT'])
        C.append(ea['reused_paragraphs_label']['CENTER'])
        R.append(ea['reused_paragraphs_label']['RIGHT'])
        ea_ids.append(f'{ea["article_id"]}\n{ea["article_label"]}')

    df = pd.DataFrame({
        'Left': L,
        'Right': R,
        'Center': C
    })    

    colors = ['red', 'blue', 'gray']
    ax = df.plot(stacked=True, kind='bar', color=colors, figsize=(10,10))

    for bar in ax.patches:
        height = bar.get_height()
        width = bar.get_width()
        x = bar.get_x()
        y = bar.get_y()
        label_text = int(height)
        label_x = x + width / 2
        label_y = y + height / 2
        ax.text(label_x, label_y, label_text, ha='center', va='center', color='white', weight='bold')

    ax.set_xticklabels(ea_ids)
    ax.set_title(f'Number of paragraphs resued by article {results["article_id"]} and their classified polarity')
    ax.set_xlabel('Id of earlier articles')
    ax.set_ylabel('Number of reused paragraphs')
    # plt.show()
    plt.savefig(filename)
if __name__ == "__main__":
    DATASET = 'GROUNDNEWS'
    DATASET_VERSION = 'Full'
    FOLDER = dataset[DATASET][DATASET_VERSION]['FOLDER']
    FILES = dataset[DATASET][DATASET_VERSION]['FILES']

    for i, file in enumerate(FILES):
        df = f.read_data(FOLDER, [file])
        df = df.dropna()
        try:
            features = f.read_features(FOLDER, f'./{file.split("/")[1]}/features.json')
        except:
            print(f'File {file} does not have features extracted. Skipped')
            continue
        classifier = pipeline("text-classification", model=f'./model/{consts.polarity_classifier_path}')
        analyze_source_selection(
            df=df, 
            features_collection=features, 
            polarity_classifier=classifier, 
            paraphrase_threshold=consts.paraphrase_threshold
        )

    if consts.openShell:
        f.showToast("Bias by source selection - Baseline")
        f.openShell()
        