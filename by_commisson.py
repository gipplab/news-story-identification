
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
def analyze_commission(df, features_collection, polarity_classifier, paraphrase_threshold=0.75, commission_bias_threshold=0.8, reused_threshold=0.5, folder=None, topic_name='Unknown'):
    node_list = []
    edges = []
    file_info = {
        "topic_id": topic_name,
        "network": "",
        "articles": []
    }
    for _, row in df.iterrows():
        aic_id = int(row['id'])
        aic_title = row['title']
        aic_label = row['label']
        results = {            
            'article_id': row['id'],
            'article_datetime': row['datetime'],
            'article_label': aic_label,
            'article_total_paragraphs': 0,
            'total_reused_paragraphs': 0,
            'reused_percentage': 0,
            'is_biased': "No",
            'biased_by_these_sources': [],
            'biased_labels': {
                'LEFT': 0,
                'CENTER': 0,
                'RIGHT': 0
            },
            'earlier_articles': [],
            'reused_label': {
                'LEFT': 0,
                'CENTER': 0,
                'RIGHT': 0
            },
            'reused_label_ratio': {
                'LEFT': 0,
                'CENTER': 0,
                'RIGHT': 0
            },
            'sentences': {}
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
            results['article_total_paragraphs'] = aic['paragraphs_length']
            print("article_total_paragraphs", results['article_total_paragraphs'])
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
                },
                'reused_details': []
            }
            # results['total_paragraphs'] = aic['paragraphs_length']
            
            for i in range(ea['paragraphs_length']):
                classified_label = polarity_classifier(ea['sentences'][i])[0]
                reused_detail = {
                    'reused': ea['sentences'][i],
                    'label': classified_label,
                    'reused_by_aic': []
                }
                has_reused = False
                for j in range(aic['paragraphs_length']):
                    # sim_scores[i][j] = similarity score between article_in_consideration's paragraph i-th and earlier_article's paragraph j-th
                    # if sim_scores > threshold, that means aic has reused paragraph i-th in its paragraph j-th
                    if sim_scores[i][j] > paraphrase_threshold:
                        if has_reused == False:
                            analyzed['total_reused_paragraphs'] += 1
                            analyzed['reused_paragraphs_label'][classified_label["label"]] += 1                            
                            has_reused = True
                        reused_detail['reused_by_aic'].append(aic['sentences'][j])
                if len(reused_detail['reused_by_aic']) > 0:
                    analyzed['reused_details'].append(reused_detail)
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
                if highest_label != 'CENTER':
                    is_biased_to = f'Yes, to the {highest_label}. Percentage: {"{:0.2%}".format(highest_percent)}'
                analyzed['is_biased_to'] = is_biased_to
                results['biased_by_these_sources'].append(ea['id'])
                results['biased_labels'][highest_label] += 1

            results['earlier_articles'].append(analyzed)

        if is_earliest:
            results['is_biased'] = 'This is the earliest article'
        if len(results['biased_by_these_sources']) > 0:
            max_l = 0
            max_label = None
            # for label in consts.Labels:
            #     if
        for k in results['sentences']:
            if results['sentences'][k]['reused']:
                label = results['sentences'][k]['classified_label']
                results['total_reused_paragraphs'] += 1
                results['reused_label'][label] += 1

        if results['total_paragraphs'] > 0:
            results['reused_percentage'] = results['total_reused_paragraphs'] / results['total_paragraphs']
            if results['total_reused_paragraphs'] > 0:
                for label in consts.Labels:
                    results['reused_label_ratio'][label] = results['reused_label'][label] / results['total_reused_paragraphs']
            
            is_bias = ''
            if results['reused_percentage'] > reused_threshold:
                for label in consts.Labels:
                    if label != "CENTER":
                        if results['reused_label_ratio'][label] > commission_bias_threshold:
                            is_bias += f'Yes, to the {label}. Percentage: {results["reused_label_ratio"][label]}'
            if len(is_bias) > 0:
                results['is_biased'] = is_bias
        results_folder = folder if folder != None else f'./{FOLDER}/by_commission' 
        Path(results_folder).mkdir(parents=True, exist_ok=True)
        results_filename = f"./{results_folder}/by_commission_{DATASET}_of_article_{aic_id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        chart_filename = f"./{results_folder}/by_commission_{DATASET}_of_article_{aic_id}.png"
        file_info['articles'].append({
            "article_id": aic_id,
            "article_title": f'{aic_title} + ({aic_label})',
            "analyzed": results_filename,
            "chart": chart_filename if (len(results['biased_by_these_sources']) > 0) else ''
        })
        io.write_json(results_filename, results)

if __name__ == "__main__":
    DATASET = 'GROUNDNEWS'
    DATASET_VERSION = 'Full'
    FOLDER = consts.dataset[DATASET][DATASET_VERSION]['FOLDER']
    FILES = consts.dataset[DATASET][DATASET_VERSION]['FILES']

    files_info = []
    for i, file in enumerate(FILES):
        df = f.read_data(FOLDER, [file])
        df = df.dropna()
        try:
            topic_id = file.split("/")[1]
            features = f.read_features(FOLDER, f'./{topic_id}/features.json')
        except Exception as e:
            print(e)
            continue
        classifier = pipeline("text-classification", model=f'./model/{consts.polarity_classifier_path}')
        file_info = analyze_commission(
            df=df, 
            features_collection=features, 
            polarity_classifier=classifier, 
            paraphrase_threshold=consts.paraphrase_threshold,
            topic_name=topic_id            
        )

        files_info.append(file_info)

    if consts.openShell:
        f.showToast("Bias by commission - Main")
        f.openShell()
        