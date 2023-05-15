from datetime import datetime
from os.path import join
from pathlib import Path

import numpy as np
from transformers import pipeline
import functions as f
import consts
from consts import dataset
from functions import openShell, read_data
from modules.preprocessing import io
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.distance_measures import center
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

#   'aic' is short for 'article in consideration'
#   'ea' is short for 'earlier article'
#
#   paraphrase threshold: if simscore between 2 paragraphs/sentences are higher than this value, then they are plagiarized
#   reused_threshold: e.g 0.5, means more than 50% paragraphs/sentences from aic are paraphrased in ea
def analyze_source_selection(df, features_collection, polarity_classifier, paraphrase_threshold=0.75, reused_threshold=0.8, folder=None, topic_name='Unknown'):   
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
            'biased_by_these_sources': [],
            'biased_labels': {
                'LEFT': 0,
                'CENTER': 0,
                'RIGHT': 0
            },
            'is_biased': "No",
            'earlier_articles': [],
        }

        #
        # Prepare nodes for Directed graph
        #
        node_list.append((str(results['article_id']), str(consts.COLOR_CODE[results["article_label"]])))
    
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
                },
                'reused_details': []
            }

            for i in range(ea['paragraphs_length']):
                classified_label = polarity_classifier(ea['sentences'][i])[0]
                reused_detail = {
                    'reused': ea['sentences'][i],
                    'label': classified_label,
                    'reused_by_aic': []
                }
                has_reused = False
                for j in range(aic['paragraphs_length']):
                    # sim_scores[i][j] = similarity score between earlier_article's paragraph i-th and article_in_consideration's paragraph j-th
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
            for label in consts.Labels:
                if results['biased_labels'][label] > max_l:
                    max_l = results['biased_labels'][label] 
                    max_label = label
            if max_label != 'CENTER':
                results['is_biased'] = f"Yes, to the {max_label}"

            #
            # Prepare edges for Directed graph
            #
            for ea_id in results['biased_by_these_sources']:
                edges.append((str(ea_id), str(aic_id)))

        results_folder = folder if folder != None else f'./{FOLDER}/by_source_selection' 
        Path(results_folder).mkdir(parents=True, exist_ok=True)
        # datetime: _{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        results_filename = f"./{results_folder}/by_source_selection_{DATASET}_of_article_{aic_id}.json"
        chart_filename = f"./{results_folder}/by_source_selection_{DATASET}_of_article_{aic_id}.png"
        file_info['articles'].append({
            "article_id": aic_id,
            "article_title": f'{aic_title} + ({aic_label})',
            "analyzed": results_filename,
            "chart": chart_filename if (len(results['biased_by_these_sources']) > 0) else ''
        })
        io.write_json(results_filename, results)
        if len(results['biased_by_these_sources']) > 0:
            #
            # STACKED BAR CHART
            #
            build_chart(results, chart_filename)
            
    #
    # BUILD DIRECTED GRAPH
    #
    if len(node_list) and len(edges):
        graph_filename = f"./{results_folder}/by_source_selection_{DATASET}_network_topic_{topic_name}.png"
        graph_title = f'Plagiarism map among articles in topic {topic_name}'
        file_info['network'] = graph_filename
        build_graph(node_list=node_list, edges=edges, filename=graph_filename, graph_title=graph_title)
    
    return file_info


def build_chart(results, filename):
    plt.figure()
    
    L = []
    R = []
    C = []
    ea_ids = []
    for ea in results['earlier_articles']:
        if ea['is_biased_to'] != 'No':
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
        if(label_text != 0):
            ax.text(label_x, label_y, label_text, ha='center', va='center', color='white', weight='bold')

    ax.set_xticklabels(ea_ids)
    ax.set_title(f'Number of paragraphs resued by article {results["article_id"]} ({results["article_label"]}) and their classified polarity')
    ax.set_xlabel('Id of earlier articles')
    ax.set_ylabel('Number of reused paragraphs')
    plt.savefig(filename)
    plt.cla()

#
# element of node_list is a tuple of (name, color). e.g: ('1', 'red'), ('2', 'gray')...
# element of edges is a tuple of names. E.g: ('1', '2') means node 1 to node 2
#
def build_graph(node_list, edges, filename, graph_title):    

    G = nx.DiGraph()
    for n in node_list:
        G.add_node(n[0], color=n[1])
    G.add_edges_from(edges)
    colors = [node[1]['color'] for node in G.nodes(data=True)]
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    ax.set_title(graph_title)
    plt.ylabel('Red nodes denote leaning left articles, Blue nodes denote leaning right articles, Center nodes denote neutral articles')
    nx.draw(G, ax=ax, node_color=colors, with_labels=True, font_color='white')
    plt.title(graph_title)
    plt.savefig(filename)
    plt.cla()

if __name__ == "__main__":
    DATASET = 'GROUNDNEWS'
    DATASET_VERSION = 'Full'
    FOLDER = dataset[DATASET][DATASET_VERSION]['FOLDER']
    FILES = dataset[DATASET][DATASET_VERSION]['FILES']

    files_info = []
    for i, file in enumerate(FILES):
        df = read_data(FOLDER, [file])
        df = df.dropna()
        try:
            topic_id = file.split("/")[1]
            features = read_features(FOLDER, f'./{topic_id}/features.json')
        except:
            print(f'File {file} does not have features extracted. Skipped')
            continue
        classifier = pipeline("text-classification", model=f'./model/{consts.polarity_classifier_path}')
        file_info = analyze_source_selection(
            df=df, 
            features_collection=features, 
            polarity_classifier=classifier, 
            paraphrase_threshold=consts.paraphrase_threshold,
            topic_name=topic_id
        )

        files_info.append(file_info)

    folder = f'./{FOLDER}/by_source_selection'
    io.write_json(f"./{folder}/files_info.json", files_info)

    if consts.openShell:
        f.showToast("Bias by source selection - Main")
        openShell()
        