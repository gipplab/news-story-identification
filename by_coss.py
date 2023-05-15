
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
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
def analyze_coss(df, features_collection, polarity_classifier, folder, paraphrase_threshold=0.75, commission_bias_threshold=0.8, omission_bias_threshold=0.5, reused_threshold=0.5, topic_name='Unknown'):
    node_list = []
    edges = []
    topic_info = {
        "topic_id": topic_name,
        "network": "",
        "graph": {
            "nodes": [],
            "edges": []
        },
        "articles": []
    }
    for _, row in df.iterrows():
        aic_id = int(row['id'])
        aic_title = row['title']
        aic_label = row['label']
        results = {            
            'aic_id': row['id'],
            'aic_datetime': row['datetime'],
            'aic_label': aic_label,
            'aic_total_paragraphs': 0,
            'aic_reused_paragraphs': 0,
            'aic_reused_percent': 0,
            'aic_non_reused_paragraphs': 0,
            'aic_non_reused_percent': 0,
            'is_earliest': False,
            'by_source_selection': {
                'verdict': 'No',
                'committed_articles': [],
                'biased_source_label': {
                    'LEFT': 0,
                    'CENTER': 0,
                    'RIGHT': 0
                },
            },
            'by_commission': {
                'verdict': 'No',
                'aic_reused_label': {
                    'LEFT': 0,
                    'CENTER': 0,
                    'RIGHT': 0
                },
                'aic_reused_label_percent': {
                    'LEFT': 0,
                    'CENTER': 0,
                    'RIGHT': 0
                }
            },
            'by_omission': {
                'verdict': 'No',
                'omitted_articles': [],
                'aic_non_reused_label': {
                    'LEFT': 0,
                    'CENTER': 0,
                    'RIGHT': 0
                },
                'aic_non_reused_label_percent': {
                    'LEFT': 0,
                    'CENTER': 0,
                    'RIGHT': 0
                }
            },            
            # 'biased_by_these_sources': [],
            # 'biased_source_label': {
            #     'LEFT': 0,
            #     'CENTER': 0,
            #     'RIGHT': 0
            # },
            'is_biased_by_source_selection': "No",
            'earlier_articles': [],
        }
        aic_marked_reused = []
        aic_marked_nonreused = []
        #
        # Prepare nodes for Directed graph
        #
        node_list.append((str(results['aic_id']), str(consts.COLOR_CODE[results["aic_label"]])))
        topic_info["graph"]["nodes"].append({"id": int(results['aic_id']), "label": str(results['aic_id']), "color": str(consts.COLOR_CODE[results["aic_label"]]) })
        
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
                ea = { 
                    'id': feature['article_1_id'], 
                    'label': feature['article_1_label'], 
                    'datetime': feature['article_1_publish_date'], 
                    'paragraphs_length': feature['article_1_paragraph_length'], 
                    'sentences': feature['article_1_sentences'] }
                aic = { 
                    'id': feature['article_2_id'], 
                    'label': feature['article_2_label'], 
                    'datetime': feature['article_2_publish_date'], 
                    'paragraphs_length': feature['article_2_paragraph_length'], 
                    'sentences': feature['article_2_sentences'] }
            else:
                sim_scores = sim_scores.transpose()
                ea = { 
                    'id': feature['article_2_id'], 
                    'label': feature['article_2_label'], 
                    'datetime': feature['article_2_publish_date'], 
                    'paragraphs_length': feature['article_2_paragraph_length'], 
                    'sentences': feature['article_2_sentences'] 
                }
                aic = { 
                    'id': feature['article_1_id'], 
                    'label': feature['article_1_label'], 
                    'datetime': feature['article_1_publish_date'], 
                    'paragraphs_length': feature['article_1_paragraph_length'], 
                    'sentences': feature['article_1_sentences']
                }
            results['aic_total_paragraphs'] = aic['paragraphs_length']
            if aic_marked_reused == []:
                aic_marked_reused = [False for i in range(results['aic_total_paragraphs'])]
            if aic_marked_nonreused == []:
                aic_marked_nonreused = [False for i in range(results['aic_total_paragraphs'])]

            analyzed = {
                'ea_id': ea['id'],
                'ea_label': ea['label'],
                'ea_datetime': ea['datetime'],
                'ea_total_paragraphs': ea['paragraphs_length'],
                'ea_total_reused_paragraphs': 0,    
                'ea_reused_ratio': 0,
                'ea_total_non_reused_paragraphs': 0,
                'ea_non_reused_ratio': 0,
                'ea_by_source_selection': 'No',
                'ea_reused_label': {
                    'LEFT': 0,
                    'CENTER': 0,
                    'RIGHT': 0
                },
                'ea_reused_label_percent': {
                    'LEFT': 0,
                    'CENTER': 0,
                    'RIGHT': 0
                },
                'ea_non_reused_label': {
                    'LEFT': 0,
                    'CENTER': 0,
                    'RIGHT': 0
                },
                'ea_non_reused_label_percent': {
                    'LEFT': 0,
                    'CENTER': 0,
                    'RIGHT': 0
                },
                'ea_reused_details': [],
                'ea_non_reused_details': []
            }

            ea_marked_reused = [False for i in range(ea['paragraphs_length'])]
            for i in range(ea['paragraphs_length']):
                for j in range(aic['paragraphs_length']):
                    if sim_scores[i][j] > paraphrase_threshold:
                        ea_marked_reused[i] = True

            analyzed['ea_total_reused_paragraphs'] = ea_marked_reused.count(True)
            analyzed['ea_reused_ratio'] = analyzed['ea_total_reused_paragraphs'] / analyzed['ea_total_paragraphs']
            analyzed['ea_total_non_reused_paragraphs'] = ea_marked_reused.count(False)
            analyzed['ea_non_reused_ratio'] = analyzed['ea_total_non_reused_paragraphs'] / analyzed['ea_total_paragraphs']

            #   Stats
            for i in range(ea['paragraphs_length']):
                ea_classified_label = polarity_classifier(ea['sentences'][i])[0]
                if ea_marked_reused[i] == True:
                    analyzed['ea_reused_label'][ea_classified_label["label"]] += 1
                else:
                    analyzed['ea_non_reused_label'][ea_classified_label["label"]] += 1

            for label in consts.Labels:
                if analyzed['ea_total_reused_paragraphs'] > 0:
                    analyzed['ea_reused_label_percent'][label] = analyzed['ea_reused_label'][label] / analyzed['ea_total_reused_paragraphs']
                if analyzed['ea_total_non_reused_paragraphs'] > 0:
                    analyzed['ea_non_reused_label_percent'][label] = analyzed['ea_non_reused_label'][label] / analyzed['ea_total_non_reused_paragraphs']
                
            #   Details
            for i in range(ea['paragraphs_length']):
                ea_classified_label = polarity_classifier(ea['sentences'][i])[0]
                reused_detail = {
                    'text': ea['sentences'][i],
                    'label': ea_classified_label,
                    'reused_by_aic': []
                }

                if ea_marked_reused[i] == True:
                    # has_reused = False      
                    for j in range(aic['paragraphs_length']):
                        # sim_scores[i][j] = similarity score between earlier_article's paragraph i-th and article_in_consideration's paragraph j-th
                        # if sim_scores > threshold, that means aic has reused paragraph i-th in its paragraph j-th
                        aic_classified_label = polarity_classifier(aic['sentences'][j])[0]

                        if sim_scores[i][j] > paraphrase_threshold:
                            reused_detail['reused_by_aic'].append(aic['sentences'][j])
                            if aic_marked_reused[j] == False:
                                aic_marked_reused[j] = True                       
                                results['by_commission']['aic_reused_label'][aic_classified_label['label']] += 1   
                else:
                    analyzed['ea_non_reused_details'].append({
                        'text': ea['sentences'][i],
                        'label': ea_classified_label,
                    })
                if len(reused_detail['reused_by_aic']) > 0:
                    analyzed['ea_reused_details'].append(reused_detail)
            
            # By commission
            results['aic_reused_paragraphs'] = aic_marked_reused.count(True)
            results['aic_reused_percent'] = results['aic_reused_paragraphs'] / results['aic_total_paragraphs']
            by_commission_max_reused_percent = 0
            by_commission_max_reused_label = None
            if results['aic_reused_paragraphs'] > 0:
                for label in consts.Labels:
                    results['by_commission']['aic_reused_label_percent'][label] = results['by_commission']['aic_reused_label'][label] / results['aic_reused_paragraphs']
                    if results['by_commission']['aic_reused_label_percent'][label] > by_commission_max_reused_percent:
                        by_commission_max_reused_percent = results['by_commission']['aic_reused_label_percent'][label]
                        by_commission_max_reused_label = label
            is_by_commission = ''
            if results['aic_reused_percent'] > commission_bias_threshold and by_commission_max_reused_label != "CENTER" and by_commission_max_reused_label != None:
                is_by_commission = f'Yes, to the {by_commission_max_reused_label} {"{:0.2%}".format(by_commission_max_reused_percent)}'
            results['by_commission']['verdict'] = is_by_commission

            # By omission           
            if analyzed['ea_non_reused_ratio'] > omission_bias_threshold:
                highestOmittedLabel = None
                highestOmittedPercent = 0
                for label in consts.Labels:
                    if analyzed['ea_non_reused_label_percent'][label] > highestOmittedPercent:
                        highestOmittedPercent = analyzed['ea_non_reused_label_percent'][label]
                        highestOmittedLabel = label
                if highestOmittedLabel == "CENTER":
                    results['by_omission']['omitted_articles'].append(analyzed['ea_id'])
                    results['by_omission']['aic_non_reused_label'][analyzed['ea_label']] += 1            
            
            # By source selection
            if analyzed['ea_total_reused_paragraphs'] > 0:
                for label in consts.Labels:
                    analyzed['ea_reused_label_percent'][label] = analyzed['ea_reused_label'][label] / analyzed['ea_total_reused_paragraphs']
            is_by_source_selection = ''
            analyzed['ea_reused_ratio'] = round(analyzed['ea_total_reused_paragraphs'] / analyzed['ea_total_paragraphs'], 2)
            if analyzed['ea_reused_ratio'] > reused_threshold: 
                highest_percent = 0
                highest_label = None      
                for label in consts.Labels:
                    if analyzed['ea_reused_label_percent'][label] > highest_percent:
                        highest_percent = analyzed['ea_reused_label_percent'][label]
                        highest_label = label
                if highest_label != 'CENTER':
                    is_by_source_selection = f'Yes, to the {highest_label}. Percentage: {"{:0.2%}".format(highest_percent)}'
                analyzed['ea_by_source_selection'] = is_by_source_selection
                results['by_source_selection']['committed_articles'].append(ea['id'])
                results['by_source_selection']['biased_source_label'][highest_label] += 1
            
            results['earlier_articles'].append(analyzed)

        if is_earliest:
            results['is_biased'] = 'This is the earliest article'
            results['is_earliest'] = True

        # By omission conclusion
        is_by_omission = "No"
        if len(results['by_omission']['omitted_articles']) > 0:
            is_by_omission = "Yes"
            for label in consts.Labels:
                results['by_omission']['aic_non_reused_label_percent'][label] = round(results['by_omission']['aic_non_reused_label'][label] / len(results['by_omission']['omitted_articles']), 2)
        results['by_omission']['verdict'] = is_by_omission
        
        # By source selection conclusion
        if len(results['by_source_selection']['committed_articles']) > 0:
            max_l = 0
            max_label = None
            for label in consts.Labels:
                if results['by_source_selection']['biased_source_label'][label] > max_l:
                    max_l = results['by_source_selection']['biased_source_label'][label] 
                    max_label = label
            if max_label != 'CENTER':
                results['by_source_selection']['verdict'] = f"Yes, to the {max_label}"

            #
            # Prepare edges for Directed graph
            #
            for ea_id in results['by_source_selection']['committed_articles']:
                edges.append((str(ea_id), str(aic_id)))
                topic_info["graph"]["edges"].append({
                    "from": int(ea_id),
                    "to": int(aic_id),
                    "value": analyzed['ea_reused_ratio'], 
                    "label": f'{"{:0.2%}".format(analyzed["ea_reused_ratio"])}',
                })
        results_folder = folder if folder != None else f'./{FOLDER}/by_coss' 
        Path(results_folder).mkdir(parents=True, exist_ok=True)
        # datetime: _{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        results_filename = f"./{results_folder}/by_coss_{DATASET}_of_article_{aic_id}.json"
        chart_filename = f"./{results_folder}/by_coss_{DATASET}_of_article_{aic_id}.png"
        topic_info['articles'].append({
            "article_id": aic_id,
            "article_title": f'{aic_title} + ({aic_label})',
            "analyzed": results_filename,
            "chart": chart_filename if (len(results['by_source_selection']['committed_articles']) > 0) else ''
        })
        io.write_json(results_filename, results)
        if len(results['by_source_selection']['committed_articles']) > 0:
            # STACKED BAR CHART
            build_chart(results, chart_filename)
            
    # #
    # # BUILD DIRECTED GRAPH
    # #
    if len(node_list) and len(edges):
        graph_filename = f"./{results_folder}/by_coss_{DATASET}_network_topic_{topic_name}.png"
        graph_title = f'Plagiarism map among articles in topic {topic_name}'
        topic_info['network'] = graph_filename
        build_graph(node_list=node_list, edges=edges, filename=graph_filename, graph_title=graph_title)
    
    return topic_info

def build_chart(results, filename):
    plt.figure()
    
    L = []
    R = []
    C = []
    ea_ids = []
    for ea in results['earlier_articles']:
        if ea['ea_by_source_selection'] != 'No':
            L.append(ea['ea_reused_label']['LEFT'])
            C.append(ea['ea_reused_label']['CENTER'])
            R.append(ea['ea_reused_label']['RIGHT'])
            ea_ids.append(f'{ea["ea_id"]}\n{ea["ea_label"]}')

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
    ax.set_title(f'Number of paragraphs resued by article {results["aic_id"]} ({results["aic_label"]}) and their classified polarity')
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
        file_info = analyze_coss(
            df=df, 
            folder=f'./{FOLDER}/by_coss',
            features_collection=features, 
            polarity_classifier=classifier, 
            paraphrase_threshold=consts.paraphrase_threshold,
            topic_name=topic_id            
        )

        files_info.append(file_info)

    folder = f'./{FOLDER}/by_coss'
    io.write_json(f"./{folder}/files_info.json", files_info)

    if consts.openShell:
        f.showToast("Media Bias by COSS - Main")
        f.openShell()
        