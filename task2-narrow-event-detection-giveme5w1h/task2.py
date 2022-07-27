import sys, json, argparse, pickle
import pandas as pd
from os.path import exists, isfile, join, isdir
from os import listdir
from datetime import datetime
from ECB import ECBDocument
from utils import build_giveme5w1h_training_dataset
from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor
from Giveme5W1H.extractor.preprocessors.preprocessor_core_nlp import Preprocessor
from gensim.models.tfidfmodel import TfidfModel
from gensim.corpora.dictionary import Dictionary
from gensim.matutils import corpus2dense
from transformers import BertTokenizer
from sklearn.cluster import KMeans

def pickle_save(object, file_path): 
    with open(file_path, 'wb') as f:
        pickle.dump(object, f)
    f.close()

def pickle_load(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    f.close()
    return obj

###   Convert ECB+ dataset into Giveme5W1H-XML format for training
def ecbplus_conversion():
    dataPath = 'data/ECB+'
    folders = listdir(dataPath)
    docs = []
    convert_to_goldenstandard_format = True
    for subFolder in folders:
        if isdir(f'{dataPath}/{subFolder}'):
            items = listdir(f'{dataPath}/{subFolder}')
            for file in items:
                # print(f'./{dataPath}/{subFolder}/{file}')

                doc = ECBDocument()
                doc.read(f'./{dataPath}/{subFolder}/{file}')
                docs.append(doc)

                if convert_to_goldenstandard_format:
                    build_giveme5w1h_training_dataset(doc, outputFolder="data/ECBplus_giveme5w1h")
                
    print(f'{len(docs)} documents read')

def load_documents(input_dir=None):
    if isinstance(input_dir, str) == False:
        raise Exception(f'Input directory {input_dir} is wrong')
    documents = {}
    print(f'[PREPROCESSING] Loading documents...')
    files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
    for file in files:
        if file.endswith('.csv'):         
            df = pd.read_csv(f'{input_dir}{file}', encoding='utf-8')              
            for i, row in df.iterrows():   
                doc_id = row['id']             
                #documents[row['id']] = row['body']
                documents[doc_id] = row['body']
                # doc_id = doc_id + 1
        elif file.endswith('.txt'):
            print(f'TODO read .txt files')    
        
    return documents
    
def extract_4ws(documents, input_dir, output_dir, topics=None):
    # use remote server
    # preprocessor = Preprocessor('http://164.92.173.99:9000')
    # extractor = MasterExtractor(preprocessor=preprocessor)    

    # use local server
    extractor = MasterExtractor()                               

    error_logs = []
    if topics is not None:
        for topic_i in topics:            
            if exists(f'{output_dir}topic_{topic_i}_4ws.json'):
                print(f'[EXTRACTING-4Ws] Already extracted. Skipped #{topic_i} with keywords: {topics[topic_i]["keywords"]}')
            else:
                print(f'[EXTRACTING-4Ws] Topic #{topic_i} with keywords: {topics[topic_i]["keywords"]}')
                doc_ids = topics[topic_i]['document_ids']
                report = {
                    'topic_id': topic_i,
                    'keywords': topics[topic_i]["keywords"],
                    'documents_4ws': []
                }
                for i, doc_id in enumerate(doc_ids):
                    # if int(topic_i) == 1 and int(doc_id) > 1500 and int(doc_id) < 1600:
                    print(f'({i+1}/{len(doc_ids)})[EXTRACTING-4Ws] Topic #{topic_i}, document #{doc_id}: ', end='')
                    doc_body = documents[doc_id]
                    # print(doc_body)
                    try:
                        doc = Document.from_text(doc_body)
                        print(f'[EXTRACTING-4Ws] from_text: {doc}')
                        doc = extractor.parse(doc)
                        print(f'[EXTRACTING-4Ws] parse: {doc}')
                        mentions = {}                        
                        try:
                            mentions['who'] = doc.get_top_answer('who').get_parts_as_text()
                        except IndexError:
                            mentions['who'] = 'undefined'
                        try:
                            mentions['what'] = doc.get_top_answer('what').get_parts_as_text()
                        except IndexError:
                            mentions['what'] = 'undefined'
                        try:
                            mentions['when'] = doc.get_top_answer('when').get_parts_as_text()
                        except IndexError:
                            mentions['when'] = 'undefined'
                        try:
                            mentions['where'] = doc.get_top_answer('where').get_parts_as_text()
                        except IndexError:
                            mentions['where'] = 'undefined'
                        report['documents_4ws'].append({ doc_id: mentions})
                        topics[topic_i]['mentions'] = { doc_id: mentions }
                        print(mentions)

                    except Exception as e:
                        err_msg = f'[ERR] {datetime.now()} Topic #{topic_i}\n\t\tdocument {doc_id}n\t\tdoc_body: {doc_body}\n\t\terr_msg: {str(e)}'
                        print(err_msg)
                        error_logs.append(err_msg)
                        with open(f'{output_dir}_error_logs.txt', 'a') as f:
                            f.write(err_msg)
                        f.close()

                with open(f'{output_dir}topic_{topic_i}_4ws.json', 'w') as fp:
                    json.dump(report, fp, indent=4)
                fp.close()
    else:
        raise Exception('[EXTRACTING] No topic-clusters provided. Exit')
# source: https://towardsdatascience.com/an-approach-for-choosing-number-of-clusters-for-k-means-c28e614ecb2c
def scaled_inertia(inertia_1, inertia, k, alpha):
    return (inertia / inertia_1) + (alpha * k)

def clustering_4ws(documents, input_dir, output_dir, topics=None):
    tz = BertTokenizer.from_pretrained("bert-base-cased")
    topic_4ws_files = [f for f in listdir(output_dir) if f.startswith('topic_') and f.endswith('_4ws.json')]
    for i, file in enumerate(topic_4ws_files):        
        print(f'[CLUSTERING] {"{0:0.2f}".format(i / len(topic_4ws_files) * 100)}% ({i}/{len(topic_4ws_files)}) clustering 4Ws from {file}... - ', end='')        
        with open(f'{output_dir}{file}', 'r', encoding='utf-8') as f:
            topic_4ws = json.load(f)
        f.close()
        # if exists(f'{output_dir}topic_{topic_4ws["topic_id"]}_4ws_clustered.json'):
        #     print(f'[CLUSTERING] {"{0:0.2f}".format(i / len(topic_4ws_files) * 100)}% ({i}/{len(topic_4ws_files)}) {file}- Already clustered. Skipped!')   
        #     continue 
        doc_ids = [id for doc in topic_4ws['documents_4ws'] for id in doc]
        entities_4ws = [doc[id] for doc in topic_4ws['documents_4ws'] for id in doc]
        best_score = 9 * 10**7
        best_clusters = []
        best_num_clusters = 0
        if len(doc_ids) > 0:
            corpus = [documents[int(id)] for id in doc_ids]
            corpus = [tz.tokenize(c) for c in corpus]
            dictionary = Dictionary(corpus)
            corpus_bow = [dictionary.doc2bow(doc) for doc in corpus]
            tfidf_model = TfidfModel(corpus_bow)
            corpus_tfidf = tfidf_model[corpus_bow]
            num_docs = dictionary.num_docs
            num_terms = len(dictionary.keys())
            corpus_tfidf_dense = corpus2dense(corpus_tfidf, num_terms, num_docs)

            kmean_model_1 = KMeans(init="k-means++", n_clusters=1)
            clusters = kmean_model_1.fit_predict(corpus_tfidf_dense.T)
            base_inertia = kmean_model_1.inertia_

            for num_clusters in range(2, len(corpus) + 1):
                # print(num_clusters, ' ', end='')
                kmean_model = KMeans(init="k-means++", n_clusters=num_clusters)
                clusters = kmean_model.fit_predict(corpus_tfidf_dense.T)
                score = kmean_model.inertia_
                scaled_score = scaled_inertia(inertia_1=base_inertia, inertia=score, k=num_clusters, alpha=0.02)

                if scaled_score < best_score:
                    best_score = scaled_score
                    best_clusters = clusters 
                    best_num_clusters = num_clusters

        print(f'best: n_clusters: {best_num_clusters}, score: {best_score}')
        report = {
            'topic_id': topic_4ws['topic_id'],
            'keywords': topic_4ws['keywords'],
            'num_of_clusters': best_num_clusters,
            'score': best_score,
            'clusters': {}
        }
        for i in range(best_num_clusters):
            report['clusters'][i] = []
        for i, cluster_id in enumerate(best_clusters):
            report['clusters'][cluster_id].append((doc_ids[i], entities_4ws[i]))

        with open(f'{output_dir}topic_{topic_4ws["topic_id"]}_4ws_clustered.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4)
        f.close()

##  4W-Questions Extraction Pipeline
def process(input_dir, output_dir, weights=None, topics=None):
    documents = load_documents(input_dir)
    
    #   Part 1 - extract 4Ws entites with Giveme5W1H
    #   TODO Uncomment
    extract_4ws(documents=documents, input_dir=input_dir, output_dir=output_dir)
    
    #   Part 2 - cluster 4Ws entities
    clustering_4ws(documents=documents, input_dir=input_dir, output_dir=output_dir, topics=topics)

def load_best_weights(path):
    with open(path, encoding='utf-8') as data_file:
        data = json.load(data_file)
    data_file.close()
    return data['best_dist']['weights']
    
def load_weights(input_dir=None):
    if isinstance(input_dir, str) == True:                
        if exists(f'{input_dir}weights_who.json') and exists(f'{input_dir}weights_what.json') and exists(f'{input_dir}weights_when.json') and exists(f'{input_dir}weights_where.json'):
            weights = {}    
            weights['who'] = load_best_weights(f'{input_dir}weights_who.json')
            weights['what'] = load_best_weights(f'{input_dir}weights_what.json')
            weights['when'] = load_best_weights(f'{input_dir}weights_when.json')
            weights['where'] = load_best_weights(f'{input_dir}weights_where.json')    
            return weights
    return None

def load_topics(input_dir=None):
    if isinstance(input_dir, str) == True:           
        if exists(f'{input_dir}topics_report.json'):
            with open(f'{input_dir}topics_report.json', 'r') as fp:
                topics = json.load(fp)
            fp.close()
            return topics
    return None
    
if __name__ == "__main__":
    if len(sys.argv) == 3:
        beginning = datetime.now()

        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        if input_dir[-1] != "/":
            input_dir+="/"
        if output_dir[-1] != "/":
            output_dir+="/"

        weights = load_weights(input_dir)        
        topics = load_topics(input_dir)    
        
        process(input_dir, output_dir, weights, topics)
        
        print(f"=== DONE ! Total times for Task 2 is {datetime.now() - beginning}")
    else:
        print('\n'.join(["Unexpected number of commandline arguments.",
                         "Usage: python ./task2.py {input-dir} {output-dir}",
                         "Optional: --weights {weight-dir}"]))