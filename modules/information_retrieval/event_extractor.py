import sys, json, pickle
import pandas as pd
from os.path import exists, isfile, join, isdir
from os import listdir, makedirs
from datetime import datetime
from information_retrieval.giveme5w1h.ecb.ECB import ECBDocument
from clustering.cluster import Cluster
from word_embedding.embedder import Embedder
from information_retrieval.giveme5w1h.ecb.ecb_converter import convert
from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor
from Giveme5W1H.extractor.preprocessors.preprocessor_core_nlp import Preprocessor
from base.cache import has_cache
from geopy.exc import GeocoderTimedOut
from preprocessing.tokenizer import tokenize

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
                doc = ECBDocument()
                doc.read(f'./{dataPath}/{subFolder}/{file}')
                docs.append(doc)

                if convert_to_goldenstandard_format:
                    convert(doc, outputFolder="data/ECBplus_giveme5w1h")
                
    print(f'{len(docs)} documents read')

class EventExtractor():
    def __init__(self, config):
        self.config = config
        self.input_dir = config['GLOBAL']['InputFolder']
        self.output_dir = config['GLOBAL']['OutputFolder']
        self.report_full_filename = config['TASK-1']['ReportFullFilename']
        self.corenlp_addr = config['TASK-2']['CORENLP_ADDR']
        self.corenlp_port = config['TASK-2']['CORENLP_PORT']

    def extract_4ws(self, documents, topics):
        # use remote server
        if len(self.corenlp_addr):
            preprocessor = Preprocessor(self.corenlp_addr)
            extractor = MasterExtractor(preprocessor=preprocessor)    

        # use local server
        extractor = MasterExtractor()                               

        error_logs = []
        if topics is not None:
            for topic_i in topics:            
                if exists(f'{self.output_dir}topic_{topic_i}_4ws.json'):
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
                        print(f'({i+1}/{len(doc_ids)})[EXTRACTING-4Ws] Topic #{topic_i}, document #{doc_id}: ', end='')
                        doc_body = documents[doc_id]

                        try:
                            doc = Document.from_text(doc_body)
                            print(f'[EXTRACTING-4Ws] from_text: {doc}')
                            doc = extractor.parse(doc)
                            print(f'[EXTRACTING-4Ws] parse: {doc}')
                            mentions = {}                        
                            try:
                                mentions['who'] = doc.get_top_answer('who').get_parts_as_text()
                            except IndexError:
                                mentions['who'] = ''
                            try:
                                mentions['what'] = doc.get_top_answer('what').get_parts_as_text()
                            except IndexError:
                                mentions['what'] = ''
                            try:
                                mentions['when'] = doc.get_top_answer('when').get_parts_as_text()
                            except IndexError:
                                mentions['when'] = ''
                            try:
                                mentions['where'] = doc.get_top_answer('where').get_parts_as_text()
                            except IndexError:
                                mentions['where'] = ''
                            report['documents_4ws'].append({ doc_id: mentions})
                            topics[topic_i]['mentions'] = { doc_id: mentions }
                            print(mentions)
                        except GeocoderTimedOut as e:
                            err_msg = f'[ERR] {datetime.now()} Topic #{topic_i}\n\t\tdocument {doc_id}n\t\tdoc_body: {doc_body}\n\t\terr_msg: {str(e)}'
                            print(err_msg)
                            error_logs.append(err_msg)
                            with open(f'{self.output_dir}_error_logs.txt', 'a') as f:
                                f.write(err_msg)
                            f.close()
                        except Exception as e:
                            err_msg = f'[ERR] {datetime.now()} Topic #{topic_i}\n\t\tdocument {doc_id}n\t\tdoc_body: {doc_body}\n\t\terr_msg: {str(e)}'
                            print(err_msg)
                            error_logs.append(err_msg)
                            with open(f'{self.output_dir}_error_logs.txt', 'a') as f:
                                f.write(err_msg)
                            f.close()
                    with open(f'{self.output_dir}topic_{topic_i}_4ws.json', 'w') as fp:
                        json.dump(report, fp, indent=4)
                    fp.close()
        else:
            raise Exception('[EXTRACTING] No topic-clusters provided. Exit')
        
    # source: https://towardsdatascience.com/an-approach-for-choosing-number-of-clusters-for-k-means-c28e614ecb2c
    def scaled_inertia(self, inertia_1, inertia, k, alpha):
        return (inertia / inertia_1) + (alpha * k)

    def clustering_4ws(self, documents, cluster_method='agglomerative', embedder_method='tfidf', topics=None):
        topic_4ws_files = [f for f in listdir(self.output_dir) if f.startswith('topic_') and f.endswith('_4ws.json')]
        out_dir = f'{self.output_dir}{embedder_method}_{cluster_method}/'
        if not has_cache([out_dir]):
            makedirs(out_dir)
        for i, file in enumerate(topic_4ws_files):     
            print(f'[CLUSTERING] {"{0:0.2f}".format(i / len(topic_4ws_files) * 100)}% ({i}/{len(topic_4ws_files)}) clustering 4Ws from {file}... - ', end='')        
            with open(f'{self.output_dir}{file}', 'r', encoding='utf-8') as f:
                topic_4ws = json.load(f)
            f.close()
            
            if exists(f'{out_dir}topic_{topic_4ws["topic_id"]}_4ws_clustered.json'):
                print(f'[CLUSTERING] {"{0:0.2f}".format(i / len(topic_4ws_files) * 100)}% ({i}/{len(topic_4ws_files)}) {file}- Already clustered. Skipped!')   
                continue 
            
            doc_ids = [id for doc in topic_4ws['documents_4ws'] for id in doc]
            entities_4ws = [doc[id] for doc in topic_4ws['documents_4ws'] for id in doc]
            corpus = [tokenize(documents[int(id)]) for doc in topic_4ws['documents_4ws'] for id in doc]
            n_sample = len(corpus)
            feasible_corpus = [c for i, c in enumerate(corpus) if int(len(entities_4ws[i]['who']) > 0) + int(len(entities_4ws[i]['what']) > 0) + int(len(entities_4ws[i]['when']) > 0) + int(len(entities_4ws[i]['where']) > 0) >= 1]
            
            if len(feasible_corpus) > 3:
                emd = Embedder(input_dir=self.input_dir, output_dir=self.output_dir, method=embedder_method, use_cache=False)
                emd.process(feasible_corpus)
                clt = Cluster(input_dir=self.input_dir, output_dir=out_dir, corpus=feasible_corpus, dist_matrix=emd.distance_matrix, method=cluster_method, use_cache=False)
                clt.process()

                report = {
                    'topic_id': topic_4ws['topic_id'],
                    'keywords': topic_4ws['keywords'],
                    'num_of_clusters': clt.num_clusters,
                    'num_of_documents': n_sample,
                    'num_of_feasible_documents': len(feasible_corpus),
                    'score': str(clt.score),
                    'clusters': {}
                }  
                if clt.num_clusters:
                    for k in range(0, clt.num_clusters):
                        report['clusters'][k] = []
                    for i, label in enumerate(clt.clusters.labels_):
                        report['clusters'][label].append({
                            "doc_id": doc_ids[i], 
                            "4W": entities_4ws[i]
                        })
                
                with open(f'{out_dir}topic_{topic_4ws["topic_id"]}_4ws_clustered.json', 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=4)
                f.close()
                
                print(f'score: {"{0:0.4f}".format(clt.score)}, num_clusters: {clt.num_clusters}, num_samples: {n_sample}')   
                # clt.save_plot(join(out_dir, f'topic_{topic_4ws["topic_id"]}.png'))
                # df = pd.DataFrame(list(zip(doc_ids, clusters.labels_, entities_4ws)), columns=columns)
                # df.to_csv(join(out_dir, f'topic_{topic_4ws["topic_id"]}.csv'))
            else:
                print('not enough feasible documents. Skipped')

    ##  4W-Questions Extraction Pipeline
    def start_extracting(self):
        weights = self.load_weights()    
        topics = self.load_topics()       
        documents = self.load_documents()
        
        #   Part 1 - extract 4Ws entites with Giveme5W1H
        self.extract_4ws(documents=documents, topics=topics)
        
    def start_clustering(self):
        topics = self.load_topics()
        documents = self.load_documents()

        #   Part 2 - cluster 4Ws entities
        self.clustering_4ws(documents=documents, topics=topics)

    def load_documents(self):
        if isinstance(self.input_dir, str) == False:
            raise Exception(f'Input directory {self.input_dir} is wrong')
        documents = {}
        print(f'[PREPROCESSING] Loading documents...')
        files = [f for f in listdir(self.input_dir) if isfile(join(self.input_dir, f))]
        for file in files:
            if file.endswith('.csv'):         
                df = pd.read_csv(f'{self.input_dir}{file}', encoding='utf-8')              
                for _, row in df.iterrows():   
                    doc_id = row['id']             
                    #documents[row['id']] = row['body']
                    documents[doc_id] = row['body']
                    # doc_id = doc_id + 1
            elif file.endswith('.txt'):
                print(f'TODO read .txt files')    
            
        return documents

    def load_best_weights(self, path):
        with open(path, encoding='utf-8') as data_file:
            data = json.load(data_file)
        data_file.close()
        return data['best_dist']['weights']
        
    def load_weights(self):
        if isinstance(self.input_dir, str) == True:                
            if exists(f'{self.input_dir}weights_who.json') and exists(f'{self.input_dir}weights_what.json') and exists(f'{self.input_dir}weights_when.json') and exists(f'{self.input_dir}weights_where.json'):
                weights = {}    
                weights['who'] = self.load_best_weights(f'{self.input_dir}weights_who.json')
                weights['what'] = self.load_best_weights(f'{self.input_dir}weights_what.json')
                weights['when'] = self.load_best_weights(f'{self.input_dir}weights_when.json')
                weights['where'] = self.load_best_weights(f'{self.input_dir}weights_where.json')    
                return weights
        return None

    def load_topics(self):
        if isinstance(self.input_dir, str) == True:           
            if exists(join(self.input_dir, self.report_full_filename)):
                with open(join(self.input_dir, self.report_full_filename), 'r') as fp:
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
        
        # start_extracting(input_dir, output_dir)

        # start_clustering(input_dir, output_dir)
        
        print(f"=== DONE ! Total times for Task 2 is {datetime.now() - beginning}")
    else:
        print('\n'.join(["Unexpected number of commandline arguments.",
                        "Usage: python ./task2.py {input-dir} {output-dir}",
                        "Optional: --weights {weight-dir}"]))