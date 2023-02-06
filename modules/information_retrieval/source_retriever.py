import codecs
from os.path import join, exists
from base.logger import Logger
from information_retrieval.google_search.google_api import search as google_search, download as gg_download
from information_retrieval.google_news.newsapi import search as newsapi_search, download as newsapi_download
from chatnoir_api.v1 import search as chatnoir_search
from chatnoir_api import html_contents, Index
from preprocessing.io import list_files, load_json, read_csv_to_dataframe, read_file, unicode2ascii, update_file, write_file
from information_retrieval.near_duplicate_detector import NearDuplicateEvaluator
from preprocessing.tokenizer import tokenize
from preprocessing.sentence_splitter import split
from config import Config
from tqdm import *
from preprocessing.tokenizer import clean_html

# Using SBert as text-alignment detector
# ========================

""" 
The following class implements a naive strategy to retrieve sources for a 
given suspicious document. It is merely intended as an example, not as a
serious solution to the problem.
"""

class SourceRetriever(Logger):
    """
        :param chatnoir_token  : str    (required)  : API token string for access the ChatNoir search engine
        :param threshold       : number (optional)  : similarity threshold for near duplicate detector
        :output_dir            : string (optional)  : output directory
    """
    def __init__(self, config: Config):
        self.config = config
        self.chatnoir_key = config['TASK-3']['ChatNoirApiKey']
        self.google_key = config['TASK-3']['GoogleSearchApiKey']
        self.google_cse = config['TASK-3']['GoogleSearchCSE']
        self.newsapi_key = config['TASK-3']['NewsApiKey']
        self.retrieve_method = self.config['TASK-3']['SourceRetrievalMethod']
        self.detector = NearDuplicateEvaluator(
            threshold=float(config['TASK-3']['NearDuplicateSimilarityThreshold']),
            threshold_length=float(config['TASK-3']['NearDuplicateSimilarityThresholdLength'])
        )
        self.input_dir = config['GLOBAL']['InputFolder']
        self.output_dir = config['GLOBAL']['OutputFolder']
        self.source_dir = config['TASK-3']['SourceFolder']
        self.source_downloaded_dir = config['TASK-3']['SourceDownloadedFolder']
        self.susp_dir = config['TASK-4']['SuspFolder']
        self.src_dir = config['TASK-4']['SrcFolder']
        self.pair_filename = 'pairs'
        self.report_filename = 'source_retrieved.json'
        self.progress_filename = 'source_cache_progress.pkl'
        self.use_cache = bool(config['GLOBAL']['Usecache'])
        self.report = None
        self.progress = None

    def start_retrieve(self):     
        if exists(join(self.output_dir, self.report_filename)):
            self.report = load_json(join(self.output_dir, self.report_filename))
        if exists(join(self.output_dir, self.progress_filename)):
            self.progress = load_json(join(self.output_dir, self.progress_filename))        
        files = list_files(self.input_dir, ext='csv')
        files = [join(self.input_dir, f) for f in files]
        df = read_csv_to_dataframe(files)
        if not self.report:
            self.report = {}
            self.progress = {}

            for _, row in df.iterrows():
                self.report[str(row['id'])] = {
                    'sources': [],
                }
                self.progress[str(row['id'])] = []   
            # TODO
        for _, row in tqdm(df.iterrows()):
            self.process(doc=row['body'], doc_id=row['id'])

        self.postprocess()

    """
        :param doc: str (required)    : document content in plain text
        :param doc_id: str (required)    : document id
    """
    def process(self, doc, doc_id: str):
        doc_id = str(doc_id)
        if len(self.progress[doc_id]) and not (False in self.progress[doc_id]):
            print(f'Skipped #{doc_id}. already processed')
        self.retrieve(text=doc, doc_id=doc_id)

    """
        :param text             : str (required)    : document as full string 
    """
    def retrieve(self, text, doc_id):        
        query = tokenize(text)
        queries = split(text)
        if len(self.progress[doc_id]) == 0:
            self.progress[doc_id] = [False for i in range(len(queries))]
        for i, query in tqdm(enumerate(queries)):  
            if self.use_cache and self.progress[doc_id][i]:
                print(f'Already processed. Skipped')
                continue

            if len(tokenize(query)) <= 3:
                print(f'#{doc_id}[{i}] Query too short. Skipped')
                self.progress[doc_id][i] = True
                update_file(join(self.output_dir, self.progress_filename), obj=self.progress, ext='json')
                continue

            try:
                results = self.pose_query(query=query)
                if len(results) == 0:
                    self.progress[doc_id][i] = True
                    update_file(join(self.output_dir, self.progress_filename), obj=self.progress, ext='json')
                    continue    # The query returned no results.

            except IndexError:
                print(f'IndexError. Skipped')
                continue
            except RuntimeError:
                print(f'RuntimeError. Skipped')
                continue
            self.progress[doc_id][i] = True # mark as processed
            download = clean_html(self.download_document(results))
            if self.retrieve_method == 'google':
                r = { 
                    "uuid": unicode2ascii(results[0]['title']),
                    "uri": results[0]['link'], 
                    "score": ""
                }
            elif self.retrieve_method == 'chatnoir':
                r = { 
                    "uuid": str(results[0].uuid), 
                    "uri": results[0].target_uri, 
                    "score": results[0].score
                }
            elif self.retrieve_method == 'newsapi':
                r = { 
                    "uuid": unicode2ascii(results[0]['title']), 
                    "uri": results[0]['url'], 
                    "score": ""
                }

            print(f'=== Evaluating {r["uri"]} with query `{query}`: ', end='')
            isSource = self.detector.evaluate(download, text)
            print(f'{isSource}')               

            if isSource:
                self.report[doc_id]['sources'].append(r)
                write_file(path=join(self.source_downloaded_dir, f'{r["uuid"]}.html'), obj=download)
            update_file(join(self.output_dir, self.progress_filename), obj=self.progress, ext='json')
            update_file(join(self.output_dir, self.report_filename), obj=self.report, ext='pkl')

        return self.report[doc_id]['sources']
    
    """
        Reads the file suspdoc and returns its text content.
        :param suspdoc          : str (required)    :        
    """
    def read_file(self, suspdoc):
        f = codecs.open(suspdoc, 'r', 'utf-8')
        text = f.read()
        f.close()
        return text

    """
        Poses the query to the ChatNoir search engine.
        :param query             : str (required)    :
    """
    def pose_query(self, query):
        # query = unicode2ascii(query)
        try:        
            if self.retrieve_method == 'google':
                results = google_search(query, self.google_key, self.google_cse)['items']
                return results                                  
            elif self.retrieve_method == 'chatnoir':
                print(f'Querying {len(tokenize(query))} tokens')
                results = chatnoir_search(self.chatnoir_key, query)
                return results
            elif self.retrieve_method == 'newsapi':
                if len(query.split(" ")) >= 10:
                    query = " ".join(query.split()[:10])
                results = newsapi_search(query, self.newsapi_key)['articles']
                return results
        except Exception as e:
            print(f'Exception occured {e}')
        return []

    def download_document(self, results, plain=False):
        downloaded = ""
        if self.retrieve_method == 'google':
            downloaded = gg_download(results[0]['link'])            
        elif self.retrieve_method == 'chatnoir':
            downloaded = html_contents(
                results[0].uuid,
                Index.CommonCrawl1511,
                plain=plain
            )
        elif self.retrieve_method == 'newsapi':
            downloaded = gg_download(results[0]['url']) 
        return downloaded

    """
        Generate input data for text-alignment
    """
    def postprocess(self):
        files = list_files(self.input_dir, ext='csv')
        files = [join(self.input_dir, f) for f in files]
        df = read_csv_to_dataframe(files)

        with open(join(self.output_dir, self.pair_filename), 'w', encoding='utf-8') as ffp:
            # write suspicious doc
            for _, row in tqdm(df.iterrows()):
                susp_file = join(self.susp_dir, f'{row["id"]}.txt')
                if not exists(susp_file):
                    susp_content = row['body']
                    write_file(susp_file, susp_content)

                # write source doc
                for source in self.report[str(row['id'])]['sources']:
                    src_file = join(self.src_dir, f'{source["uuid"]}.txt')
                    if not exists(src_file):
                        src_content = clean_html(read_file(join(self.source_downloaded_dir, f'{source["uuid"]}.html')))
                        write_file(src_file, src_content)

                    # write pair     
                    pair = f'{row["id"]}.txt {source["uuid"]}.txt\n'
                    ffp.write(pair)
        ffp.close()