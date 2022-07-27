from datetime import datetime
from bertopic import BERTopic
from os.path import join, exists

from utils import list_files, pickle_save, pickle_load
from gensim.models.coherencemodel import CoherenceModel

import gensim.corpora as corpora
import json
import pandas as pd

def preprocess(input_dir, files):
    if len(files) == 0:
        raise Exception(f'[PREPROCESSING] No files provided. Exit')
    docs = []
    doc_ids = []
    for file in files:
        df = pd.read_csv(join(input_dir, file))
        for i, item in df.iterrows():
            docs.append(item['body'])
            doc_ids.append(item['id'])
    return docs, doc_ids

def postprocess(output_dir, docs, doc_ids, coherence_type='c_v', model_components=None):
    if not model_components:
        raise Exception(f'[POSTPROCESSING] No model_components provided. Exit')

    (topic_model, topics, probs) = model_components
    pickle_save(topic_model, output_dir, '_cache_sbert_topic_model.pkl')
    pickle_save(topics, output_dir, '_cache_sbert_topics.pkl')
    pickle_save(probs, output_dir, '_cache_sbert_probs.pkl')
    
    coherence_score = evaluate(model=topic_model, docs=docs, topics=topics, coherence=coherence_type)

    save_report(output_dir, topic_model, docs, doc_ids, coherence_score, coherence_type)

def start_bertopic(input_dir, output_dir, embedding_model_name='', coherence_type='c_v'):
    files = list_files(input_dir)

    docs, doc_ids = preprocess(input_dir, files)
    
    if exists(join(output_dir, '_cache_sbert_topic_model.pkl')) and exists(join(output_dir, '_cache_sbert_topics.pkl')) and exists(join(output_dir, '_cache_sbert_probs.pkl')):
        print(f'[LEARNING] Load model from cache...')
        topic_model = pickle_load(output_dir, '_cache_sbert_topic_model.pkl')
        print(f'[LEARNING] Load topics from cache...')
        topics = pickle_load(output_dir, '_cache_sbert_topics.pkl')
        print(f'[LEARNING] Load probs from cache...')
        probs = pickle_load(output_dir, '_cache_sbert_probs.pkl')
    else:
        print(f'[LEARNING] Fitting Bertopic with corpus...')
        if len(embedding_model_name) > 0:
            topic_model = BERTopic(embedding_model=embedding_model_name)
        else:
            topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs)
    
    postprocess(output_dir=output_dir, docs=docs, doc_ids=doc_ids, coherence_type=coherence_type, model_components=(topic_model, topics, probs))


def evaluate(model, docs, topics, coherence='c_v'):
    # Preprocess Documents
    documents = pd.DataFrame({"Document": docs,
                            "ID": range(len(docs)),
                            "Topic": topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = model._preprocess_text(documents_per_topic.Document.values)

    # Extract vectorizer and analyzer from BERTopic
    vectorizer = model.vectorizer_model
    analyzer = vectorizer.build_analyzer()

    # Extract features for Topic Coherence evaluation
    words = vectorizer.get_feature_names()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [[words for words, _ in model.get_topic(topic)] 
                for topic in range(len(set(topics))-1)]

    # Evaluate
    coherence_model = CoherenceModel(topics=topic_words, 
                                    texts=tokens, 
                                    corpus=corpus,
                                    dictionary=dictionary, 
                                    coherence=coherence)
    coherence = coherence_model.get_coherence()
    print(f'[EVALUATING] Bertopic coherence score: {coherence}')

    return coherence

def save_report(output_dir, model, docs, doc_ids, coherence_score, coherence_type):
    print(f'[POSTPROCESSING] Generating reports...')
    reports = {}
    topics = model.get_topics()
    if not exists(join(output_dir, 'bertopic_report_full.json')):        
        print(f'[POSTPROCESSING] Mapping topics...')
        
        for i, topic_id in enumerate(topics):
            reports[topic_id] = {
                'topic_id': topic_id,
                'keywords': [probs[0] for probs in topics[topic_id]],
                'document_ids': []
            }
        for i, doc in enumerate(docs):
            potential_topics = model.find_topics(doc)
            reports[potential_topics[0][0]]['document_ids'].append(doc_ids[i])

        with open(join(output_dir, 'bertopic_report_full.json'), 'w') as f:
            json.dump(reports, f, indent=4)
        f.close()        
    else:
        print(f'[POSTPROCESSING] Topics already mapped. Skipping this step.')
        with open(join(output_dir, 'bertopic_report_full.json'), 'r') as f:
            reports = json.load(f)
        f.close()

    with open(f'{output_dir}bertopic_report_short.txt', 'w', encoding='utf-8') as f:
        f.write(f"""
            BERTopic run on {datetime.now()}
            Num of topics generated: {len(topics)}
            Coherence score: {coherence_score}
            Coherence type: {coherence_type}
        """)
        f.close()
    return reports