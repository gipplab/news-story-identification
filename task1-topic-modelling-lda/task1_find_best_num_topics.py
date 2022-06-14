#https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
from gensim.models import CoherenceModel
from gensim import corpora
from corpus import Corpus
from dictionary import Dictionary
from data_generator import DataGenerator
from tokens import TokenGenerator
import gensim
from utils import OUTPUT_FOLDER, DATA_FOLDER
import pandas as pd
from gensim.topic_coherence import direct_confirmation_measure
from my_custom_module import custom_log_ratio_measure
# https://github.com/RaRe-Technologies/gensim/issues/3040
# direct_confirmation_measure.log_ratio_measure = custom_log_ratio_measure
def get_coherence_score(model, num_topic=0, data_dir=DATA_FOLDER, out_dir=OUTPUT_FOLDER, coherence='c_v'):
    if num_topics == 0:
        print(f'Num of topics in invalid (0). Exit')
        return 0
    texts = TokenGenerator(data_dir=data_dir)
    if coherence in ['c_v', 'c_uci', 'c_npmi']:
        coherence_model = CoherenceModel(
            model=model,
            texts=texts,
            coherence=coherence
        )
    elif coherence == 'u_mass':
        # u_mass
        # dictionary = corpora.Dictionary.load(f'./output/{out_dir}/Task1_dictionary_full.gensim')
        dictionary = Dictionary().build(data_dir=data_dir, out_dir=out_dir)
        corpus = Corpus(dictionary=dictionary, data_dir=data_dir)
        # corpus = DataGenerator(data_dir)
        coherence_model = CoherenceModel(
            model=model,
            dictionary=dictionary,
            corpus=corpus,
            coherence=coherence
        )
    else:
        raise Exception(f'Unknown {coherence}')
    return coherence_model.get_coherence()

def optimize(files, num_topics=[], data_dir=DATA_FOLDER, out_dir=OUTPUT_FOLDER, coherence='c_v'):
    if len(num_topics) == 0:
        print(f'Num of topics to be evaluated is invalid (EMPTY). Exiting...')
        return None, None
    best_score = 0.0
    best_num_topic = 1
    results = {coherence: []}
    for i, file in enumerate(files):
        model = gensim.models.LdaModel.load(file[1])
        print(f'==== Calculating coherence score for model with #{num_topics[i]} topics... ====')        
        score = get_coherence_score(model=model, num_topic=num_topics[i], data_dir=data_dir, out_dir=out_dir, coherence=coherence)
        print(f'==== {i+1} topics yields score {score} ====')
        if score < 0:
            score *= -1
        if score > best_score:
            best_score = score
            best_num_topic = num_topics[i]
        results[coherence].append((best_num_topic, score))
    print(f'..:: Done. Best performing model with topic num #{best_num_topic} with score {best_score} ::..')
    pd.DataFrame(results).to_csv(f'./output/{out_dir}/scores_{out_dir}_{coherence}.csv', index=False)
    return best_num_topic, best_score

if __name__ == "__main__":
    import argparse

    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')

    # Required positional argument
    parser.add_argument('--num-topics', type=int, nargs='+', default=20,
                        help='Number of desired topics')

    # Optional argument
    parser.add_argument('--coherence', type=str, nargs='?', const=f'c_v', default='c_v',
                        help='Number of times to pass through whole corpus')

    # Optional argument
    parser.add_argument('--dir', type=str, nargs='?', const=f'{OUTPUT_FOLDER}', default='polusa_balanced',
                        help='Number of times to pass through whole corpus')

    # Optional argument
    parser.add_argument('--data-dir', type=str, nargs='?', const=f'{OUTPUT_FOLDER}', default='polusa_balanced',
                        help='Number of times to pass through whole corpus')

    # Optional argument
    parser.add_argument('--out-dir', type=str, nargs='?', const=f'{DATA_FOLDER}', default='polusa_balanced',
                        help='Number of times to pass through whole corpus')

    args = parser.parse_args()

    print(f"""Calculating coherence score(s):
    Num of topics: {args.num_topics}
    Type of coherence measure: {args.coherence}
    Data directory location: {args.dir}
    Output directory location: {args.dir}""")
    num_topics = args.num_topics

    files = [(ntopic, f'./output/{args.dir}/Task1_ldamodel_full_{ntopic}_topics.gensim') for ntopic in num_topics]

    best_num_topic, best_score = optimize(files=files, num_topics=num_topics, data_dir=args.dir, out_dir=args.dir, coherence=args.coherence)
    print(f'=== Best score: {best_score}\n=== Best num topic: {best_num_topic}')
