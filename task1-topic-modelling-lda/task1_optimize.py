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
def get_coherence_score(model, data_dir=DATA_FOLDER, out_dir=OUTPUT_FOLDER, coherence='c_v'):

    texts = TokenGenerator(data_dir=data_dir)
    if coherence in ['c_v', 'c_uci', 'c_npmi']:
        coherence_model = CoherenceModel(
            model=model,
            texts=texts,
            coherence=coherence
        )
    elif coherence == 'u_mass':
        # u_mass
        dictionary = corpora.Dictionary.load(f'./output/{out_dir}/Task1_dictionary_full.gensim')
        # dictionary = Dictionary().build(data_dir=data_dir, out_dir=out_dir)
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

def find_best_model(models, data_dir=DATA_FOLDER, out_dir=OUTPUT_FOLDER, coherence='c_v'):
    best_score = 0.0
    best_model = None
    results = []
    for model in models:
        print(f'Calculating coherence score for {model[0]} topics model')
        score = get_coherence_score(model[1], data_dir, out_dir, coherence)
        if coherence in ['u_mass', 'c_uci']:
            score *= -1
        if score > best_score:
            best_score = score
            best_model = model
        results.append((model[0], score))
        print(f'Model: {model}\n Score: {score}')
    pd.DataFrame(results).to_csv(f'coherence_scores_{model[0]}.csv', index=False)
    return best_model, best_score

def optimize(files, data_dir=DATA_FOLDER, out_dir=OUTPUT_FOLDER, coherence='c_v'):
    models = []
    for file in files:
        models.append((file[0], gensim.models.LdaModel.load(file[1])))

    return find_best_model(models, data_dir, out_dir, coherence)

if __name__ == "__main__":
    import argparse

    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')

    # Required positional argument
    parser.add_argument('--num-topics', type=int, nargs='*', default=20,
                        help='Number of desired topics')

    # Optional argument
    parser.add_argument('--coherence', type=str, nargs='?', const=f'c_v', default='c_v',
                        help='Number of times to pass through whole corpus')

    # Optional argument
    parser.add_argument('--data-dir', type=str, nargs='?', const=f'{OUTPUT_FOLDER}', default='polusa_balanced',
                        help='Number of times to pass through whole corpus')

    # Optional argument
    parser.add_argument('--out-dir', type=str, nargs='?', const=f'{DATA_FOLDER}', default='polusa_balanced',
                        help='Number of times to pass through whole corpus')

    args = parser.parse_args()

    print(f"""Calculating coherence score:
    Num of topics: {args.num_topics}
    Type of coherence score: {args.coherence}
    Data directory location: {args.data_dir}
    Output directory: {args.out_dir}""")
    num_topics = args.num_topics

    files = [(ntopic, f'./output/{args.out_dir}/Task1_ldamodel_full_{ntopic}_topics.gensim') for ntopic in num_topics]

    best_model, best_score = optimize(files, args.data_dir, args.out_dir, args.coherence)
    print(f'Best score: {best_score}\nBest model: {best_model}')
