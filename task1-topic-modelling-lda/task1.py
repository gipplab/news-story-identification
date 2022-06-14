#source https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21

# from utils import pipeline
from dictionary import Dictionary
from corpus import Corpus
from gensim import corpora
import gensim
from datetime import datetime
from utils import DATA_FOLDER, OUTPUT_FOLDER

def start(num_topics=100, chunk_size=10000, passes=0, data_dir=DATA_FOLDER, output_dir=OUTPUT_FOLDER):
    """
        Uncomment and run this line if to build dictionary from scratch
        Building dictionary on full POLUSA dataset takes couple of hours
    """
    print(f'==== Use one dictionary for multiple LDA model is not recommended ====')
    print(f'==== Building dictionary ===')
    start = datetime.now()
    dictionary =  Dictionary().build(data_dir=data_dir, out_dir=output_dir, filename=f'Task1_dictionary_{num_topics}_topics.gensim')
    # dictionary = corpora.Dictionary.load(f'./output/{output_dir}/Task1_dictionary_full.gensim')

    corpus = Corpus(dictionary=dictionary, data_dir=data_dir)

    print(f"""
        Generating topics: 
            num_topics: {num_topics} 
            chunk_size: {chunk_size}
            passes: {passes}
            data directory: {data_dir}
            output directory: {output_dir}
    """)
    #### Time profiling the code block using %%timeit
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                          id2word=dictionary,
                                          num_topics=num_topics,
                                          update_every=1,
                                          chunksize=chunk_size,
                                          passes=passes)
    print(f'=== Topics generation finished! {datetime.now() - start} elapsed ===')

    lda.save(f'./output/{output_dir}/Task1_ldamodel_full_{num_topics}_topics.gensim')
    print(f'=== LDA model saved at ./{output_dir}/Task1_ldamodel_full_{num_topics}_topics.gensim ===')

if __name__ == "__main__":
    import argparse

    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')

    # Required positional argument
    parser.add_argument('--num-topics', type=int, nargs='?', const=10, default=10,
                        help='Number of desired topics')

    # Optional positional argument
    parser.add_argument('--chunk-size', type=int, nargs='?', const=10000, default=10000,
                        help='Number of documents in a chunk')

    # Optional argument
    parser.add_argument('--passes', type=int, nargs='?', const=0, default=0,
                        help='Number of times to pass through whole corpus')

    # Optional argument
    parser.add_argument('--out-dir', type=str, nargs='?', const=f'{DATA_FOLDER}', default=f'{DATA_FOLDER}',
                        help='Number of times to pass through whole corpus')

    # Optional argument
    parser.add_argument('--data-dir', type=str, nargs='?', const=f'{OUTPUT_FOLDER}', default=f'{OUTPUT_FOLDER}',
                        help='Number of times to pass through whole corpus')
    args = parser.parse_args()

    start(args.num_topics, args.chunk_size, args.passes, args.data_dir, args.out_dir)