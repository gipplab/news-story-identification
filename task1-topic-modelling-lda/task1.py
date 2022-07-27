#https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
import sys
from datetime import datetime
from method_gridsearch_gensim_lda import start_gridsearch
from method_bertopic import start_bertopic

if __name__ == "__main__":
    if len(sys.argv) == 3:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        if input_dir[-1] != "/":
            input_dir+="/"
        if output_dir[-1] != "/":
            output_dir+="/"
        begin_time = datetime.now()

        method = [
            # 'gensim_lda_gridsearch',      # already
            '',                             # bert default
            'all-MiniLM-L6-v2'              # already
        ]

        for m in method:
            if m == 'gensim_lda_grisearch':
                start_gridsearch(input_dir, output_dir)
            else:
                start_bertopic(input_dir, output_dir, embedding_model_name=m, coherence_type='c_uci')

        print(f"=== DONE ! Total time elapsed is {datetime.now() - begin_time}")
    else:
        print('\n'.join(["Unexpected number of commandline arguments.",
                         "Usage: ./task1.py {input-dir} {output-dir}"]))
