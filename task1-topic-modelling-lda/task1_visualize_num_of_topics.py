import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplcursors

def read_scores(file_name):
    scores = pd.read_csv(file_name, encoding="utf8")
    return scores

def fix_score(score):
    max = np.nanmax(score[score != np.inf])
    argmax = np.nanargmax(score[score != np.inf])
    print(f"Max: {max} at amount #{argmax}")

    fixed_score = [s if s != float("inf") else max for s in score]

    return fixed_score[:20]

def visualize(scores):
    x = scores['num_of_topics'][:20]
    y_c_v = fix_score(scores['c_v'])
    y_u_mass = fix_score(scores['u_mass'])
    y_c_npmi = fix_score(scores['c_npmi'])
    y_c_uci = fix_score(scores['c_uci'])

    fig, ax = plt.subplots()
    plt.plot(x, y_c_v, label="c_v")
    plt.plot(x, y_u_mass, label="u_mass")
    plt.plot(x, y_c_npmi, label="c_npmi")
    plt.plot(x, y_c_uci, label="c_uci")
    plt.xlabel("Num of topics")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Coherence scores for LDA model with 4 different types of measures")
    mplcursors.cursor(hover=True)

    plt.show()  

if __name__ == "__main__":
    score = read_scores('./num_of_topics_results.csv')
    print(score)
    visualize(score)
    