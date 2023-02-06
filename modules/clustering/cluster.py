import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram
from os.path import join
from word_embedding.embedder import Embedder
from base.cache import has_cache, save_cache, load_cache
from scipy.cluster.hierarchy import dendrogram

sys.setrecursionlimit(15000)

class Cluster():
    def __init__(self, input_dir, output_dir, dist_matrix, corpus, method='kmeans', use_cache=True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.method = method
        self.distance_matrix = dist_matrix
        self.use_cache = use_cache
        self.num_clusters = 0
        self.corpus = corpus
        
    def process(self):
        if self.use_cache and has_cache([ join(self.output_dir, f'_cache_clusters_{self.method}.pkl') ]):
            objs = load_cache( [ join(self.output_dir, f'_cache_clusters_{self.method}.pkl') ] )
            self.clusters = objs[0]
            return self.clusters

        if self.method == 'kmeans':
            print("TODO kmeans")
        elif self.method == 'agglomerative':
            self.clusters, self.score, self.num_clusters = self.agglomerative()

        return self.clusters

    def agglomerative(self):
        n_sample = len(self.distance_matrix)
        best_score = -1 * 10 ** 7
        best_n_cluster = 0
        best_clusters = None
        for i in range(2, n_sample):
            clusters = AgglomerativeClustering(n_clusters=i, affinity="precomputed", linkage="single").fit(self.distance_matrix)
            silhouette_avg = silhouette_score(self.distance_matrix, clusters.labels_)
            # print(i, silhouette_avg)

            if silhouette_avg > best_score:
                best_score, best_n_cluster, best_clusters = silhouette_avg, i, clusters

        if self.use_cache:
            save_cache([clusters], [ join(self.output_dir, f'_cache_clusters_{self.method}.pkl') ])
        return best_clusters, best_score, best_n_cluster

    def kmeans(self):
        # best_score = 9 * 10**7
        # best_clusters = []
        # best_num_clusters = 0
        # if len(doc_ids) > 0:
        #     corpus = [documents[int(id)] for id in doc_ids]
        #     corpus = self.corpus
        #     dictionary = Dictionary(corpus)
        #     corpus_bow = [dictionary.doc2bow(doc) for doc in corpus]
        #     tfidf_model = TfidfModel(corpus_bow)
        #     corpus_tfidf = tfidf_model[corpus_bow]
        #     num_docs = dictionary.num_docs
        #     num_terms = len(dictionary.keys())
        #     corpus_tfidf_dense = corpus2dense(corpus_tfidf, num_terms, num_docs)

        #     kmean_model_1 = KMeans(init="k-means++", n_clusters=1)
        #     clusters = kmean_model_1.fit_predict(corpus_tfidf_dense.T)
        #     base_inertia = kmean_model_1.inertia_

        #     for num_clusters in range(2, len(corpus) + 1):
        #         # print(num_clusters, ' ', end='')
        #         kmean_model = KMeans(init="k-means++", n_clusters=num_clusters)
        #         clusters = kmean_model.fit_predict(corpus_tfidf_dense.T)
        #         score = kmean_model.inertia_
        #         scaled_score = scaled_inertia(inertia_1=base_inertia, inertia=score, k=num_clusters, alpha=0.02)

        #         if scaled_score < best_score:
        #             best_score = scaled_score
        #             best_clusters = clusters 
        #             best_num_clusters = num_clusters

        # print(f'best: n_clusters: {best_num_clusters}, score: {best_score}')
        # report = {
        #     'topic_id': topic_4ws['topic_id'],
        #     'keywords': topic_4ws['keywords'],
        #     'num_of_clusters': best_num_clusters,
        #     'score': best_score,
        #     'clusters': {}
        # }
        # for i in range(best_num_clusters):
        #     report['clusters'][i] = []
        # for i, cluster_id in enumerate(best_clusters):
        #     report['clusters'][cluster_id].append((doc_ids[i], entities_4ws[i]))

        # with open(f'{output_dir}topic_{topic_4ws["topic_id"]}_4ws_clustered.json', 'w', encoding='utf-8') as f:
        #     json.dump(report, f, indent=4)
        # f.close()
        pass

    def plot_dendrogram(self, model, **kwargs):        
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)

    def save_plot(self, file_name):
        if self.method == 'kmeans':
            pass
        elif self.method == 'agglomerative':
            plt.title("Hierarchical Clustering Dendrogram")
            self.plot_dendrogram(self.clusters, truncate_mode="level", p=3)
            plt.xlabel("Number of points in node (or index of point if no parenthesis).")
            plt.savefig(file_name)
