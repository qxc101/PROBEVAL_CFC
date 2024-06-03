import numpy as np
from collections import defaultdict
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.gmeans import gmeans
from scipy.cluster.hierarchy import linkage, fcluster
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
# from commonsense_validation_api.clustering import kmeans_plusplus_initializer
from typing import List, Dict, Optional, Tuple


def clustering_G_probabilities(
        sampled_answers_G: List[int],
        embeddings: Dict[str, List],
        n_component: float,
        clustering_algorithm: Optional[str] = "gmeans",

) -> Tuple[Dict[str, float], dict, Dict[str, int], int]:
    """
    Run xmeans or gmeans over ground-truth answer list.
    Args:
        sampled_answers_G: ground-truth answer list
        embeddings: pre-computed fast text embeddings
        n_component: number of starting component for the clustering algorithm
        clustering_algorithm
    Returns:
        probabilities_C_G: probabilities of each ground truth cluster.
        individual_cluster_info_C_G: dictionary with cluster name as key, and answer list, count as value.
    """
    
    random_seed=153
    # assert len(embeddings) == len(set(sampled_answers_G))
    normalized_embedding_C_G = [embeddings[i] for i in sampled_answers_G] # shape: answer_number * embedding_size
    amount_initial_centers = int(n_component)

    if clustering_algorithm == "gmeans":
        cluster_instance = gmeans(
            normalized_embedding_C_G, k_init=amount_initial_centers, repeat=20, random_state=random_seed)
    elif clustering_algorithm == "xmeans":
        initial_centers = kmeans_plusplus_initializer(normalized_embedding_C_G,
                                                      amount_initial_centers,
                                                      random_state=random_seed).initialize()
        cluster_instance = xmeans(normalized_embedding_C_G, initial_centers,
                                  kmax=20, tolerance = 0.0025, random_state=random_seed)
    else:
        raise ValueError('Invalid clustering algorithm {}'.format(clustering_algorithm))

    cluster_instance.process()
    formed_clusters = cluster_instance.get_clusters()
    num_of_C_clusters = len(formed_clusters)

    # group the clusters
    grouped_clusters_C_G = defaultdict(list)
    for c in range(1, len(formed_clusters) + 1, 1):
        formed_cluster = formed_clusters[c - 1]
        for x in formed_cluster:
            grouped_clusters_C_G['c_' + str(c)].append(sampled_answers_G[x])

    grouped_clusters_C_G['wrong'] = []
    num_of_C_clusters += 1

    probabilities_C_G = {}
    individual_cluster_info_C_G = {}
    for k, v in grouped_clusters_C_G.items():
        individual_cluster_info_C_G[k] = {}
        individual_cluster_info_C_G[k]["count"] = len(v)
        individual_cluster_info_C_G[k]["answers"] = v
        probabilities_C_G[k] = individual_cluster_info_C_G[k]["count"] / \
                               len(sampled_answers_G)
    # print("Number of ground truth clusters using Fasttext with {} is {}".format(clustering_algorithm, num_of_C_clusters))
    
    assert abs(np.sum(np.asarray(list(probabilities_C_G.values())))-1)<1e-10
    if abs(np.sum(np.asarray(list(probabilities_C_G.values())))-1)>=1e-10:
        import pdb; pdb.set_trace()
    return probabilities_C_G, individual_cluster_info_C_G


def clustering_hac_G_probabilities(
        sampled_answers_G: List[int],
        embeddings: Dict[str, List],
        n_component: float,
        clustering_algorithm: Optional[str] = "average"
) -> Tuple[Dict[str, float], dict, Dict[str, int], int]:
    """
    Run Hierarchical Agglomerative Clustering over ground-truth answer list.
    Args:
        sampled_answers_G: ground-truth answer list
        embeddings: pre-computed fast text embeddings
        n_component: number of starting component for the clustering algorithm
        clustering_algorithm: hac
    Returns:
        probabilities_C_G: probabilities of each ground truth cluster.
        individual_cluster_info_C_G: dictionary with cluster name as key, and answer list, count as value.
    """
    # print(len(embeddings),len(sampled_answers_G))
    # print(sampled_answers_G)
    # print(embeddings.keys())
    # assert len(embeddings) == len(set(sampled_answers_G))
    normalized_embedding_C_G = [embeddings[i] for i in sampled_answers_G] # shape: answer_number * embedding_size

    Z = linkage(normalized_embedding_C_G, clustering_algorithm)
    labels = fcluster(Z, t=n_component, criterion='distance')
    num_of_C_clusters = max(labels)
    grouped_clusters_C_G = defaultdict(list)

    for j, c in enumerate(labels):
        grouped_clusters_C_G['c_' + str(c)].append(sampled_answers_G[j])

    grouped_clusters_C_G['wrong'] = []
    num_of_C_clusters += 1

    probabilities_C_G = {}
    individual_cluster_info_C_G = {}
    for k, v in grouped_clusters_C_G.items():
        individual_cluster_info_C_G[k] = {}
        individual_cluster_info_C_G[k]["count"] = len(v)
        individual_cluster_info_C_G[k]["answers"] = v
        probabilities_C_G[k] = individual_cluster_info_C_G[k]["count"] / \
                               len(sampled_answers_G)
    
    # print("Number of ground truth clusters using Fasttext with HAC is {}".format(num_of_C_clusters))
    
    assert abs(np.sum(np.asarray(list(probabilities_C_G.values())))-1)<1e-10
    if abs(np.sum(np.asarray(list(probabilities_C_G.values())))-1)>=1e-10:
        import pdb; pdb.set_trace()
    return probabilities_C_G, individual_cluster_info_C_G