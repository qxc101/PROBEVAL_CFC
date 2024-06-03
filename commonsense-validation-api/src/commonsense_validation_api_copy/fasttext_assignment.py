from collections import defaultdict
from typing import Dict, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import accuracy_score
from functools import lru_cache

def fasttext_assignment(
        question_data: dict,
        prediction_data: List[str],
        embeddings: Dict[str, List[float]],
        additional_answer_embeddings: Dict[str, List[float]]
) -> Dict[str, str]:
    true_answers = question_data['answers']['clusters']

    pred_embeds = np.asarray([additional_answer_embeddings[i]
                              for i in prediction_data])

    # cluster_wise_scores = []
    # for k, v in true_answers.items():
    #     if k=="wrong": continue
    #     cluster_k_answers_embeds = np.asarray([embeddings[i] for i in v['answers']])
    #     pairwise_cos_sim = cosine_similarity(cluster_k_answers_embeds, pred_embeds)

    #     max_sim = np.max(pairwise_cos_sim, axis=0).reshape(1, -1)
    #     cluster_wise_scores.append(max_sim)
    # stacked_scores = np.vstack(np.asarray(cluster_wise_scores))
    # cluster_assignments = np.argmax(stacked_scores, axis=0)
    # assigned_clusters = [list(true_answers.keys())[x]
    #                      for x in cluster_assignments]
    # assigned_clusters_per_pred_word = {
    #     w: assigned_clusters[i]
    #     for i, w in enumerate(prediction_data)
    # }
    # print(assigned_clusters_per_pred_word)

    ########################################################

    # thres = []
    # for k, v in true_answers.items():
    #     if k == "wrong":
    #         continue
    #     cluster_k_answers_embeds = np.asarray(
    #         [embeddings[i] for i in v['answers']])
    #     pairwise_cos_sim = cosine_similarity(
    #         cluster_k_answers_embeds, cluster_k_answers_embeds)
    #     try:
    #         # up_tr = np.triu(pairwise_cos_sim, k=1)
    #         # print(up_tr)
    #         # #up_tr = up_tr[up_tr > 0]
    #         # min_sim = np.min(up_tr)
    #         min_sim = 0.5
    #     except ValueError:
    #         min_sim = 0.5
    #     print(k, min_sim)
    #     thres.append(min_sim)

    # cluster_wise_assignments = []
    # for i, k in enumerate(true_answers):
    #     cluster_k_answers_embeds = np.asarray(
    #         [embeddings[i] for i in true_answers[k]['answers']])
    #     pairwise_cos_sim = cosine_similarity(cluster_k_answers_embeds, pred_embeds)
    #     bool_sim_grt_thres = pairwise_cos_sim > thres
    #     bool_sim_grt_thres = np.any(bool_sim_grt_thres, axis=0)
    #     cluster_wise_assignments.append(bool_sim_grt_thres)
    # stacked_scores = np.vstack(np.asarray(cluster_wise_assignments))
    # cluster_assignments = np.argwhere(stacked_scores==True)

    # assignment_dict = defaultdict(list)
    # cluster_names = list(true_answers.keys())
    # for assignment in cluster_assignments.tolist():
    #     assignment_dict[prediction_data[assignment[1]]].append(
    #         cluster_names[assignment[0]])

    # def get_cluster(c):
    #     answers = ""
    #     for x in c:
    #         answers += x + ","
    #     return answers.strip(",")

    # assigned_clusters_per_pred_word = {}
    # for w, cs in assignment_dict.items():
    #     assigned_clusters_per_pred_word[w] = get_cluster(cs)

    assigned_clusters_per_pred_word = {}
    thres = 0.67
    cluster_wise_scores = []
    for k, v in true_answers.items():

        if k == "wrong": continue
        cluster_k_answers_embeds = np.asarray([embeddings[i] for i in v['answers']])
        pairwise_cos_sim = cosine_similarity(cluster_k_answers_embeds, pred_embeds)
        max_sim = np.max(pairwise_cos_sim, axis=0).reshape(1, -1)
        cluster_wise_scores.append(max_sim)
    stacked_scores = np.vstack(np.asarray(cluster_wise_scores))
    pred_answer_max_score = np.max(stacked_scores, axis=0)
    cluster_assignments = np.argmax(stacked_scores, axis=0)
    for i, x in enumerate(pred_answer_max_score):
        if x < thres:
            assigned_clusters_per_pred_word[prediction_data[i]] = 'wrong'
        else:
            assigned_clusters_per_pred_word[prediction_data[i]] = list(true_answers.keys())[
                cluster_assignments[i]]

    return assigned_clusters_per_pred_word

def truncate_to_5_digits(value: float) -> float:
    """
    Truncate a float value to its first 5 digits.

    Args:
    - value (float): The value to be truncated.

    Returns:
    - float: Truncated value.
    """
    multiplier = 10 ** 5
    return abs(int(value * multiplier) / multiplier)

# @lru_cache(maxsize=None)
def dynamic_threshold(
        question_data: dict,
        prediction_data: List[str],
        embeddings: Dict[str, List[float]],
        additional_answer_embeddings: Dict[str, List[float]], 
        threshold: float
):  
    answer_embed_list = []
    cluster_list = []
    seen_words = defaultdict(str)
    true_answers = question_data['answers']['clusters']

    for cid, each_cluster in true_answers.items():
        # cluster_list.extend([cid] * len(each_cluster['answers']))
        for ans in each_cluster['answers']:
            seen_words[ans] = cid
            cluster_list.append(cid)
            answer_embed_list.append(np.asarray(embeddings[ans]))
    # because current sampled answers comes from the combination of ground-truth data and prediction data, so we check both.
    try:
        pred_embeds = [additional_answer_embeddings[ans] if ans in additional_answer_embeddings else embeddings[ans] for ans in prediction_data]
    except Exception as e:
        print(e)
        import pdb; pdb.set_trace()
    assert len(pred_embeds) == len(prediction_data)
    # answer_embed_list = [np.vectorize(truncate_to_5_digits)(embed) for embed in answer_embed_list]
    # pred_embeds = [np.vectorize(truncate_to_5_digits)(embed) for embed in pred_embeds]

    predictors = {}

    # train a Gaussian Regression classifier for each cluster
    for each_group in list(set(cluster_list)):
        # if each_group == "c_1":
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # # print(answer_embed_list[0])
        # # print(answer_embed_list[20])
        # # zzzzzzz = (answer_embed_list[100])
        # total_embed_sum = np.sum(answer_embed_list, axis=0)
    
        # print(f"Sum of answer_embed_list for cluster {each_group}: {total_embed_sum}")
        # # Printing the sum of y
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        clf = GaussianProcessRegressor(alpha=1e-7, n_restarts_optimizer=4)
        # Create binary labels based on the current group
        y = np.array([float(int(x == each_group)) for x in cluster_list])

        # print(f"Sum of y for cluster {each_group}: {np.sum(y)}")

        # Train the GaussianProcessRegressor
        clf.fit(answer_embed_list, y)
        # Get predictions on training data
        y_pred = clf.predict(answer_embed_list)
        # Convert continuous predictions to binary
        y_pred_binary = (y_pred > threshold).astype(float)
        # Calculate accuracy
        acc = accuracy_score(y, y_pred_binary)
        # Store the trained model and its accuracy
        predictors[each_group] = clf

    # predict each prediction answer using the trained Gaussian classifier
    predictions_dictionary = defaultdict(list)
    for predictor_id in sorted(predictors):
        p = predictors[predictor_id].predict(pred_embeds)
        for ans_idx in range(len(pred_embeds)):
            # if prediction_data[ans_idx] == "bus driver":
            #     print(ans_idx)
            #     print(predictor_id)
            #     print(type(pred_embeds[ans_idx]))
            #     print(type(pred_embeds[ans_idx][0]))
            #     print(f"Embedding for 'to make pasta': {pred_embeds[ans_idx]}")
            #     print(p[ans_idx])
            predictions_dictionary[ans_idx].append((p[ans_idx], predictor_id))
    assert len(predictions_dictionary) == len(pred_embeds)
    assert len(predictions_dictionary[0]) == len(set(cluster_list))

    # assign clusters
    result = {}
    for ans_idx in predictions_dictionary:
        ans_string = prediction_data[ans_idx]
        best = deal_multiple_best(predictions_dictionary[ans_idx])
        # if ans_string in seen_words:
            # if best[1] != seen_words[ans_string]:
            #     if ans_string == "bus driver":
            #         print("fffffffffffffff")
            #         print(f"Prediction: {ans_string}, Predicted Cluster: {best[1]}, Actual Cluster: {seen_words[ans_string]}")
            #         print(predictions_dictionary[ans_idx])
            # else:
            #     if ans_string == "bus driver":
            #         print("ttttttttttttttt")
            #         print(f"Prediction: {ans_string}, Predicted Cluster: {best[1]}, Actual Cluster: {seen_words[ans_string]}")
            #         print(predictions_dictionary[ans_idx])
        result[ans_string] = best[1] if best[0]>threshold else "wrong"
    return result


def deal_multiple_best(scores):
    """
    Args:
        scores: list of tuples (score, cluster_name)
    Returns:
        tuple of (best_score, best_cluster)
    """
    best_cluster, best_score = '', 0
    sorted_scores = sorted(scores, key = lambda x: x[0])
    best_score = sorted_scores[-1][0]
    for (score, name) in sorted_scores:
        if score==best_score:
            if best_cluster=='':
                best_cluster=name
            else:
                best_cluster = best_cluster+','+name
    return (best_score, best_cluster)
