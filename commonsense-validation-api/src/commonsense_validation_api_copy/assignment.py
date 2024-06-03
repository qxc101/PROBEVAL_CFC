import numpy as np
from functools import lru_cache, partial
from collections import defaultdict
from typing import List, Dict, Union, Tuple

from commonsense_validation_api_copy.data_processing import load_question_answer_clusters_from_jsonl
from commonsense_validation_api_copy.scoring_evaluator import general_eval, evaluate
from commonsense_validation_api_copy.fasttext_assignment import dynamic_threshold
from commonsense_validation_api_copy.scoring import *

def human_assignment_probability(
        sampled_answers_G: List[str],
        sampled_clusters_G: List[str],
        ground_truth_clusters: Dict[str, dict],
        num_of_clusters: int,
        answer_assignment: bool = False,
        is_evaluation_set: bool = False
) -> Union[Tuple[Dict[str, float], dict], Dict[str, float]]:
    
    probabilities_H_G = defaultdict(float)
    grouped_clusters = defaultdict(list)
    individual_cluster_info = defaultdict(dict)
    
    for i, cluster in enumerate(sampled_clusters_G):
        grouped_clusters[cluster].append(sampled_answers_G[i])
    
    if not is_evaluation_set:
        grouped_clusters["wrong"] = []

    # initialize cluster probability in case the sampled cluster do not cover all.
    for cluster_number, cluster in enumerate(ground_truth_clusters):
        probabilities_H_G[cluster] = 1 / (len(sampled_answers_G) + num_of_clusters)

    for cluster, answers in grouped_clusters.items():
        individual_cluster_info[cluster] = {}
        # this is for smoothing
        individual_cluster_info[cluster]["count"] = len(answers) + 1
        individual_cluster_info[cluster]["answers"] = answers
        probabilities_H_G[cluster] = individual_cluster_info[cluster]["count"] / \
                                     (len(sampled_answers_G) + num_of_clusters)
    #     print(individual_cluster_info[cluster]["count"])
    #     print(individual_cluster_info[cluster]["answers"])
    #
    # print(abs(np.sum(np.asarray(list(probabilities_H_G.values())))-1.0))
    if abs(np.sum(np.asarray(list(probabilities_H_G.values())))-1.0)>=1e-10:
        import pdb; pdb.set_trace()
    assert abs(np.sum(np.asarray(list(probabilities_H_G.values())))-1.0)<1e-10
    if answer_assignment:
        return probabilities_H_G, individual_cluster_info
    return probabilities_H_G

# @lru_cache(maxsize=None)
def fasttext_assignment_probabilities(
        question_number_string: str,
        targets_json: dict,
        embeddings: dict,
        sampled_embeddings: dict,
        sampled_answers_frequency_dict: Dict[str, int]
) -> Tuple[Dict[str, float], dict, int]:
    # import pdb; pdb.set_trace()

    sampled_answers = list(sampled_answers_frequency_dict.keys())
    sampled_answers_count = list(sampled_answers_frequency_dict.values())
    num_of_answers_assigned = np.sum(np.array(sampled_answers_count))
    clusters = targets_json["answers"]["clusters"]
    num_of_clusters = len(clusters)

    answer_assignments = dynamic_threshold(targets_json, sampled_answers, embeddings, sampled_embeddings, threshold=0.2)
    prediction_assigned_cluster_list = [answer_assignments[x] for x in sampled_answers]

    grouped_predicted_clusters = defaultdict(dict)
    for i, v in enumerate(prediction_assigned_cluster_list):
        split_clusters = v.split(",")
        n = len(split_clusters)
        if n > 1:
            print('')
            for k in range(n):
                grouped_predicted_clusters[split_clusters[k]].append(
                    sampled_answers[i])
                grouped_predicted_clusters[split_clusters[k]].append(
                    sampled_answers_count[i] * (1 / n))
        else:
            cluster_name = split_clusters[0]
            grouped_predicted_clusters = construct_cluster_dict(cluster_name, grouped_predicted_clusters,
                sampled_answers, sampled_answers_count, i)

    probabilities_P = {}
    for cluster in clusters:
        dominator = num_of_answers_assigned+num_of_clusters
        if cluster in grouped_predicted_clusters:
            cluster_count = grouped_predicted_clusters[cluster]['count']
            probabilities_P[cluster] = (cluster_count+1) / dominator
        else:
            probabilities_P[cluster] = 1/dominator
    assert (sum(list(probabilities_P.values()))-1)<1e-10
    return probabilities_P

def wn_automatic_probabilities(
        question_number_string: str,
        targets_json: dict,
        embeddings: dict,
        sampled_embeddings: dict,
        sampled_answers_frequency_dict: Dict[str, int],  
) -> Tuple[Dict[str, float], dict, int]:

    sampled_answers = list(sampled_answers_frequency_dict.keys())
    sampled_answers_count = list(sampled_answers_frequency_dict.values())
    num_of_answers_assigned = np.sum(np.array(sampled_answers_count))
    clusters = targets_json["answers"]["clusters"]
    num_of_clusters = len(clusters)

    predictions_json = {question_number_string: sampled_answers}
    question_data = load_question_answer_clusters_from_jsonl(targets_json)

    wn_func = partial(general_eval, score_func=wordnet_score, assign_cluster_scores=False)
    # import pdb; pdb.set_trace()
    wn_result = evaluate(wn_func, question_data, predictions_json)

    answer_assignments = wn_result[question_number_string].answer_assignment
    # import pdb; pdb.set_trace()
    prediction_assigned_cluster_list = []
    for x in sampled_answers:
        normalized_x = x.strip().lower()  # Normalize the string for matching
        for matched_x in answer_assignments:
            normalized_matched_x = matched_x.strip().lower()
            if normalized_x[:40] == normalized_matched_x[:40]:
                prediction_assigned_cluster_list.append(answer_assignments[matched_x])
                break
        else: 
            # This else clause will execute if the for loop completes without a break
            # Indicating no match was found
            print(f"No match for: {x}")
    # print("sampled_answers  ", sampled_answers)
    # print("answer_assignments  ", answer_assignments)
    # print("prediction_assigned_cluster_list  ", prediction_assigned_cluster_list)
    # prediction_assigned_cluster_list = [answer_assignments[x] for x in sampled_answers]

    grouped_predicted_clusters = defaultdict(dict)
    try:
        assert len(prediction_assigned_cluster_list) == len(sampled_answers)
    except:
        import pdb; pdb.set_trace()
    for i, v in enumerate(prediction_assigned_cluster_list):
        split_clusters = v.split(",")
        n = len(split_clusters)
        if n > 1:
            for k in range(n):
                cluster_name = split_clusters[k]
                try:
                    grouped_predicted_clusters = construct_cluster_dict(cluster_name, grouped_predicted_clusters,
                    sampled_answers, sampled_answers_count, i,
                    split=True, radio=n)
                except:
                    import pdb; pdb.set_trace()
        else:
            cluster_name = split_clusters[0]
            grouped_predicted_clusters = construct_cluster_dict(cluster_name, grouped_predicted_clusters,
                sampled_answers, sampled_answers_count, i)

    probabilities_P = {}
    for cluster in clusters:
        dominator = num_of_answers_assigned+num_of_clusters
        if cluster in grouped_predicted_clusters:
            cluster_count = grouped_predicted_clusters[cluster]['count']
            probabilities_P[cluster] = (cluster_count+1) / dominator
        else:
            probabilities_P[cluster] = 1/dominator
    # print(sum(list(probabilities_P.values()))-1)
    assert (sum(list(probabilities_P.values()))-1)<1e-10
    return probabilities_P


def construct_cluster_dict(cluster_name, result_dict, answer_list, count_list, idx, split=False, radio=1):
    if cluster_name in result_dict:
        result_dict[cluster_name]['answers'].append(answer_list[idx])
        if split:
            result_dict[cluster_name]['count']+=(count_list[idx]*(1/radio))
        else:
            result_dict[cluster_name]['count']+=count_list[idx]
    else:
        temp = {}
        temp['answers']=[answer_list[idx]]
        if split:
            temp['count']=(count_list[idx]*(1/radio))
        else:
            temp['count']=count_list[idx]
        result_dict[cluster_name] = temp
    return result_dict
