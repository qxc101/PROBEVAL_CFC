import os
import random
import json
from collections import defaultdict
from functools import partial
from datetime import datetime
from typing import Dict, List

from commonsense_validation_api_copy.evaluation import general_eval
from commonsense_validation_api_copy.assignment import wn_automatic_probabilities, fasttext_assignment_probabilities
from commonsense_validation_api_copy.data_processing import QuestionAndAnswerClusters


# fasttext
gt_cluster_and_fasttext = partial(
        general_eval, num_sampling_steps=30,
        assignment_answer_distribution=fasttext_assignment_probabilities, eval_type='gt_fasttext',
        repeated_human_answers_sampling=True)

xmeans_cluster_and_fasttext = partial(
        general_eval, use_custom_clustering=True, num_sampling_steps=30,
        eval_type='xmeans_cluster_and_fasttext', clustering_algorithm="xmeans",
        assignment_answer_distribution=fasttext_assignment_probabilities, repeated_human_answers_sampling=True)

gmeans_cluster_and_fasttext = partial(
        general_eval, use_custom_clustering=True, num_sampling_steps=30,
        eval_type='gmeans_cluster_and_fasttext', clustering_algorithm="gmeans",
        assignment_answer_distribution=fasttext_assignment_probabilities, repeated_human_answers_sampling=True)

hac_cluster_and_fasttext = partial(
        general_eval, use_custom_clustering=True, num_sampling_steps=30,
        eval_type='hac_cluster_and_fasttext', clustering_algorithm='hac',
        assignment_answer_distribution=fasttext_assignment_probabilities,
        repeated_human_answers_sampling=True, hac_linkage_function='average')

# wordnet
gt_cluster_and_wordnet = partial(
        general_eval, num_sampling_steps=30,
        assignment_answer_distribution=wn_automatic_probabilities, eval_type='gt_wordnet',
        repeated_human_answers_sampling=True)

xmeans_cluster_and_wordnet = partial(
        general_eval, use_custom_clustering=True, num_sampling_steps=30,
        assignment_answer_distribution=wn_automatic_probabilities, eval_type='cluster_wordnet_xmeans',
        clustering_algorithm="xmeans", repeated_human_answers_sampling=True)

gmeans_cluster_and_wordnet = partial(
        general_eval, use_custom_clustering=True, num_sampling_steps=30,
        assignment_answer_distribution=wn_automatic_probabilities, eval_type='gmeans_cluster_and_wordnet',
        clustering_algorithm="gmeans", repeated_human_answers_sampling=True
    )

hac_cluster_and_wordnet = partial(
        general_eval, use_custom_clustering=True, num_sampling_steps=30,
        assignment_answer_distribution=wn_automatic_probabilities, eval_type='hac_cluster_and_wordnet',
        clustering_algorithm='hac', repeated_human_answers_sampling=True,
        hac_linkage_function='average'
    )


all_eval_funcs = {
    "xmeans_cluster_and_fasttext": xmeans_cluster_and_fasttext,
    "gmeans_cluster_and_fasttext": gmeans_cluster_and_fasttext,
    "hac_cluster_and_fasttext": hac_cluster_and_fasttext,
    "xmeans_cluster_and_wordnet": xmeans_cluster_and_wordnet,
    "gmeans_cluster_and_wordnet": gmeans_cluster_and_wordnet,
    "hac_cluster_and_wordnet": hac_cluster_and_wordnet,
    "gt_cluster_and_fasttext": gt_cluster_and_fasttext,
    "gt_cluster_and_wordnet": gt_cluster_and_wordnet
}


def multi_evals(
        eval_type: str,
        question_data: Dict[str, QuestionAndAnswerClusters],
        prediction_data: Dict[str, List[str]],
        embedding: str,
        annotated_prediction_answers,
        clustering_algo: str,
        n_component: int,
        num_sampling_steps: int = 30,
        save_files: bool = True,
        scores_location: str = "./src/scores/",
        debug: bool = False,
        hac_linkage_function: str = "average",
        sample_function: str="diverse",
        assignment: str="gaussian",
        do_protoQA: bool = False
) -> None:
    now = datetime.now()
    # current_time = now.strftime("%m_%d_%Y::%H:%M:%S/")
    filename = 'cfc_round1_cluster_'+clustering_algo+'_sample_'+sample_function+'/'
    path = os.path.join(scores_location, filename)
    # path = os.path.join(scores_location, 'diverse/')


    answer_with_vectors = defaultdict(dict)
    eval_score_details = defaultdict(dict)
    with open(embedding, 'r') as f:
        embeds = json.load(f)

    # random.seed(10)
    # randomqid = random.sample(list(prediction_data.keys()), 5)
    for qid in prediction_data:
        # if qid in randomqid:
        true_answer_vectors = embeds[qid]["true_answer_vectors"]
        pred_answer_vectors = embeds[qid]["pred_answer_vectors"]
        answer_with_vectors[qid]["true_answer_vectors"] = true_answer_vectors
        answer_with_vectors[qid]["pred_answer_vectors"] = pred_answer_vectors

#     print("answer_with_vectors", len(answer_with_vectors))
    
    eval_fn = all_eval_funcs[eval_type]
    do_sampling = True if annotated_prediction_answers else False
    print(len(answer_with_vectors))
    print(do_sampling)
    eval_score_details[eval_type], average_ranking_score = eval_fn(question_data, prediction_data, answer_with_vectors,
        num_sampling_steps=num_sampling_steps, save_files=save_files,
        n_component=n_component,
        path=path, run_sampling=do_sampling, debug=debug, hac_linkage_function=hac_linkage_function,
        sample_function=sample_function, assignment=assignment, 
        do_protoQA=do_protoQA)
    return sum(eval_score_details[eval_type])/len(eval_score_details[eval_type]), average_ranking_score

