from collections import defaultdict
import json
import pandas as pd
from typing import *
import numpy as np


def pre_process(filename):
    """Assigns the answers that are not present in any clusters to a new cluster and updates the count

    Parameters
    ----------
        
    Returns
    -------
    dict
        new dict with an additional clusters for the answers that are not clustered
    """

    protoqa_jsonl_filename = filename
    protoqa_json = [json.loads(line) for line in open(protoqa_jsonl_filename)]

    cluster_number_list = []
    for example in protoqa_json:
        answers = list(example['answers']['raw'].keys())
        cluster_answers = []
        next_cluster_id = 0
        cluster_name = ''
        for i, cluster_id in enumerate(list(example['answers']['clusters'])):
            cluster = example['answers']['clusters'][cluster_id]['answers']
            for x in cluster:
                cluster_answers.append(x)
            next_cluster_id = i
            cluster_name = cluster_id
        next_cluster_id = next_cluster_id + 1
        cluster_number_list.append(next_cluster_id)
        new_cluster_id = cluster_name.split('.')[0] + '.' + str(next_cluster_id)

        example['answers']['clusters']['wrong'] = {}
        example['answers']['clusters']['wrong']['answers'] = []
        example['answers']['clusters']['wrong']['count'] = 0

        # # some answers are not clustered in to clusters. So, assigning them to the new cluster
        # outliers = [x for x in answers if x not in cluster_answers]
        # outlier_count = 0

        # for x in outliers:
        #     outlier_count += int(example['answers']['raw'][x])
        # if outlier_count > 0:
        #     example['answers']['clusters'][new_cluster_id] = {}
        #     example['answers']['clusters'][new_cluster_id]['answers'] = outliers
        #     example['answers']['clusters'][new_cluster_id]['count'] = str(outlier_count)
        #     example['num']['clusters'] = str(int(example['num']['clusters'])+1)
        #     example['num']['answers'] = str(int(example['num']['answers'])+outlier_count)
    return protoqa_json


def get_additional_annotated_answers(
        df: pd.DataFrame,
        hm: pd.DataFrame,
        question_number_string: str,
        question_string: str,
        cluster_map: Dict[str, str],
        repeated_human_answers_sampling: bool = True
) -> Tuple[List[str], List[int], Dict[str, str]]:
    gpt2_question_information = df.loc[(df['qid'] == question_number_string)]
    # & (df['cluster'] != '?')]
    gpt2_additional_answers_info, num_of_gpt2_answers = get_valid_answers_from_additional_answers(
        gpt2_question_information,
        cluster_map,
        False
    )

    if repeated_human_answers_sampling:
        human_question_information = hm.loc[(hm['question'] == question_string)]
        # & (hm['cluster'] != '?')]
        human_additional_answers_info, num_of_human_answers = get_valid_answers_from_additional_answers(
            human_question_information,
            cluster_map,
            True
        )
        ratio = int(num_of_gpt2_answers / num_of_human_answers)

        for k in human_additional_answers_info:
            human_additional_answers_info[k]['count'] = human_additional_answers_info[k]['count'] * ratio
        # Functioning under the assumption that the same answer in GPT2 and the human set are mapped to same cluster.
        gpt2_human_union = defaultdict(dict)
        union_answers = sorted(list(set(gpt2_additional_answers_info.keys()) | set(human_additional_answers_info.keys())))

        for ua in union_answers:
            if ua in gpt2_additional_answers_info and ua in human_additional_answers_info:
                gpt2_human_union[ua]['count'] = gpt2_additional_answers_info[ua]['count'] + \
                                                human_additional_answers_info[ua]['count']
                gpt2_human_union[ua]['cluster'] = gpt2_additional_answers_info[ua]['cluster']
            elif ua in gpt2_additional_answers_info:
                gpt2_human_union[ua]['count'] = gpt2_additional_answers_info[ua]['count']
                gpt2_human_union[ua]['cluster'] = gpt2_additional_answers_info[ua]['cluster']
            else:
                gpt2_human_union[ua]['count'] = human_additional_answers_info[ua]['count']
                gpt2_human_union[ua]['cluster'] = human_additional_answers_info[ua]['cluster']

        return gpt2_human_union

    else:
        return gpt2_additional_answers_info


# need to remove this
def get_inverse_map(
        raw_answers: dict
) -> Tuple[Dict[int, str], int]:
    """Computes the inverse map of answer strings. ex 1->broken socks, 2->religion

    Parameters
    ----------
    raw_answers: dict
        The answers and their counts

    Returns
    -------
    inverse_map: dict
        total variance value
    total_answers_count: int
        the sum of the answers length form the raw form
    """

    # Create an inverse map that'll be used for sampling
    inverse_map = {}
    count = 0
    total_answers_count = 0
    answer_counts = list(raw_answers.values())
    answers_list = list(raw_answers.keys())
    for answer_number, c in enumerate(answer_counts):
        total_answers_count += c
        for i in range(c):
            inverse_map[count] = answers_list[answer_number]
            count += 1
    return inverse_map, total_answers_count


def get_inverse_map_1(
        question_data: dict,
        raw_answers: dict

) -> Tuple[Dict[int, str], int]:
    """
    Computes the inverse map of answer strings. ex 1->broken socks, 2->religion

    Parameters
    ----------
    answer_dict: dict
        The answers and their counts

    Returns
    -------
    inverse_map: dict
        total variance value
    total_answers_count: int
        the sum of the answers length form the raw form
    """

    # Create an inverse map that'll be used for sampling
    inverse_map = {}
    count = 0
    total_answers_count = 0

    for cluster, count_and_cluster_str in question_data.items():
        for answer in cluster:
            if answer in raw_answers:
                c = raw_answers[answer]
                total_answers_count += c
                for i in range(c):
                    inverse_map[count] = answer
                    count += 1
    return inverse_map, total_answers_count

    # answer_counts = list(raw_answers.values())
    # answers_list = list(raw_answers.keys())
    # for answer_number, c in enumerate(answer_counts):
    #     total_answers_count += c
    #     for i in range(c):
    #         inverse_map[count] = answers_list[answer_number]
    #         count += 1
    # return inverse_map, total_answers_count


def get_valid_answers_from_additional_answers(
        question_information: pd.DataFrame,
        cluster_map: Dict[str, str],
        is_human: bool
) -> Tuple[dict, int]:
    answers = question_information['answer'].tolist()
    if not is_human:
        answers_count = question_information['count'].tolist()
    answers_clusters = question_information['cluster'].tolist()
    valid_cluster_numbers = []
    valid_answers_count = []
    valid_answers = []
    number_of_answers = 0
    additional_answers_cluster_map = {}
    additional_answers_info = defaultdict(dict)

    for i, cluster_answer in enumerate(answers_clusters):
        if cluster_answer in cluster_map:
            valid_cluster_numbers.append(cluster_map[cluster_answer])
            additional_answers_cluster_map[answers[i]] = cluster_map[cluster_answer]
            additional_answers_info[answers[i]]['cluster'] = cluster_map[cluster_answer]
        else:
            valid_cluster_numbers.append("wrong")
            additional_answers_cluster_map[answers[i]] = "wrong"
            additional_answers_info[answers[i]]['cluster'] = "wrong"
        valid_answers.append(answers[i])
        if is_human:
            valid_answers_count.append(1)
            number_of_answers += 1
            additional_answers_info[answers[i]]['count'] = 1
        else:
            valid_answers_count.append(answers_count[i])
            number_of_answers += answers_count[i]
            additional_answers_info[answers[i]]['count'] = answers_count[i]

    return additional_answers_info, number_of_answers
