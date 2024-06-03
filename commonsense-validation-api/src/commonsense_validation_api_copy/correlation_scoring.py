import numpy as np
import json
from collections import defaultdict
import itertools
import math


def correlation_scoring(
        kl_scores_per_n: dict,
        ranking_coefficients: list,
        do_protoQA: bool
):
    ranking_scores_per_component = defaultdict(dict)
    for n_component in kl_scores_per_n:
        component_scores = kl_scores_per_n[n_component]
        ranking_score_list = defaultdict(list)
        auto_score_list, auto_protoQA_score_list, human_score_list = [], [], []
        for qid in component_scores:
            question_scores = component_scores[qid]
            auto_score_list.append(question_scores['auto'])
            human_score_list.append(question_scores['human'])

            type_, ranking_type = ranking_coefficients[0]
            ranking_score = ranking_type(question_scores['auto'], question_scores['human'])
            ranking_score_list['spearman'].append(ranking_score[0])

            type_, ranking_type = ranking_coefficients[1]
            ranking_score = ranking_type(question_scores['auto'], question_scores['human'])
            ranking_score_list['pearson'].append(ranking_score[0])

            if do_protoQA:
                type_, ranking_type = ranking_coefficients[0]
                auto_protoQA_score_list.append(question_scores['auto_protoQA'])
                ranking_score = ranking_type(question_scores['auto_protoQA'], question_scores['human'])
                if np.isnan(ranking_score[0]):
                    ranking_score_list['spearman_protoQA'].append(0)
                else:
                    ranking_score_list['spearman_protoQA'].append(ranking_score[0])
                # import pdb; pdb.set_trace()
                # ranking_score = ranking_type(question_scores['auto_protoQA'], question_scores['human'])
                # ranking_score_list[type_].append(ranking_score[0])

        
        # Computer correlation over the KL scores of one question and averaging the correlation across all questions
        avg_spearman = np.mean(np.asarray(ranking_score_list['spearman']))
        avg_pearson = np.mean(np.asarray(ranking_score_list['pearson']))
        print("Avg spearman = {}, and Avg pearson = {} for n_component={}".format(
            avg_spearman, avg_pearson, n_component))
        ranking_scores_per_component[n_component]['average'] = (avg_spearman, avg_pearson)
        
        # Computer correlation over the KL scores for all questions
        all_auto_score_list = list(itertools.chain(*auto_score_list))
        all_human_score_list = list(itertools.chain(*human_score_list))

        ranking_score_overall = defaultdict(list)
        type_, ranking_type = ranking_coefficients[0]
        ranking_score = ranking_type(all_auto_score_list, all_human_score_list)
        ranking_score_overall[type_].append(ranking_score[0])

        type_, ranking_type = ranking_coefficients[1]
        ranking_score = ranking_type(all_auto_score_list, all_human_score_list)
        ranking_score_overall[type_].append(ranking_score[0])
        print("Overall spearman = {} and Overall pearson  = {} for n_component={}".format(
            ranking_score_overall['spearman'], ranking_score_overall['pearson'], n_component))
        ranking_scores_per_component[n_component]['overall'] = (
            ranking_score_overall['spearman'][0], ranking_score_overall['pearson'][0])
        
        if do_protoQA:
            # Replace NaN values with 0
            
            avg_spearman_protoQA = np.mean(np.asarray(ranking_score_list['spearman_protoQA']))
            type_, ranking_type = ranking_coefficients[0]
            all_auto_protoQA_score_list = list(itertools.chain(*auto_protoQA_score_list))
            ranking_score = ranking_type(all_auto_protoQA_score_list, all_human_score_list)
            ranking_score_overall_spearman_protoQA = ranking_score[0]
            ranking_scores_per_component[n_component]['protoQA_avg_overrall'] = (
            avg_spearman_protoQA, ranking_score_overall_spearman_protoQA)
            print("avg_spearman_protoQA, ranking_score_overall_spearman_protoQA")
            print(avg_spearman_protoQA, ranking_score_overall_spearman_protoQA)
    # import pdb; pdb.set_trace()
    return ranking_scores_per_component
