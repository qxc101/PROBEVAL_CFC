import os
import wandb
from pathlib import Path
from typing import Callable, List, Union, Optional
from collections import Counter
from scipy.stats.stats import pearsonr, spearmanr
from tqdm import tqdm
import time
from commonsense_validation_api_copy.utilities import save_to_file
from commonsense_validation_api_copy.assignment import wn_automatic_probabilities, human_assignment_probability
from commonsense_validation_api_copy.distributions import clustering_G_probabilities, clustering_hac_G_probabilities
from commonsense_validation_api_copy.correlation_scoring import correlation_scoring
from commonsense_validation_api_copy.get_kl_divergence import KLDivergence
from protoqa_evaluator.data_processing import load_question_answer_clusters_from_jsonl
import protoqa_evaluator.evaluation
from protoqa_evaluator.common_evaluations import exact_match_all_eval_funcs, wordnet_all_eval_funcs
from protoqa_evaluator.scoring import wordnet_score
import json
from typing import NamedTuple, Dict, FrozenSet
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import csv

def matching2probs(sampled_answers, detail_max_answers, qid, individual_cluster_info_C_G):
    # Create a dictionary to map clusters to their index in initial_cluster_counts
    cluster_index_map = {cluster_id: index for index, cluster_id in enumerate(individual_cluster_info_C_G.keys())}

    # Initialize counts for each cluster
    cluster_counts = [0] * len(individual_cluster_info_C_G)

    # Process each answer in sampled_answers
    for answer, _ in sampled_answers:
        if answer in detail_max_answers[qid][2]:
            # Find the cluster this answer belongs to
            for cluster_id, cluster_info in individual_cluster_info_C_G.items():
                if answer in cluster_info['answers']:
                    # Increment the count for this cluster
                    cluster_index = cluster_index_map[cluster_id]
                    cluster_counts[cluster_index] += 1
                    break  # Stop searching once the cluster is found
    total_answers = sum(cluster_counts)
    cluster_probabilities = [count / total_answers for count in cluster_counts]

    return cluster_probabilities

def create_bar_chart_i(list1, list2, list3, list4, list5, x, plot_details):
    import matplotlib.pyplot as plt
    import numpy as np


    # Setting up the x-coordinates
    x = np.arange(len(list1))  # the label locations
    width = 0.15  # the width of the bars

    # Plotting the bars
    bars1 = plt.bar(x - 2*width, list1, width, label='gt_dist', color='blue')
    bars2 = plt.bar(x - width, list2, width, label='intende_dist', color='green')
    bars3 = plt.bar(x, list3, width, label='actual_dist', color='red')
    bars4 = plt.bar(x + width, list4, width, label='protoqa_max-10_matching_dist', color='black')
    bars5 = plt.bar(x + 2*width, list5, width, label='cfc_matching_dist', color='pink')

    #     # Function to add value labels on the bars
    # def add_value_labels(bars):
    #     for bar in bars:
    #         height = bar.get_height()
    #         plt.text(bar.get_x() + bar.get_width() / 2, height,
    #                 f'{height:.2f}',  # Formatting the value to 2 decimal places
    #                 ha='center', va='bottom')

    # # Adding value labels to each set of bars
    # add_value_labels(bars1)
    # add_value_labels(bars2)
    # add_value_labels(bars3)
    # add_value_labels(bars4)
    # add_value_labels(bars5)
    # Adding labels, title, and legend
    plt.xlabel('Categories')
    plt.ylabel('Probabilities')
    plt.title('Comparison of Three Probability Lists')
    plt.xticks(x, plot_details["x_labels"]) 
    plt.legend()

    # Svae the plot
    plt.savefig(plot_details["file_path"])
    plt.close()

def create_scatter_plot(filename, score_key, kl_scores_per_each_question):
    human_scores = []
    auto_scores = []
    point_colors = []

    # Data for CSV
    r1q1_human_scores = []
    r1q1_auto_scores = []

    # Pick colors for the specific questions
    specific_questions = ['r2q44', 'r2q45', 'r2q46', 'r2q47', 'r2q49', 'r3q16', 'r3q17','r3q18','r3q19','r3q20']
    colors_for_specific = plt.cm.rainbow(np.linspace(0, 1, len(specific_questions)))
    color_map_specific = dict(zip(specific_questions, colors_for_specific))

    for qid in kl_scores_per_each_question:
        # Assign color based on whether the question is specific or not
        if qid in specific_questions:
            qid_color = color_map_specific[qid]
        else:
            qid_color = 'grey'  # Grey color for other questions

        qid_human_scores = kl_scores_per_each_question[qid]['human']
        qid_auto_scores = kl_scores_per_each_question[qid][score_key]

        if qid == 'r2q44' or qid == 'r3q16':
            r1q1_human_scores.extend(qid_human_scores)
            r1q1_auto_scores.extend(qid_auto_scores)

        human_scores.extend(qid_human_scores)
        auto_scores.extend(qid_auto_scores)
        point_colors.extend([qid_color] * len(qid_human_scores))

    plt.figure(figsize=(8, 6))
    plt.scatter(human_scores, auto_scores, s=20, c=point_colors, alpha=1)
    plt.title(f"Scatter Plot of Human vs {score_key} Scores")
    plt.xlabel("Human Scores")
    plt.ylabel(f"{score_key} Scores")
    plt.savefig(filename)

    # Save r1q1 data to CSV
    with open(filename + ".csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Human Score', score_key])
        for human, auto in zip(r1q1_human_scores, r1q1_auto_scores):
            writer.writerow([human, auto])

class QuestionAndAnswerClusters(NamedTuple):
    question_id: str
    question: str
    answer_clusters: Dict[FrozenSet[str], int]


def ranked_answers_by_frequency(sampled_ans_freq, qid):
    """
    Create a dictionary with the question ID mapping to a list of answers ranked by frequency.

    :param sampled_ans_freq: The dictionary of answers with their frequencies.
    :param qid: The question ID.
    :return: A dictionary in the required format.
    """
    # Sort the items by frequency in descending order
    ranked_answers = sorted(sampled_ans_freq, key=sampled_ans_freq.get, reverse=True)

    # Create the new dictionary with all answers ranked by frequency
    return {qid: ranked_answers}


from collections import defaultdict, Counter

def convert_format(individual_cluster_info, question_text, qid):
    # Assuming all clusters belong to the same question
    question_id = qid
    question_text = question_text
    result_dict = {}
    aggregated_clusters = defaultdict(list)
    raw_answer_counts = Counter()

    for cluster_id, info in individual_cluster_info.items():
        if cluster_id == 'wrong':  # Skip 'wrong' cluster if present
            continue

        answers_set = frozenset(info['answers'])
        count = info['count']
        aggregated_clusters[answers_set] = count
        raw_answer_counts.update(info['answers'])

    result_dict[question_id] = QuestionAndAnswerClusters(question_id=question_id, question=question_text, answer_clusters=dict(aggregated_clusters))
    return result_dict


# def protoQA_evaluate(cluster_path, prediction_answer):
#     predict_answer = {}
#     with open(prediction_answer, 'r') as f:
#         for i, line in enumerate(f.readlines()):
#             json_obj = json.loads(line.strip())
#             for key, answers in json_obj.items():
#                 predict_answer[key] = answers
#     question_data = load_question_answer_clusters_from_jsonl(cluster_path)
#     detail, result = multiple_evals(exact_match_all_eval_funcs, question_data, answers_dict=predict_answer)
#     return result

def general_eval(
        questions_data,
        prediction_data,
        answer_with_vectors,
        *,
        answer_assignment: bool = True,
        use_custom_clustering: bool = False,
        clustering_algorithm: str = "gmeans",
        assignment_answer_distribution: Callable = wn_automatic_probabilities,
        n_component: Optional[float],
        ranking_coefficients: List = [('spearman', spearmanr), ('pearson', pearsonr)],
        num_sampling_steps: int = 30,
        repeated_human_answers_sampling: bool = True,
        eval_type: str = 'gt_wordnet',
        save_files: bool = True,
        path: Union[Path, str],
        run_sampling: bool = True,
        debug: bool = False,
        hac_linkage_function: str="average",
        sample_function: str="diverse",
        assignment: str="gaussian",
        do_protoQA: bool = False,
        do_plot_check = True
) -> dict:
    scores_location = path
    # import pdb; pdb.set_trace()
    if not run_sampling:
        kl_scores_per_each_question = []
        original_clustering = clustering_algorithm

        for qid, answer_vectors in tqdm(answer_with_vectors.items()):
            prediction_answers = prediction_data[qid]
            prediction_answer_frequency = Counter(prediction_answers)
            kl_obj = KLDivergence(qid, questions_data, prediction_data, answer_vectors)

            n_component=n_component

            # automatic clustering for ground-truth answers
            if use_custom_clustering:
                if original_clustering=='gmeans' or original_clustering == 'xmeans':
                    auto_clustered_distribution=clustering_G_probabilities
                elif original_clustering =='hac':
                    auto_clustered_distribution=clustering_hac_G_probabilities
                    clustering_algorithm = hac_linkage_function
                else:
                    raise ValueError('Invalid clustering algorithm, choose from gmeans, xmeans or hac')
                
                probabilities_C_G, individual_cluster_info_C_G = auto_clustered_distribution(
                kl_obj.ground_truth_answer_list,
                kl_obj.true_answer_vectors,
                n_component,
                clustering_algorithm
            )
            else:
                probabilities_C_G, individual_cluster_info_C_G = ground_truth_distribution, individual_cluster_info_G
            # print("******************")
            # for key in individual_cluster_info_C_G:
            #     print(len(key["answers"]))
            # print(individual_cluster_info_C_G)
            kl_obj.targets_json["answers"]["clusters"] = individual_cluster_info_C_G

            new_prediction_answers_eval_kl_score, _ = kl_obj.automatic_kl_score(
                assignment_answer_distribution,
                prediction_answer_frequency,
                n_component,
                probabilities_C_G)

            kl_scores_per_each_question.append(new_prediction_answers_eval_kl_score)
        print('Average KL score for all the questions {}'.format(sum(kl_scores_per_each_question)/len(kl_scores_per_each_question)))
       
        return kl_scores_per_each_question, None

    else:

        kl_scores_per_n = {}

        print("*"*20)
        print("Starting number of clusters: " + str(n_component))
        original_clustering = clustering_algorithm
        kl_scores_per_each_question = {}
        for qid, answer_vectors in tqdm(answer_with_vectors.items()):
            sample_distribution = []
            cluster_map, kl_scores_per_each_question[qid] = {}, {}
            auto_protoQA_scores, hm_protoQA_scores = [], []
            auto_kl_scores, hm_kl_scores, assigned_cluster_count_list = [], [], []
            num_of_clusters, clusters_answer_count = 0, 0
            answer_cluster_set = questions_data[qid].answer_clusters.copy()
            kl_obj = KLDivergence(qid, questions_data, prediction_data, answer_vectors)
            # cluster_map: {"name": r1q1.0}
            for cluster, count_and_cluster_str in answer_cluster_set.items():
                clusters_answer_count += count_and_cluster_str[0]
                num_of_clusters += 1
                for answer in cluster:
                    cluster_map[answer] = count_and_cluster_str[1]

            # additional answers include: 30 samples from ground-truth 100 answer set and all the annotated prediction answers.

            additional_gt_answers, additional_gt_answers_clusters = kl_obj.create_annotated_prediction_answers(
                clusters_answer_count, cluster_map, debug)

            # used for sample answers: interporating between actual prediction and ground-truth-answers.
            # human clustering for evluation answers
            additional_answer_distribution = human_assignment_probability(
                additional_gt_answers,
                additional_gt_answers_clusters,
                kl_obj.ground_truth_cluster_to_ans,
                num_of_clusters,
                answer_assignment=False,
                is_evaluation_set=True
            )

            # human clustering for ground-truth answers
            answers_G = kl_obj.ground_truth_answer_list
            clusters_G = [cluster_map[i] for i in answers_G]
            ground_truth_distribution, individual_cluster_info_G = human_assignment_probability(
                answers_G,
                clusters_G,
                kl_obj.ground_truth_cluster_to_ans,
                num_of_clusters,
                answer_assignment=True,
                is_evaluation_set=False
            )


            # automatic clustering for ground-truth answers
            if use_custom_clustering:
                if original_clustering=='gmeans' or original_clustering == 'xmeans':
                    auto_clustered_distribution=clustering_G_probabilities
                elif original_clustering =='hac':
                    auto_clustered_distribution=clustering_hac_G_probabilities
                    clustering_algorithm = hac_linkage_function
                else:
                    raise ValueError('Invalid clustering algorithm, choose from gmeans, xmeans or hac')

                probabilities_C_G, individual_cluster_info_C_G = auto_clustered_distribution(
                    kl_obj.ground_truth_answer_list,
                    kl_obj.true_answer_vectors,
                    n_component,
                    clustering_algorithm
                )

            else:
                probabilities_C_G, individual_cluster_info_C_G = ground_truth_distribution, individual_cluster_info_G
            kl_obj.targets_json["answers"]["clusters"] = individual_cluster_info_C_G
            # import pdb; pdb.set_trace()
            r2q44_sconverted_sampled_answers = []
            r2q44_sconverted_sampled_ans_freq = []
            r2q44_human_score, r2q44_auto_score, r2q44_protoQA_score = [], [], []
            answer_eg = []
            for i in range(0, num_sampling_steps):
                # import pdb; pdb.set_trace()
                if (qid == "r2q44" or qid == 'r3q16') and do_plot_check:
                    sampled_ans, sampled_ans_clusters, sampled_ans_freq, sample_category_distribution, demo, list1, list2, list3, x, plot_details = kl_obj.sample_evaluation_answers(
                    num_of_clusters,
                    additional_answer_distribution,
                    ground_truth_distribution, sampling_type=sample_function, debug=debug, random_seed=1, do_plot=True, sampling_step=i)
                else:
                    sampled_ans, sampled_ans_clusters, sampled_ans_freq, sample_category_distribution, demo = kl_obj.sample_evaluation_answers(
                    num_of_clusters,
                    additional_answer_distribution,
                    ground_truth_distribution, sampling_type=sample_function, debug=debug, random_seed=1, do_plot=False, sampling_step=i)

                
                human_sampling_kl_score, probabilities_H_P = kl_obj.human_kl_score(
                    sampled_ans,
                    sampled_ans_clusters,
                    ground_truth_distribution,
                    num_of_clusters
                )

                auto_sampling_kl_score, pred_probability = kl_obj.automatic_kl_score(
                    assignment_answer_distribution,
                    sampled_ans_freq,
                    n_component,
                    probabilities_C_G
                )
                if do_protoQA :
                    auto_clusters = convert_format(individual_cluster_info_C_G, questions_data[qid][1], qid)
                    converted_sampled_answers = ranked_answers_by_frequency(sampled_ans_freq, qid)

                    # print("*************** r1 ***************")
                    detail, auto_sampling_protoQA_score = protoqa_evaluator.evaluation.multiple_evals(wordnet_all_eval_funcs, auto_clusters, answers_dict=converted_sampled_answers)
                    # 1: max-1, 2:max-3, 3:max-5, 4:max-10
                    # print(detail["Max Answers - 10"])
                    
                    auto_sampling_protoQA_score = 1 - auto_sampling_protoQA_score[4]

                    # max_correct_10 = partial(protoqa_evaluator.evaluation.general_eval, max_pred_answers=len(converted_sampled_answers[qid]), score_func=wordnet_score)
                    # auto_sampling_protoQA_score = protoqa_evaluator.evaluation.evaluate(max_correct_10, auto_clusters, answers_dict=converted_sampled_answers)
                    # auto_sampling_protoQA_score = 1 - auto_sampling_protoQA_score[qid].score

                    auto_protoQA_scores.append(auto_sampling_protoQA_score)
                    
                    if (qid == "r2q44" or qid == 'r3q16') and do_plot_check:
                        r2q44_sconverted_sampled_answers.append(converted_sampled_answers)
                        r2q44_sconverted_sampled_ans_freq.append(sampled_ans_freq)
                        r2q44_human_score.append(human_sampling_kl_score)
                        r2q44_auto_score.append(auto_sampling_kl_score)
                        r2q44_protoQA_score.append(auto_sampling_protoQA_score)
                        answer_eg.append(demo)

                        # create_bar_chart_i(list1, list2, list3, matching2probs(demo, detail["Max Answers - 10"], qid, individual_cluster_info_C_G), list(pred_probability.values()), x, plot_details)
                        
                auto_kl_scores.append(auto_sampling_kl_score)
                hm_kl_scores.append(human_sampling_kl_score)
                sample_distribution.append(probabilities_H_P)
            
            if (qid == "r2q44" or qid == 'r3q16') and do_plot_check:
                with open('/home/qic69/Desktop/CFC/commonsense-validation-api/src/commonsense_validation_api_copy/fig/r2q44_sconverted_sampled_answers_' + sample_function + '.json', 'w') as file:
                    json.dump(r2q44_sconverted_sampled_answers, file, indent=4)  # indent for pretty printing 
                with open('/home/qic69/Desktop/CFC/commonsense-validation-api/src/commonsense_validation_api_copy/fig/r2q44_sconverted_sampled_ans_freq_' + sample_function + '.json', 'w') as file:
                    json.dump(r2q44_sconverted_sampled_ans_freq, file, indent=4)  # indent for pretty printing
                # Specify the CSV file name
                filename = 'output.csv'

                with open('/home/qic69/Desktop/CFC/commonsense-validation-api/src/commonsense_validation_api_copy/fig/ranking_' + sample_function + '.csv', 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(['r2q44_auto_score', 'r2q44_human_score', 'r2q44_protoQA_score', 'answer'])
                    for row in zip(r2q44_auto_score, r2q44_human_score, r2q44_protoQA_score, answer_eg):
                        csvwriter.writerow(row)
               
            if do_protoQA:
                kl_scores_per_each_question[qid]['auto_protoQA'] = auto_protoQA_scores
            kl_scores_per_each_question[qid]['auto'] = auto_kl_scores
            kl_scores_per_each_question[qid]['human'] = hm_kl_scores
            # sample_probability = probabilities_C_G
            kl_scores_per_each_question[qid]['sample_true_distribution'] = list(sample_category_distribution)
            kl_scores_per_each_question[qid]['sample_distribution'] = sample_distribution

            # print(time.time() - start_time)
            # print(kl_scores_per_each_question[qid]['auto'])
            # print(kl_scores_per_each_question[qid]['human'])
            # print(kl_scores_per_each_question[qid]['sample_true_distribution'])
            # print(kl_scores_per_each_question[qid]['sample_distribution'])
        kl_scores_per_n[n_component] = kl_scores_per_each_question
        
        # Get the correlation scores between the Human and the Automatic method.
        # print(kl_scores_per_n)
        do_plot = do_plot_check
        if do_plot:
            unique_qids = list(kl_scores_per_each_question.keys())
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_qids)))
            color_map = dict(zip(unique_qids, colors))
            create_scatter_plot('/home/qic69/Desktop/CFC/commonsense-validation-api/src/commonsense_validation_api_copy/fig/scatter_plot_protoQA_' + sample_function + '.png', 'auto_protoQA', kl_scores_per_each_question)
            create_scatter_plot('/home/qic69/Desktop/CFC/commonsense-validation-api/src/commonsense_validation_api_copy/fig/scatter_plot_' + sample_function + '.png', 'auto', kl_scores_per_each_question)


        ranking_scores_per_component = correlation_scoring(kl_scores_per_n, ranking_coefficients, do_protoQA)
        # import pdb; pdb.set_trace()
        # print(ranking_scores_per_component)


        # wandb.init(project="CFC_Evaluator", entity="iesl-boxes")
        # wandb.log({'average spearman': ranking_scores_per_component[n_component]['average'][0],
        #            'average pearson': ranking_scores_per_component[n_component]['average'][1],
        #            'overall spearman': ranking_scores_per_component[n_component]['overall'][0],
        #            'overall pearson': ranking_scores_per_component[n_component]['overall'][1]})
        # print(kl_scores_per_n)
        # print(ranking_scores_per_component)

        kl_path = path+'kl_score/'
        correlation_path = path + 'correlation/'
        filename = 'eval_'+eval_type +'_n_'+ str(n_component) +'_linkage_'+ hac_linkage_function
        if save_files:
            try:
                os.mkdir(path)
                os.mkdir(kl_path)
                os.mkdir(correlation_path)
                print("Successfully created the directory %s " % path)
            except OSError:
                print("Creation of the directory %s failed" % path)
            save_to_file(kl_path + filename +".json", kl_scores_per_n)
            save_to_file(correlation_path + filename+ ".json", ranking_scores_per_component)
       
        return kl_scores_per_n, ranking_scores_per_component
