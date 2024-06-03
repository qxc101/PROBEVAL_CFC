import copy
import random
import math
from typing import *
import numpy as np
from collections import Counter, defaultdict, OrderedDict
from scipy.special import kl_div

from commonsense_validation_api.utilities import sample_indices, get_random_distribution
from commonsense_validation_api.assignment import human_assignment_probability


def centered_sampling(
        additional_answer_distribution,
        ground_truth_distribution,
        rand_distribution,
        debug, random_seed=0):
    """
    Sample from the linear interpolation between ground-truth answer and random distribution.
    Centered on the actual prediction from GPT2.
    Args:
        additional_answer_distribution: full sample candidate cluster distribution
        ground_truth_distribution: ground-truth cluster distribution
        rand_distribution: uniform cluster distribution
    Returns:
        sample_category_distribution: probabilities of each ground truth cluster.
    """

    ## Source of output randomness
    # np.random.seed(random_seed)
    z = np.random.uniform(0.5, 1)
    # np.random.seed(random_seed+1)
    w1 = np.random.uniform(0, 1)
    # np.random.seed(random_seed+2)
    w2 = np.random.uniform(0, 1)
    w1_normalized = (w1/(w1+w2))*(1-z)
    w2_normalized = (w2/(w1+w2))*(1-z)
    assert abs(z+w1_normalized+w2_normalized-1)<1e-10
    sample_category_distribution = z * (np.asarray(list(additional_answer_distribution.values()))) + \
                                   w1_normalized * np.asarray(list(ground_truth_distribution.values())) + \
                                   w2_normalized * np.asarray(rand_distribution)
    assert abs(sample_category_distribution.sum()-1)<1e-10
    return sample_category_distribution


def alpha_sampling(ground_truth_distribution, rand_distribution, debug, random_seed=1):
    """
    Sample from the linear interpolation between ground-truth answer and random distribution.
    Args:
        additional_answer_distribution: full sample candidate cluster distribution
        rand_distribution: uniform cluster distribution
    Returns:
        sample_category_distribution: probabilities of each ground truth cluster.
    """
    # np.random.seed(random_seed+200)
    w = np.random.uniform(0, 1)
    # w = 1.0
    # print(w)
    sample_category_distribution = w * np.asarray(list(ground_truth_distribution.values())) + \
                                (1 - w) * np.asarray(rand_distribution)
    assert abs(sample_category_distribution.sum()-1)<1e-10
    return sample_category_distribution


def missing_answers_sampling(ground_truth_distribution, random_seed=None):
    """
    Sample from the ground truth distribution but with probabilities of some categories set to 0.
    Ensures that at least one category has zero probability and one category retains its original non-zero probability.
    Args:
        ground_truth_distribution: Full sample candidate cluster distribution (OrderedDict).
        random_seed: Seed for random number generator (optional).
    Returns:
        list_of_probabilities: List of probabilities corresponding to each category in the ground truth distribution.
    """

    # Set random seed for reproducibility if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Convert to a regular dict for easier manipulation
    ground_truth_distribution = dict(ground_truth_distribution)

    # Randomly decide which categories to set to 0 (excluding one category)
    categories = list(ground_truth_distribution.keys())
    num_to_zero = np.random.randint(1, len(categories)-1)  # Number of categories to set to 0
    zeroed_categories = np.random.choice(categories, num_to_zero, replace=False)

    # Set selected categories to 0
    for cat in zeroed_categories:
        ground_truth_distribution[cat] = 0.0

    # Normalize the remaining distribution
    total_prob = sum(ground_truth_distribution.values())
    normalized_distribution = {cat: prob / total_prob for cat, prob in ground_truth_distribution.items()}

    # Extract the probabilities as a list
    list_of_probabilities = list(normalized_distribution.values())

    return list_of_probabilities


def wrong_ranking_sampling(ground_truth_distribution):
    """
    Randomly switch the probabilities of categories in the ground truth distribution.
    Args:
        ground_truth_distribution: full sample candidate cluster distribution (OrderedDict).
        random_seed: Seed for random number generator.
    Returns:
        list_of_probabilities: List of probabilities with randomized switching corresponding to each category.
    """
    # Extract probabilities
    probabilities = list(ground_truth_distribution.values())
    # Shuffle the probabilities
    np.random.shuffle(probabilities)
    # Reassign the shuffled probabilities to the original categories
    shuffled_distribution = {cat: prob for cat, prob in zip(ground_truth_distribution.keys(), probabilities)}
    # Extract the probabilities as a list
    list_of_probabilities = list(shuffled_distribution.values())
    gt_dist = list(ground_truth_distribution.values())
    while list_of_probabilities == gt_dist:
        list_of_probabilities = wrong_ranking_sampling(ground_truth_distribution)
    # print(gt_dist)
    # print(list_of_probabilities)
    return list_of_probabilities

def wrong_score_sampling(ground_truth_distribution):
    """
    Adjust the probabilities of categories to a degree that almost changes their ranking but doesn't.
    Args:
        ground_truth_distribution: full sample candidate cluster distribution (OrderedDict).
    Returns:
        list_of_probabilities: List of adjusted probabilities corresponding to each category in the original order.
    """
    # Sort the probabilities while keeping track of the original order
    sorted_probs_with_index = sorted(
        enumerate(ground_truth_distribution.values()), key=lambda x: x[1], reverse=True
    )
    sorted_indices, sorted_probs = zip(*sorted_probs_with_index)

    # Adjust each probability slightly
    adjusted_probabilities = []
    for i in range(len(sorted_probs)):
        gap_to_prev = sorted_probs[i] - sorted_probs[i - 1] if i > 0 else float('inf')
        gap_to_next = sorted_probs[i + 1] - sorted_probs[i] if i < len(sorted_probs) - 1 else float('inf')

        adjustment_range = min(abs(gap_to_prev), abs(gap_to_next)) / 2
        adjustment_range = min(adjustment_range, 1)  # Limit to max 1%

        lower_bound = max(-sorted_probs[i], -adjustment_range)
        upper_bound = min(1 - sorted_probs[i], adjustment_range)

        adjustment = np.random.uniform(lower_bound, upper_bound)
        adjusted_probabilities.append(sorted_probs[i] + adjustment)

    # Normalize the adjusted probabilities
    total_prob = sum(adjusted_probabilities)
    normalized_probabilities = [prob / total_prob for prob in adjusted_probabilities]

    # Revert back to the original order
    reverted_order_probabilities = [0] * len(normalized_probabilities)
    for original_index, adjusted_prob in zip(sorted_indices, normalized_probabilities):
        reverted_order_probabilities[original_index] = adjusted_prob

    return reverted_order_probabilities

class KLDivergence:
    def __init__(self, qid, questions_data, prediction_data, answer_vectors):
        self.question_number = qid
        self.prediction_data = prediction_data[qid]
        self.raw_answers = questions_data[qid].raw_answers
        self.question_string = questions_data[qid].question
        self.answer_cluster_set = questions_data[qid].answer_clusters.copy()

        self.ground_truth_cluster_to_ans = self.create_cluster_to_ans(self.answer_cluster_set)
        self.ground_truth_answer_list = self.create_ground_truth_answer_list(self.answer_cluster_set, self.raw_answers)
        
        # print(self.answer_cluster_set)
        # print(self.raw_answers)
        self.true_answer_vectors = answer_vectors["true_answer_vectors"]
    
        # print("raw answer len ", len(self.raw_answers))
        # print("embedding len ", len(self.true_answer_vectors))
        # print("GT len", len(self.ground_truth_answer_list))
        # print("Set GT len", len(set(self.ground_truth_answer_list)))
        # print()
        self.pred_answer_vectors = answer_vectors["pred_answer_vectors"]

        # Target data for ground_truth clusters: default is human cluster. 
        # clustering algorithm will modify this.
        answers_dict = {}
        answers_dict['raw_answers']=self.raw_answers
        answers_dict['clusters']={}
        self.targets_json = {"question": self.question_string,
                             "metadata": self.question_number,
                             "answers": answers_dict}
        

        # vars needed for sampling
        self.total_additional_answers_count = 0
        self.additional_cluster_to_answer = {}
        self.additional_answer_to_cluster = {}

    def create_cluster_to_ans(self, answer_cluster_set):
        cluster_to_ans = defaultdict(dict)
        for cluster_key in answer_cluster_set.keys():
            cluster_name = answer_cluster_set[cluster_key][1]
            cluster_to_ans[cluster_name] = {ans for ans in cluster_key}
        return cluster_to_ans


    def create_ground_truth_answer_list(self, answer_cluster_set, raw_answers):
        result = []
        for i, cluster in enumerate(answer_cluster_set.keys()):
            answers_per_cluster = [ans for ans in cluster for i in range(raw_answers[ans])]
            result.append(answers_per_cluster)
        result = sum(result,[])
        return result

    def create_annotated_prediction_answers(
            self,
            clusters_answer_count: int,
            cluster_map: Dict[str, str],
            debug: bool
    ):
        """
            Sample some answers from GT so that we'd have an answer from each cluster
            when trying to create the additional annotated answers. Merge the sampled answers from the GT with the additional
            annotated answers from the prediction set.
        """
        num_samples = 50
        sampled_answers = []
        remaining_answers_in_clusters = {}

        sampled_proportions = (np.array(
            [count for count, _ in self.answer_cluster_set.values()]) / clusters_answer_count) * num_samples

        for i, cluster_key in enumerate(self.answer_cluster_set.keys()):
            cluster_count = self.answer_cluster_set[cluster_key][0]
            cluster_name = self.answer_cluster_set[cluster_key][1]
            answers_per_cluster = sorted([ans for ans in cluster_key for i in range(self.raw_answers[ans])])
            
            random.seed(1)
            ktt = min(len(answers_per_cluster), math.ceil(sampled_proportions[i]))
            sampled_answers_per_cluster = random.sample(answers_per_cluster, ktt)
            sampled_answers.append(sampled_answers_per_cluster)
            remaining_answers_in_clusters[cluster_name] = list(
                (Counter(answers_per_cluster) - Counter(sampled_answers_per_cluster)).elements())

        sampled_dict = Counter(sum(sampled_answers, []))
        # shallow copy
        additional_answers_cluster_info_map = {k:self.prediction_data[k].copy() for k in self.prediction_data}

        # combine sampled grount truth answers with predicted annotated data
        for ans_str, count in sampled_dict.items():
            if ans_str in additional_answers_cluster_info_map:
                additional_answers_cluster_info_map[ans_str]['count']+=count
            else:
                additional_answers_cluster_info_map[ans_str] = {'cluster':cluster_map[ans_str], 
                                                                'count':count}
        # expand answers with the corresponding count and return
        additional_gt_answers, additional_gt_answers_clusters = [], []
        for c_a in additional_answers_cluster_info_map:
            for _ in range(additional_answers_cluster_info_map[c_a]['count']):
                cluster_name = additional_answers_cluster_info_map[c_a]['cluster']
                additional_gt_answers.append(c_a)
                additional_gt_answers_clusters.append(cluster_name)
        assert len(additional_gt_answers) == len(additional_gt_answers_clusters)

        for k, v in additional_answers_cluster_info_map.items():
            if v['cluster'] not in self.additional_cluster_to_answer:
                temp={}
                # +1 is for smoothing
                temp['count']=v['count']+1
                temp['answers']=[k]
                self.additional_cluster_to_answer[v['cluster']]=temp
            else:
                self.additional_cluster_to_answer[v['cluster']]['count']+=v['count']
                self.additional_cluster_to_answer[v['cluster']]['answers'].append(k)
        self.additional_answer_to_cluster = additional_answers_cluster_info_map
        self.total_additional_answers_count = sum([v['count'] for _, v in additional_answers_cluster_info_map.items()])+len(self.additional_cluster_to_answer)
        assert sum([self.additional_cluster_to_answer[k]['count'] for k in self.additional_cluster_to_answer]) == self.total_additional_answers_count
        
        return additional_gt_answers, additional_gt_answers_clusters

    def calculate_cluster_num(self, ground_truth_distribution, sampling_dist, ratio):
        updated_cluster_num = {}
        for k in ground_truth_distribution:
            original_cluster_num = (self.total_additional_answers_count) * sampling_dist[k]
            updated_cluster_num[k] = int(round(ratio * original_cluster_num))
        return updated_cluster_num
    
    def wrong_samplings_filter(self, sampling_type, updated_cluster_num, ground_truth_distribution):
        # Making sure all the categories exist for "wrong_ranking" and "wrong_score"
        
        empty_category = False
        for k in updated_cluster_num.keys():
            if updated_cluster_num[k] == 0:
                empty_category = True
        if empty_category:
            for k in updated_cluster_num.keys():
                updated_cluster_num[k] += 1
        if sampling_type == "wrong_score":
            sorted_updated_cluster_num = [k for k, v in sorted(updated_cluster_num.items(), key=lambda item: item[1], reverse=True)]
            sorted_ground_truth_distribution = [k for k, v in sorted(ground_truth_distribution.items(), key=lambda item: item[1], reverse=True)]
            if sorted_updated_cluster_num != sorted_ground_truth_distribution:
                # print(sorted_updated_cluster_num)
                # print(sorted_ground_truth_distribution)
                reordered_updated_cluster_num = {k: updated_cluster_num[k] for k in sorted_ground_truth_distribution}
                # print([k for k, v in sorted(reordered_updated_cluster_num.items(), key=lambda item: item[1], reverse=True)])
                updated_cluster_num = reordered_updated_cluster_num
        return updated_cluster_num
    
    def compare_actual_intended(self, updated_cluster_num, sample_category_distribution):
            # Check if the actual distribution from the cluster_num are the same ranking with the intended distribution
            total_answers = sum(updated_cluster_num.values())
            actual_distribution = [value / total_answers for value in updated_cluster_num.values()]
            sorted_indices_sample_category_distribution = sorted(range(len(sample_category_distribution)), key=lambda i: sample_category_distribution[i], reverse=True)
            sorted_indices_actual_distribution = sorted(range(len(actual_distribution)), key=lambda i: actual_distribution[i], reverse=True)
            return sorted_indices_sample_category_distribution == sorted_indices_actual_distribution, sorted_indices_sample_category_distribution, sorted_indices_actual_distribution, actual_distribution

    def sample_evaluation_answers(
            self,
            num_of_clusters,
            unordered_additional_answer_distribution,
            unordered_ground_truth_distribution,
            sampling_type: str = "centered",
            debug: bool = False,
            random_seed = 1,
            ratio=1,
            do_plot = False,
            sampling_step = 0
    ):  
        additional_answer_distribution = OrderedDict(sorted(unordered_additional_answer_distribution.items()))
        ground_truth_distribution = OrderedDict(sorted(unordered_ground_truth_distribution.items()))
        rand_distribution = get_random_distribution(num_of_clusters)
        
        # get the distribution over all cluster distributions.
        if sampling_type == "centered":
            sample_category_distribution = centered_sampling(additional_answer_distribution, ground_truth_distribution, rand_distribution, debug, random_seed)
        elif sampling_type == "diverse":
            sample_category_distribution = alpha_sampling(ground_truth_distribution, rand_distribution, debug, random_seed)
        elif sampling_type == "vanila":
            sample_category_distribution = np.asarray(list(additional_answer_distribution.values()))
        elif sampling_type == "missing_answer":
            print("ground_truth_distribution", ground_truth_distribution)
            sample_category_distribution = alpha_sampling(ground_truth_distribution, rand_distribution, debug, random_seed)
            print("alpha_sampling", sample_category_distribution)
            sample_category_distribution = missing_answers_sampling(ground_truth_distribution)
            print("missing_answers_sampling", sample_category_distribution)
        elif sampling_type == "wrong_ranking":
            print("ground_truth_distribution", ground_truth_distribution)
            sample_category_distribution = alpha_sampling(ground_truth_distribution, rand_distribution, debug, random_seed)
            print("alpha_sampling", sample_category_distribution)
            sample_category_distribution = wrong_ranking_sampling(ground_truth_distribution)
            print("wrong_ranking_sampling", sample_category_distribution)
        elif sampling_type == "wrong_score":
            print("ground_truth_distribution", ground_truth_distribution)
            sample_category_distribution = alpha_sampling(ground_truth_distribution, rand_distribution, debug, random_seed)
            print("alpha_sampling", sample_category_distribution)
            sample_category_distribution = wrong_score_sampling(ground_truth_distribution)
            print("wrong_score_sampling", sample_category_distribution)
        else:
            raise ValueError("Invalid sampling type")

        # get the actual number of answers should be sampled for each cluster. 
        # In the case of the expected sample is bigger than the actual samples in the candidate pool.
        # we reduce the sample number to keep the ratio.
        jj = 0
        sampling_dist = {}
        # ratio = 10e+5
        for k, v in ground_truth_distribution.items():
            sampling_dist[k] = sample_category_distribution[jj]
            original_cluster_num = self.total_additional_answers_count * sampling_dist[k]
            # The number of answers of a cluster from the pool
            k_count = self.additional_cluster_to_answer[k]['count']
            if original_cluster_num > k_count:
                ratio = min(ratio, k_count/original_cluster_num)
            jj += 1
            
        updated_cluster_num = self.calculate_cluster_num(ground_truth_distribution, sampling_dist, ratio)
        if sampling_type == "wrong_ranking" or sampling_type == "wrong_score":
            updated_cluster_num = self.wrong_samplings_filter(sampling_type, updated_cluster_num, ground_truth_distribution)
            print(updated_cluster_num)
            same_dis, sorted_indices_sample_category_distribution, sorted_indices_actual_distribution, actual_distribution = self.compare_actual_intended(updated_cluster_num, sample_category_distribution)
            print("actual_distribution original: ", actual_distribution)
            if not same_dis:
                print("Original rankings: ")
                print(sorted_indices_sample_category_distribution)
                print(sorted_indices_actual_distribution)
                gt_dist = list(ground_truth_distribution.values())

                while not same_dis and ratio + 0.1 <= 1:
                    print(ratio)
                    ratio += 0.1
                    updated_cluster_num = self.calculate_cluster_num(ground_truth_distribution, sampling_dist, ratio)
                    updated_cluster_num = self.wrong_samplings_filter(sampling_type, updated_cluster_num, ground_truth_distribution)
                    same_dis, sorted_indices_sample_category_distribution, sorted_indices_actual_distribution, actual_distribution = self.compare_actual_intended(updated_cluster_num, sample_category_distribution)
                    print("actual_distribution changed: ", actual_distribution)
                    print("After filters rankings: ")
                    print(sorted_indices_sample_category_distribution)
                    print(sorted_indices_actual_distribution)
                # import pdb; pdb.set_trace()
                    
            # Finally, check if the changed distribution got corrected back to the original gt distribution
            gt_dist = list(ground_truth_distribution.values())
            same_dis_gt, _, _, actual_distribution = self.compare_actual_intended(updated_cluster_num, gt_dist)
            print("actual_distribution final: ", actual_distribution)
            while actual_distribution == gt_dist and ratio + 0.1 <= 1:
                ratio += 0.1
                updated_cluster_num = self.calculate_cluster_num(ground_truth_distribution, sampling_dist, ratio)
                updated_cluster_num = self.wrong_samplings_filter(sampling_type, updated_cluster_num, ground_truth_distribution)
                same_dis, sorted_indices_sample_category_distribution, sorted_indices_actual_distribution, actual_distribution = self.compare_actual_intended(updated_cluster_num, sample_category_distribution)
            if actual_distribution == gt_dist:
                import pdb; pdb.set_trace()
        # print("updated_cluster_num", updated_cluster_num)
        # print("self.total_additional_answers_count", self.total_additional_answers_count)
        # print(self.additional_cluster_to_answer.items())
        # sample the answers according to updated_cluster_num
        sampled_answers = []
        sampled_answers_clusters = []
        for k, v in self.additional_cluster_to_answer.items():
            cluster_wise_expanded_answer_list = []
            for ans in v['answers']:
                ans_count = self.additional_answer_to_cluster[ans]['count']
                for ff in range(ans_count):
                    cluster_wise_expanded_answer_list.append(ans)
            ## Source of output randomness
            sample_s_indices = sample_indices(updated_cluster_num[k], 0, v['count'])
            for ind in sample_s_indices:
                if ind < len(cluster_wise_expanded_answer_list):
                    sampled_answers.append(cluster_wise_expanded_answer_list[ind])
                    sampled_answers_clusters.append(k)
            # import pdb; pdb.set_trace()

        assert len(sampled_answers) == len(sampled_answers_clusters) <= sum(list(updated_cluster_num.values()))
        sampled_answer_frequency = dict(Counter(sampled_answers))
        # print('Sample {} percent of the answers, {} number of answers'.format(ratio, len(sampled_answers)))
        
        demo = []
        for i in range(len(sampled_answers)):
            demo.append((sampled_answers[i], sampled_answers_clusters[i]))
        # print("sampled_answers:")
        # print(demo)
        # import pdb; pdb.set_trace()
        if do_plot:
            gt_dist = list(ground_truth_distribution.values())
            total_answers = sum(updated_cluster_num.values())
            actual_distribution = [value / total_answers for value in updated_cluster_num.values()]
            # Example lists of probabilities
            list1 = gt_dist
            list2 = sample_category_distribution
            list3 = actual_distribution
            # Setting up the x-coordinates
            x = np.arange(len(list1))  # the label locations
            # Additional details for plotting
            plot_details = {
                "x_labels": list(ground_truth_distribution.keys()),
                "title": "Comparison of Three Probability Lists",
                "xlabel": "Categories",
                "ylabel": "Probabilities",
                "legend": ["gt_dist", "intended_dist", "actual_dist"],
                "file_path": "/home/qic69/Desktop/CFC/commonsense-validation-api/src/commonsense_validation_api_copy/fig/bar_charts_" + sampling_type + "/bar_missing_answer_" + str(sampling_step) + ".png"
            }
            return sampled_answers, sampled_answers_clusters, sampled_answer_frequency, sample_category_distribution, demo, list1, list2, list3, x, plot_details
        else:
            return sampled_answers, sampled_answers_clusters, sampled_answer_frequency, sample_category_distribution, demo
    
    def human_kl_score(
            self,
            sampled_answers,
            sampled_answers_clusters,
            unordered_ground_truth_distribution,
            num_of_clusters
    ):

        probabilities_H_P = human_assignment_probability(
            sampled_answers,
            sampled_answers_clusters,
            self.ground_truth_cluster_to_ans,
            num_of_clusters,
            answer_assignment=False,
            is_evaluation_set=True
        )
        prediction_answer_distribution = OrderedDict(sorted(probabilities_H_P.items()))
        ordered_ground_truth_distribution = OrderedDict(sorted(unordered_ground_truth_distribution.items()))

        hm_kl_score = kl_div(list(ordered_ground_truth_distribution.values()), list(prediction_answer_distribution.values()))
        return np.sum(hm_kl_score), probabilities_H_P


    def automatic_kl_score(
            self,
            assignment_answer_distribution_func,
            prediction_answer_frequency,
            n_component: float, 
            probabilities_C_G
    ):
        """
        Computes the KL score using clustering method when GT and a list of NOT annotated prediction answers are given.
        """

        # assign prediction answers to ground truth clusters
        probabilities_auto_P = assignment_answer_distribution_func(
            self.question_number,
            self.targets_json,
            self.true_answer_vectors,
            self.pred_answer_vectors,
            prediction_answer_frequency
        )
        
        ordered_probabilities_C_G = OrderedDict(sorted(probabilities_C_G.items()))
        ordered_probabilities_auto_P = OrderedDict(sorted(probabilities_auto_P.items()))

        #kl input order is important, this calculate how far the prediction distribution to the ground truth distribution.
        auto_kl_score = kl_div(list(ordered_probabilities_C_G.values()), list(ordered_probabilities_auto_P.values()))
        return np.sum(auto_kl_score), ordered_probabilities_auto_P

    def get_answer_assignment_distribution(self, assignment_answer_distribution):
        pass
