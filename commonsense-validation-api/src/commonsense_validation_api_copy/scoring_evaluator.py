from commonsense_validation_api.data_processing import QuestionAndAnswerClusters, default_string_preprocessing
from commonsense_validation_api.scoring import *
import numpy as np
from typing import *
import time

def evaluate(
        evaluation_func: Callable,
        question_data: Dict[str, QuestionAndAnswerClusters],
        answers_dict: Dict[str, List[str]],
        data_preprocessing: Optional[Callable] = None,
        optimal_ranking: bool = False,
) -> Dict[str, float]:
    scores = dict()
    for qid, pred_answers in answers_dict.items():
        true_q = question_data[qid]
        if data_preprocessing is not None:
            true_q, pred_answers = data_preprocessing(true_q, answers_dict)
        true_answers = true_q.answer_clusters.copy()
        scores[qid] = evaluation_func(
            pred_answers,
            true_answers,
            question_string=true_q.question,
            optimal_ranking=optimal_ranking,
        )
    return scores


class EvalResult(NamedTuple):
    score: float
    score_matrix: np.ndarray
    answer_assignment: dict

    def __eq__(self, other):
        return (
                self.score == other.score
                and (self.score_matrix == other.score_matrix).all()
                and self.answer_assignment == other.answer_assignment
        )


def general_eval(
        pred_answers,
        true_answers,
        *,
        max_pred_answers: Optional[int] = None,
        max_incorrect: Optional[int] = None,
        string_preprocessing: Callable = default_string_preprocessing,
        question_string: str = "question string",
        score_func: Callable = exact_match,
        cluster_score_func: Callable = cluster_score,
        cluster_reduction_func: Callable = np.max,
        score_matrix_transformation: Optional[Callable] = None,
        assign_cluster_scores: bool = True,
        calc_oracle_score: bool = False,
        optimal_ranking: bool = False,
) -> EvalResult:
    if max_pred_answers is not None and not optimal_ranking:
        pred_answers = pred_answers[:max_pred_answers]
    pred_answers = [string_preprocessing(pred_answer) for pred_answer in pred_answers]
    # print(len(pred_answers), len(true_answers))
    # s = time.time()
    # print(pred_answers)
    # print(true_answers)
    score_matrix = cluster_score_func(
        pred_answers,
        true_answers,
        question_string=question_string,
        score_func=score_func,
        cluster_reduction_func=cluster_reduction_func,
    )

    
    # print("Time used: ", time.time() - s)
    # score_matrix has values in [0,1] at this point
    if score_matrix_transformation is not None:
        score_matrix = score_matrix_transformation(score_matrix)
    if max_incorrect is not None and not optimal_ranking:
        score_matrix = limit_total_wrong(score_matrix, max_incorrect)
    if assign_cluster_scores:
        score_matrix *= np.array(list(true_answers.values()))[None]
    # #score, row_ind, col_ind = get_optimal_score(score_matrix)

    # if optimal_ranking:
    #     reward_and_ind = [
    #         (score_matrix[row_ind[z], col_ind[z]], row_ind[z], col_ind[z])
    #         for z in range(len(row_ind))
    #     ]
    #     sorted_by_reward = sorted(reward_and_ind, key=lambda z: z[0], reverse=True)
    #     _, row_ind, col_ind = zip(*sorted_by_reward)
    #     row_ind = np.array(row_ind)
    #     col_ind = np.array(col_ind)
    #     if max_pred_answers is not None:
    #         row_ind = row_ind[:max_pred_answers]
    #         col_ind = col_ind[:max_pred_answers]
    #     if max_incorrect is not None:
    #         for i in range(len(row_ind)):
    #             if score_matrix[row_ind[i], col_ind[i]] == 0:
    #                 break
    #         row_ind = row_ind[:i]
    #         col_ind = col_ind[:i]
    #     score = score_matrix[row_ind, col_ind].sum()

    answer_assignment = dict()

    def get_cluster(c):
        cluster_names = list(true_answers.values())
        answers = ""
        for x in c:
            answers += str(cluster_names[x][1]) + ","
        return answers.strip(",")

    row_ind, col_ind = score_matrix.shape
    for r in range(row_ind):
        max_score = np.max(score_matrix[r])
        if max_score == 0:
            answer_assignment[pred_answers[r]] = 'wrong'
        else:
            max_score_clusters = np.argwhere(
                score_matrix[r] == np.max(score_matrix[r])).flatten().tolist()
            answer_assignment[pred_answers[r]] = get_cluster(max_score_clusters)

    return EvalResult(
        score=0, score_matrix=score_matrix, answer_assignment=answer_assignment
    )
