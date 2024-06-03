import protoqa_evaluator
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

class QuestionAndAnswerClusters(NamedTuple):
    question_id: str
    question: str
    answer_clusters: Dict[FrozenSet[str], int]

qid = "r3q16"
auto_clusters ={'r3q16': QuestionAndAnswerClusters(question_id='r3q16', question='a helicopter being used to tackle disaster. who will use helicopters to tackle the disaster?', 
                                                   answer_clusters={frozenset({'helicopter pilot', 'driver', 'an airlight pilot', 'air ambulence pilots', 'military pilot', 'the pilot', 'pilot', 'helicopter relief pilot'}): 48, 
                                                                    frozenset({'emergency services', 'commander', 'control tower', 'the military', 'disaster management team', 'red cross', 'emergency medical technicians and firefighters and helicopter pilots', 'emergency department'}): 9, 
                                                                    frozenset({'emergency worker', 'firemen', 'a search and rescue team', 'firefighter', 'the soldiers', 'firefighters'}): 8,
                                                                    frozenset({'person'}): 3, 
                                                                    frozenset({'crew'}): 2,
                                                                    frozenset({'news reporter'}): 2})
                                                                    }
converted_sampled_answers_1, converted_sampled_answers_2, converted_sampled_answers_3, converted_sampled_answers_4 = {
        "r3q16": [
            "pilot",
            "a team of firefighters",
            "person",
            "air ambulence pilots",
            "the pilot",
            "helicopter pilot",
            "government",
            "the soldiers",
            "military pilot",
            "federal agencies",
            "emergency services",
            "emergency medical technicians and firefighters and helicopter pilots",
            "commander",
            "a search and rescue team",
            "emergency worker",
            "firemen",
            "first responders",
            "news reporter"
        ]
    }, {
        "r3q16": [
            "pilot",
            "the pilot",
            "government",
            "an airlight pilot",
            "emergency medical technicians and firefighters and helicopter pilots",
            "a search and rescue team",
            "crew",
            "person",
            "military pilot",
            "helicopter pilot",
            "the military",
            "federal government",
            "disaster management team",
            "the soldiers",
            "a team of firefighters",
            "firemen",
            "firefighters",
            "rangers",
            "news reporter"
        ]
    },{
        "r3q16": [
            "pilot",
            "the pilot",
            "air ambulence pilots",
            "military",
            "emergency medical technicians and firefighters and helicopter pilots",
            "security forces",
            "state police",
            "red cross",
            "the government",
            "a search and rescue team",
            "a team of firefighters",
            "firefighters",
            "russian army",
            "crew",
            "person"
        ]
    },{
        "r3q16": [
            "pilot",
            "helicopter pilot",
            "person",
            "national security",
            "firefighter",
            "firefighters",
            "the pilot",
            "an airlight pilot",
            "the military",
            "federal agencies",
            "red cross",
            "federal government",
            "a search and rescue team",
            "people"
        ]
    }

converted_sampled_answers_8, converted_sampled_answers_9 = {
        "r3q16": [
            "pilot",
            "the military",
            "national emergency response team",
            "a team of firefighters",
            "person",
            "military pilot",
            "the pilot",
            "helicopter pilot",
            "the government",
            "state police",
            "security forces",
            "firemen",
            "the soldiers",
            "emergency worker",
            "a search and rescue team",
            "firefighter",
            "family members",
            "crew",
            "news reporter"
        ]
    },{
        "r3q16": [
            "pilot",
            "helicopter pilot",
            "the pilot",
            "military pilot",
            "the government",
            "a team of firefighters",
            "person",
            "the military",
            "disaster management team",
            "national security",
            "state police",
            "firefighter",
            "firefighters",
            "a search and rescue team",
            "motorcyclists",
            "crew"
        ]
    }    

print("For sampling step 1: ")
detail, auto_sampling_protoQA_score = protoqa_evaluator.evaluation.multiple_evals(wordnet_all_eval_funcs, auto_clusters, answers_dict=converted_sampled_answers_1)
print(detail["Max Answers - 10"])
auto_sampling_protoQA_score = 1 - auto_sampling_protoQA_score[4]
print(auto_sampling_protoQA_score)


print("For sampling step 2: ")
detail, auto_sampling_protoQA_score = protoqa_evaluator.evaluation.multiple_evals(wordnet_all_eval_funcs, auto_clusters, answers_dict=converted_sampled_answers_2)
print(detail["Max Answers - 10"])
auto_sampling_protoQA_score = 1 - auto_sampling_protoQA_score[4]
print(auto_sampling_protoQA_score)


print("For sampling step3 : ")
detail, auto_sampling_protoQA_score = protoqa_evaluator.evaluation.multiple_evals(wordnet_all_eval_funcs, auto_clusters, answers_dict=converted_sampled_answers_3)
print(detail["Max Answers - 10"])
auto_sampling_protoQA_score = 1 - auto_sampling_protoQA_score[4]
print(auto_sampling_protoQA_score)


print("For sampling step4 : ")
detail, auto_sampling_protoQA_score = protoqa_evaluator.evaluation.multiple_evals(wordnet_all_eval_funcs, auto_clusters, answers_dict=converted_sampled_answers_4)
print(detail["Max Answers - 10"])
auto_sampling_protoQA_score = 1 - auto_sampling_protoQA_score[4]
print(auto_sampling_protoQA_score)

print("For sampling step8 : ")
detail, auto_sampling_protoQA_score = protoqa_evaluator.evaluation.multiple_evals(wordnet_all_eval_funcs, auto_clusters, answers_dict=converted_sampled_answers_8)
print(detail["Max Answers - 10"])
auto_sampling_protoQA_score = 1 - auto_sampling_protoQA_score[4]
print(auto_sampling_protoQA_score)


print("For sampling step9 : ")
detail, auto_sampling_protoQA_score = protoqa_evaluator.evaluation.multiple_evals(wordnet_all_eval_funcs, auto_clusters, answers_dict=converted_sampled_answers_9)
print(detail["Max Answers - 10"])
auto_sampling_protoQA_score = 1 - auto_sampling_protoQA_score[4]
print(auto_sampling_protoQA_score)