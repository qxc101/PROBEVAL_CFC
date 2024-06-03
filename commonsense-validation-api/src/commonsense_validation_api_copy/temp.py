# this is for gypsum
import wandb
import argparse
from commonsense_validation_api.common_evaluations import multi_evals
from commonsense_validation_api.data_processing import load_question_answer_clusters_from_jsonl_file
from commonsense_validation_api.data_processing import load_predictions_from_jsonl_file

def get_function_name(clustering_algo: str, similarity_function: str):
    return clustering_algo+"_cluster_and_"+similarity_function

def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--target", default="/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_targets.jsonl", type=str)
    # parser.add_argument("--prediction", default="/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_annotated_predictions.json", type=str)
    # parser.add_argument("--embedding", default="/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_embeddings.jsonl", type=str)

    parser.add_argument("--target", default="/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/20220715_ground_truth_targets.jsonl", type=str)
    parser.add_argument("--prediction", default="/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/20220715_ground_truth_annotated_predictions.json", type=str)
    parser.add_argument("--embedding", default="/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/20220715_ground_truth_embeddings_100.jsonl", type=str)

    # parser.add_argument("--target", default="/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets copy.jsonl", type=str)
    # parser.add_argument("--prediction", default="/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_predictions copy.json", type=str)
    # parser.add_argument("--embedding", default="/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_embeddings_100 copy.jsonl", type=str)

    parser.add_argument("--similarity_function",  default="wordnet", type=str, help="Choose from: fasttext and wordnet")
    # Check why wordnet doesn't
    parser.add_argument("--clustering_algo", default="hac", type=str, help="Choose from: gmeans, xmeans, hac and gt")
    parser.add_argument("--num_sampling_steps", default=50, type=int)
    parser.add_argument("--save_files", default=False, type=bool, )
    parser.add_argument("--starting_component", default=0.4944579181731247, type=float,
                        help="Different cluster algorithm has different components."
                                                                "xmeans: 7-14 (1)"
                                                                "gmeans: 1-7 (1)"
                                                                "hac: 0.2-0.5 (0.5)")
    parser.add_argument("--debug", default=False, type=bool, help="when in debug mode, the randomness is fixed.")
    parser.add_argument("--hac_linkage_function", default="single", type=str,
                        help="Choose from single, complete, average, weighted, centroid, median, ward")
    parser.add_argument("--sample_method", default='centered', type=str, help="choose from diverse, centered or vanila")
    parser.add_argument("--assignment", default="cosine", type=str, help="if use fasttext, whether to use cosine or"
                                                                           "linear regression to do the assignment.")
    args = parser.parse_args()
   
    print("*"*20)
    print("Using {} - {} - {} - {} - {}".format(
        args.sample_method, args.clustering_algo, args.similarity_function, args.assignment, args.starting_component,))
    print(args.target)
    print(args.prediction)
    print(args.embedding)
    function_name = get_function_name(args.clustering_algo, args.similarity_function)
    targets = load_question_answer_clusters_from_jsonl_file(args.target) # len(targets)=52
    all_predictions = load_predictions_from_jsonl_file(args.prediction) #all_predictions['annotated_prediction_data'] & ['model_answers']

    if "annotated_prediction_data" in all_predictions:
        annotated_prediction_answers = all_predictions["annotated_prediction_data"]
        predictions = {k: all_predictions["annotated_prediction_data"][k] for k in targets}
    else:
        annotated_prediction_answers = None
        predictions = {k: all_predictions["model_answers"][k] for k in targets}
    
    average_score = multi_evals(
        eval_type=function_name,
        question_data=targets,
        prediction_data=predictions,
        embedding=args.embedding,
        annotated_prediction_answers=annotated_prediction_answers,
        clustering_algo=args.clustering_algo,
        num_sampling_steps=args.num_sampling_steps,
        save_files=args.save_files,
        n_component=args.starting_component,
        debug=args.debug,
        hac_linkage_function=args.hac_linkage_function,
        sample_function=args.sample_method,
        assignment=args.assignment
    )
    

if __name__ == '__main__':
    
    main()
    # main(sweep_config)

