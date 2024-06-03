# this is for gypsum
import wandb
import argparse
from commonsense_validation_api_copy.common_evaluations import multi_evals
from commonsense_validation_api_copy.data_processing import load_question_answer_clusters_from_jsonl_file
from commonsense_validation_api_copy.data_processing import load_predictions_from_jsonl_file
import json
import datetime
import time

def main(config=None):
    with open('./config/normal_args.json', 'r') as f:
        args_dict = json.load(f)
    args = argparse.Namespace(**args_dict)
    # current_datetime = datetime.datetime.now()
    # formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M")
    # unique_run_id = str(args.wandb_config) + str(formatted_datetime)
    if args.tune_parameters:
        with wandb.init(project="CFC", resume="allow", config=config):
            config = wandb.config
            print("*"*20)
            print("Using {} - {} - {} - {} - {}".format(
                config.sample_method, config.clustering_algo, config.similarity_function, config.assignment, config.starting_component,))
            print(args.target)
            print(args.prediction)
            print(args.embedding)
            function_name = get_function_name(config.clustering_algo, config.similarity_function)
            targets = load_question_answer_clusters_from_jsonl_file(args.target) # len(targets)=52
            all_predictions = load_predictions_from_jsonl_file(args.prediction) #all_predictions['annotated_prediction_data'] & ['model_answers']
            if "annotated_prediction_data" in all_predictions:
                annotated_prediction_answers = all_predictions["annotated_prediction_data"]
                predictions = {k: all_predictions["annotated_prediction_data"][k] for k in targets}
            else:
                annotated_prediction_answers = None
                predictions = {k: all_predictions["model_answers"][k] for k in targets}

            print("*" * 20)
            print("Starting Eval")
            print("Calculating protoQA correlation score: ", args.do_protoQA)
            average_score, ranking_scores_per_component = multi_evals(
                eval_type=function_name,
                question_data=targets,
                prediction_data=predictions,
                embedding=args.embedding,
                annotated_prediction_answers=annotated_prediction_answers,
                clustering_algo=config.clustering_algo,
                num_sampling_steps=config.num_sampling_steps,
                save_files=args.save_files,
                n_component=config.starting_component,
                debug=args.debug,
                hac_linkage_function=config.hac_linkage_function,
                sample_function=config.sample_method,
                assignment=config.assignment,
                do_protoQA=args.do_protoQA
            )

            if args.do_protoQA:
                wandb.log({"Correlation": ranking_scores_per_component[config.starting_component]['average'][0], 
                       'Correlation_protoQA': ranking_scores_per_component[config.starting_component]['protoQA_avg_overrall'][0]})
            else:
                wandb.log({"Correlation": ranking_scores_per_component[config.starting_component]['average'][0]})
    else:
        s = time.time()
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
        
        average_score, ranking_scores_per_component = multi_evals(
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
            assignment=args.assignment,
            do_protoQA=args.do_protoQA
        )
        print(average_score)
       
        if args.log:
            log_file = "/home/qic69/Desktop/CFC/commonsense-validation-api/src/commonsense_validation_api_copy/three_new_sampling_log.csv"
            
            # Construct the run label
            run_label = "{} - {} - {} - {} - {}".format(
                args.sample_method, args.clustering_algo, args.similarity_function, args.assignment, args.starting_component
            )
            
            # Retrieve the score
            score = ranking_scores_per_component[args.starting_component]['average'][0]
            
            # Retrieve the protoQA score
            if args.do_protoQA:
                protoQA_score = ranking_scores_per_component[args.starting_component]['protoQA_avg_overrall'][0]
            else:
                protoQA_score = -1

            # Write to the log file
            with open(log_file, 'a') as file:  # 'a' mode appends to the file if it exists, or creates a new file if it doesn't
                file.write(f"{run_label}, {score}, {protoQA_score}\n")
        print(f"Total time {s-time.time()}")

def get_function_name(clustering_algo: str, similarity_function: str):
    return clustering_algo+"_cluster_and_"+similarity_function
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--target", default="/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_targets.jsonl", type=str)
    parser.add_argument("--prediction", default="/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_annotated_predictions.json", type=str)
    parser.add_argument("--embedding", default="/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_embeddings.jsonl", type=str)


    # parser.add_argument("--target", default="/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/20220715_ground_truth_targets.jsonl", type=str)
    # parser.add_argument("--prediction", default="/home/qic69/Desktop/CFC/CFC_evaluator_data/NLP_project/gpt3.5_original__prediction.json", type=str)
    # parser.add_argument("--embedding", default="/home/qic69/Desktop/CFC/CFC_evaluator_data/NLP_project/gpt3.5_original_embeddings.jsonl", type=str)

    # parser.add_argument("--target", default="/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets copy.jsonl", type=str)
    # parser.add_argument("--prediction", default="/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_annotated_predictions.json", type=str)
    # parser.add_argument("--embedding", default="/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_annotated_embeddings.jsonl", type=str)

    parser.add_argument("--similarity_function",  default="wordnet", type=str, help="Choose from: fasttext and wordnet")
    # Check why wordnet doesn't
    parser.add_argument("--clustering_algo", default="hac", type=str, help="Choose from: gmeans, xmeans, hac and gt")
    parser.add_argument("--num_sampling_steps", default=50, type=int)
    parser.add_argument("--save_files", default=False, type=bool, )
    parser.add_argument("--starting_component", default=0.49, type=float,
                        help="Different cluster algorithm has different components."
                                                                "xmeans: 7-14 (1)"
                                                                "gmeans: 1-7 (1)"
                                                                "hac: 0.2-0.5 (0.05)")
    parser.add_argument("--debug", default=False, type=bool, help="when in debug mode, the randomness is fixed.")
    parser.add_argument("--hac_linkage_function", default="single", type=str,
                        help="Choose from single, complete, average, weighted, centroid, median, ward")
    parser.add_argument("--sample_method", default='diverse', type=str, help="choose from diverse, centered or vanila")
    parser.add_argument("--assignment", default="cosine", type=str, help="if use fasttext, whether to use cosine or"
                                                                           "linear regression to do the assignment.")
    parser.add_argument("-tune", "--tune_parameters", action="store_true", help="If we are tuning parameters.")
    parser.add_argument("--wandb_config", default='./config/exp1_r1.json', type=str)
    parser.add_argument("-dq", "--do_protoQA", action="store_true", help="If we are calculating protoQA correlation.")
    parser.add_argument("-log", action="store_true")
    args = parser.parse_args()
    args_dict = vars(args)

    # Save the dictionary to a local file
    with open('./config/normal_args.json', 'w') as f:
        json.dump(args_dict, f, indent=4)

    if args.tune_parameters:
        sweep_config = {
        'method': 'bayes'
        }
        metric = {
            'name': 'Correlation',
            'goal': 'maximize'   
            }
        
        if args.do_protoQA:
            sweep_config['name'] = str(args.wandb_config) + " - ProtoQA"
        else:
            sweep_config['name'] = str(args.wandb_config)
        # sweep_config['name'] = "test"
        sweep_config['metric'] = metric
        with open(args.wandb_config, 'r') as file:
            parameters_dict = json.load(file)
        sweep_config['parameters'] = parameters_dict
        sweep_id = wandb.sweep(sweep_config, project="CFC")
        wandb.agent(sweep_id, main, count=100)
    else:
        main()
    
