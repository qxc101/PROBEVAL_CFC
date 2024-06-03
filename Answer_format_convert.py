import json
import argparse
from pprint import pprint
import fasttext
import fasttext.util
import time

def CFC2ProtoQA(input_path_prediction, input_path_embeddings, output_path):
    # Step 1: Load Data
    with open(input_path_prediction, 'r') as file:
        prediction_data = json.load(file)
    
    with open(input_path_embeddings, 'r') as file:
        embedding_data = json.load(file)
    
    # Step 2: Extract and Organize Data
    output_data = []
    
    for question_id, item in embedding_data.items():
        # Extract ranked answers from the prediction data
        ranked_answers = prediction_data["model_ranked_answers"].get(question_id, [])
        
        # Combine ranked answers into a single item
        data_item = {
            question_id: ranked_answers
        }
        
        output_data.append(data_item)

    # Step 3: Save Data
    with open(output_path, 'w') as outfile:
        for item in output_data:
            json.dump(item, outfile)
            outfile.write('\n')

# def CFC2ProtoQA(input_path_prediction, input_path_embeddings, output_path):
#     # Step 1: Load Data
#     with open(input_path_prediction, 'r') as file:
#         prediction_data = json.load(file)
    
#     with open(input_path_embeddings, 'r') as file:
#         embedding_data = json.load(file)
    
#     # Step 2: Extract and Organize Data
#     output_data = []
    
#     for question_id, item in embedding_data.items():
#         # Extract ranked answers from the prediction data
#         ranked_answers = prediction_data["model_ranked_answers"].get(question_id, [])
        
#         # Combine ranked answers into a single item
#         data_item = {
#             question_id: ranked_answers
#         }
        
#         output_data.append(data_item)

#     # Step 3: Save Data
#     with open(output_path, 'w') as outfile:
#         for item in output_data:
#             json.dump(item, outfile)
#             outfile.write('\n')

def average_embedding(phrase, model):
    words = phrase.split()
    # print(phrase)
    embeddings = [model[word] for word in words]
    avg_embedding = sum(embeddings) / len(embeddings)

    return avg_embedding


def ProtoQA2CFC(input_path, output_path_prediction, output_path_embeddings, reference_file, question2id=False, protoqa_data=False):
    # Step 0: Load your FastText model (replace 'path/to/model' with the actual path to your model file)
    ft_model = fasttext.load_model('cc.en.300.bin')
    ft_model = fasttext.util.reduce_model(ft_model, 100)
    # data["metadata"]["id"]
    # Step 1: Load Data
    with open(input_path, 'r') as file:
        data = [json.loads(line) for line in file]

    # Step 2: Calculate Counts & Step 3: Rank Answers
    model_answers = {}
    model_ranked_answers = {}
    model_answers_with_count = {}
    x = 0
    for item in data:
        if question2id:
            qid = "r4q" + str(x + 1)
        for question_id, answers in item.items():
            if question2id:
                question_id = qid
            answers = [answer.lower().strip() for answer in answers if answer.strip() != ""]
            unique_answers = list(set(answers))  # Get unique answers
            answer_counts = [[answer, answers.count(answer)] for answer in unique_answers]  # Get counts
            ranked_answers = [answer for answer, count in
                              sorted(answer_counts, key=lambda x: x[1], reverse=True)]  # Get ranked answers

            model_answers[question_id] = unique_answers
            model_ranked_answers[question_id] = ranked_answers
            model_answers_with_count[question_id] = sorted(answer_counts, key=lambda x: x[1], reverse=True)
        x += 1

    # Step 4: Format Data
    output_data = {
        "model_answers": model_answers,
        "model_ranked_answers": model_ranked_answers,
        "model_answers_with_count": model_answers_with_count,
    }

    # Step 5: Save Data
    if protoqa_data:
        pred_dict = output_data
        with open("/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_targets.jsonl", 'r') as f:
            for l in f.readlines():
                t_dict = json.loads(l.strip())
                k = t_dict['question']['normalized']
                if k in pred_dict['model_answers'].keys():
                    pred_dict['model_answers'][t_dict['metadata']['id']] = pred_dict['model_answers'][k]
                    del pred_dict['model_answers'][k]

        # print(pred_dict)
        with open(output_path_prediction, 'w') as f:
            json.dump(pred_dict, f)
    else:
        with open(output_path_prediction, 'w') as outfile:
            json.dump(output_data, outfile)
    
    # with open(output_path_prediction, 'w') as outfile:
    #     json.dump(output_data, outfile, ensure_ascii=False, indent=4)
    # Step 6: Load Reference Data (replace 'path/to/reference_file.jsonl' with the actual path to your reference file)
    reference_data = {}
    with open(reference_file, 'r') as ref_file:
        reference_data = json.load(ref_file) 
        # for line in ref_file:
        #     try:
        #         data = json.loads(line.strip().replace("'", ""))
        #     except:
        #     reference_data.update(data)
            
    
    output_data_embeddings = {}
    
    for question_id, item in reference_data.items():
        print(question_id)
        # Map true_answers, raw_answers, etc. directly from the reference data
        # print("--------------------------------------")
        # print(item['question_string'])
        # print(item['true_answers'])
        # print(model_answers.get(question_id, []),)
        # print(item['pred_answers'])
        output_data_embeddings[question_id] = {
            'true_answers': item['true_answers'],
            'raw_answers': item['raw_answers'],
            'question_string': item['question_string'],
            'true_answer_vectors': {answer: average_embedding(answer, ft_model).tolist() for answer in item['true_answers']},
            
            # Get predicted answers from the model answers we computed earlier
            'pred_answers': model_answers.get(question_id, []),
            
            # Calculate pred_answer_vectors using FastText model
            'pred_answer_vectors': {answer: average_embedding(answer, ft_model).tolist() for answer in model_answers.get(question_id, [])}
        }
        
        
    # Step 7: Save Data (Embeddings)
    # with open(output_path_embeddings, 'w') as outfile:
    #     for question_id, data in output_data_embeddings.items():
    #         json.dump({question_id: data}, outfile)
    #         outfile.write('\n')
    # Step 7: Save Data (Embeddings)
    with open(output_path_embeddings, 'w') as outfile:
        json.dump(output_data_embeddings, outfile, indent=4)
    print("Saving file to:", output_path_embeddings)

def ProtoQA_annotated_dev_2_CFC():
    ft_model = fasttext.load_model('cc.en.300.bin')
    ft_model = fasttext.util.reduce_model(ft_model, 100)

    # Load protoqa_dev_annotated_targets.jsonl
    with open("/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_targets.jsonl", 'r') as f:
        targets_data = [json.loads(line) for line in f]

    # Load annotated_protoqa_dev_predictions.jsonl
    with open("/home/qic69/Desktop/CFC/ProtoQA_evaluator_data/annotated_protoqa_dev_predictions.jsonl", 'r') as f:
        predictions_data = [json.loads(line) for line in f]

    # Construct the desired structure
    output_data = {}
    output_data_prediction = {}
    output_data_prediction["model_answers"] = {}
    for target, prediction in zip(targets_data, predictions_data):
        question_id = target["metadata"]["id"]
        
        # Extract true answers and raw answers
        true_answers = list(target["answers"]["raw"].keys())
        print(len(true_answers))
        raw_answers = target["answers"]["raw"]
        
        # Extract prediction answers
        pred_answers = prediction[question_id]

        # Generate true and prediction answer vectors
        true_answer_vectors = {answer: average_embedding(answer, ft_model).tolist() for answer in true_answers}
        pred_answer_vectors = {answer: average_embedding(answer, ft_model).tolist() for answer in pred_answers}
        

        output_data[question_id] = {
            "true_answers": true_answers,
            "raw_answers": raw_answers,
            "pred_answers": pred_answers,
            "question_string": target["question"]["normalized"],
            "true_answer_vectors": true_answer_vectors,
            "pred_answer_vectors": pred_answer_vectors
        }
        output_data_prediction["model_answers"][question_id] = pred_answers
        print(len(set(pred_answers)), len(pred_answer_vectors))
        print(len(set(true_answers)), len(raw_answers), len(true_answer_vectors))

    # with open("/home/qic69/Desktop/CFC/commonsense-validation-api/data/annotated_predictions.json", 'r') as f:
    #     annotation_data = [json.loads(line) for line in f]
    with open("/home/qic69/Desktop/CFC/commonsense-validation-api/data/annotated_predictions.json", 'r') as f:
        annotation_content = json.load(f)
        annotation_data = annotation_content.get('annotated_prediction_data', {})
    output_data_prediction["annotated_prediction_data"] = annotation_data
    
    # Save the constructed data to a JSONL file
    with open("/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_embeddings.jsonl", 'w') as f:
        json.dump(output_data, f, indent=4)
    with open("/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_annotated_predictions.json", 'w') as f:
        json.dump(output_data_prediction, f, indent=4)  # The indent parameter makes the JSON pretty-printed

    print("Data saved to output.jsonl")



# Function to process each line and extract relevant information
def process_line(line):
    data = line
    raw_answers = data['answers']['raw']
    clusters = data['answers']['clusters']
    answer_cluster_map = {}

    # Map each answer to its respective cluster
    for cluster_id, cluster_data in clusters.items():
        for answer in cluster_data['answers']:
            answer_cluster_map[answer] = cluster_id

    # Now create a dictionary with answers, clusters, and counts
    json_content = {}
    for answer, count in raw_answers.items():
        cluster_id = answer_cluster_map.get(answer, "unknown")
        json_content[answer] = {"cluster": cluster_id, "count": count}
    json_content["wrong"] = {"cluster": "wrong", "count": 1}
    return json_content


def CFC_dev_2_CFC_dev_annotated():
    ft_model = fasttext.load_model('cc.en.300.bin')
    ft_model = fasttext.util.reduce_model(ft_model, 100)

    # Load protoqa_dev_annotated_targets.jsonl
    with open("/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets copy.jsonl", 'r') as f:
        targets_data = [json.loads(line) for line in f]

    # Load annotated_protoqa_dev_predictions.jsonl
    with open("/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets copy.jsonl", 'r') as f:
        predictions_data = [json.loads(line) for line in f]

    # Construct the desired structure
    output_data = {}
    output_data_prediction = {}
    output_data_prediction["model_answers"] = {}
    output_data_prediction["annotated_prediction_data"] = {}
    for target, prediction in zip(targets_data, predictions_data):
        question_id = target["metadata"]["id"]
        
        # Extract true answers and raw answers
        true_answers = list(target["answers"]["raw"].keys())
        print(len(true_answers))
        raw_answers = target["answers"]["raw"]
        # print(raw_answers)
        # raw_answers.append("wrong")
        # Extract prediction answers
        pred_answers = list(target["answers"]["raw"].keys())
        print(pred_answers)
        pred_answers.append("wrong")
        # Generate true and prediction answer vectors
        true_answer_vectors = {answer: average_embedding(answer, ft_model).tolist() for answer in true_answers}
        pred_answer_vectors = {answer: average_embedding(answer, ft_model).tolist() for answer in pred_answers}
        

        output_data[question_id] = {
            "true_answers": true_answers,
            "raw_answers": raw_answers,
            "pred_answers": pred_answers,
            "question_string": target["question"]["normalized"],
            "true_answer_vectors": true_answer_vectors,
            "pred_answer_vectors": pred_answer_vectors
        }
        output_data_prediction["model_answers"][question_id] = pred_answers
        print(len(set(pred_answers)), len(pred_answer_vectors))
        print(len(set(true_answers)), len(raw_answers), len(true_answer_vectors))

        output_data_prediction["annotated_prediction_data"][question_id] = process_line(target)


    # with open("/home/qic69/Desktop/CFC/commonsense-validation-api/data/annotated_predictions.json", 'r') as f:
    #     annotation_data = [json.loads(line) for line in f]
    # with open("/home/qic69/Desktop/CFC/commonsense-validation-api/data/annotated_predictions.json", 'r') as f:
    #     annotation_content = json.load(f)
    #     annotation_data = annotation_content.get('annotated_prediction_data', {})
    # output_data_prediction["annotated_prediction_data"] = annotation_data
    
    # Save the constructed data to a JSONL file
    with open("/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_annotated_embeddings.jsonl", 'w') as f:
        json.dump(output_data, f, indent=4)
    with open("/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_annotated_predictions.json", 'w') as f:
        json.dump(output_data_prediction, f, indent=4)  # The indent parameter makes the JSON pretty-printed

    print("Data saved to output.jsonl")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A simple script.")
    parser.add_argument('-pf', '--protoQA_file', help='path containing ground truth answer clusters',
                        default="/home/qic69/Desktop/CFC/ProtoQA_evaluator_data/NLP_project/exp3_caption_gpt-3.5-turbo.jsonl")
    parser.add_argument('-cfp', '--cfc_prediction', help='Name of gpt filter model',
                        default="/home/qic69/Desktop/CFC/CFC_evaluator_data/NLP_project/gpt3.5_exp3_caption_prediction.json")
    parser.add_argument('-cfe', '--cfc_embedding', help='Name of gpt filter model',
                        default="/home/qic69/Desktop/CFC/CFC_evaluator_data/NLP_project/gpt3.5_exp3_caption_embeddings.jsonl")
    parser.add_argument('-protoqa_dev2cfc', action='store_true')
    parser.add_argument('-protoqa2cfc', action='store_true')
    parser.add_argument('-cfc2protoqa', action='store_true')
    parser.add_argument('-protoqa_data', action='store_true')
    parser.add_argument('-file_selection', action='store_true')
    args = parser.parse_args()
    
    s = time.time()
    with open("CFC_evaluator_data/GPT2_Predictions/0715modelft_grad8lr5e-06sample200_temp1.0_embeddings_100.jsonl", 'r') as file:
        lines = file.readlines()
        data = [json.loads(line) for line in lines]

    import os

    # Directories
    predictions_dir = '/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/predictions'
    embeddings_dir = '/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/prediction_embeddings'

    # Get list of files in each directory
    prediction_files = os.listdir(predictions_dir)
    embedding_files = os.listdir(embeddings_dir)

    # Dictionary to store the matching
    matches = {}

    # Iterate over prediction files to find corresponding embedding file
    for prediction_file in prediction_files:
        base_name = prediction_file.split('_prediction')[0]  # Remove the '_prediction.json' part
        corresponding_embedding = [f for f in embedding_files if f.startswith(base_name) and f.endswith('_embeddings_100.jsonl')]
        
        # Assuming there's only one match per prediction file
        if corresponding_embedding:
            matches[prediction_file] = corresponding_embedding[0]

    # CFC_dev_2_CFC_dev_annotated()
    if args.protoqa_dev2cfc:
        ProtoQA_annotated_dev_2_CFC()
    elif args.protoqa2cfc:
        # ProtoQA2CFC(args.protoQA_file, args.cfc_prediction, args.cfc_embedding, "/home/qic69/Desktop/CFC/CFC_evaluator_data/GPT2_Predictions/0715modelft_grad8lr5e-06sample200_temp1.0_embeddings_100.jsonl", question2id=True)
        if args.protoqa_data:
            ProtoQA2CFC(args.protoQA_file, args.cfc_prediction, args.cfc_embedding, "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_embeddings.jsonl", question2id=False, protoqa_data=True)
        else:
            ProtoQA2CFC(args.protoQA_file, args.cfc_prediction, args.cfc_embedding, "/home/qic69/Desktop/CFC/CFC_evaluator_data/GPT2_Predictions/0715modelft_grad8lr5e-06sample200_temp1.0_embeddings_100.jsonl", question2id=False)

 
    elif args.cfc2protoqa:
        CFC2ProtoQA(args.cfc_prediction, args.cfc_embedding, args.protoQA_file)


        # Print the matches
    if args.file_selection:
        from protoqa_evaluator.data_processing import load_question_answer_clusters_from_jsonl
        from protoqa_evaluator.evaluation import general_eval, evaluate, multiple_evals
        from protoqa_evaluator.common_evaluations import exact_match_all_eval_funcs, wordnet_all_eval_funcs
        from protoqa_evaluator.scoring import wordnet_score
        import statistics
        from functools import partial
        import csv
        import time
        import signal
        def protoQA_evaluate(cluster_path, prediction_answer):
            predict_answer = {}
            with open(prediction_answer, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    json_obj = json.loads(line.strip())
                    for key, answers in json_obj.items():
                        predict_answer[key] = answers
            question_data = load_question_answer_clusters_from_jsonl(cluster_path)

            # detail, result = multiple_evals(exact_match_all_eval_funcs, question_data, answers_dict=predict_answer)

            max_incorrect_10 = partial(general_eval, max_pred_answers=10, score_func=wordnet_score)
            auto_sampling_protoQA_score = evaluate(max_incorrect_10, question_data, answers_dict=predict_answer)
            result = statistics.mean(x.score for x in auto_sampling_protoQA_score.values())

            # detail, result = multiple_evals(wordnet_all_eval_funcs, question_data, answers_dict=predict_answer)

            return result
        # for prediction, embedding in matches.items():
        #     print(f"Prediction File: {prediction} -> Embedding File: {embedding}")
        
        #     # CFC2ProtoQA('/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/predictions/' + prediction, 
        #     #             '/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/prediction_embeddings/' + embedding, 
        #     #             "/home/qic69/Desktop/CFC/ProtoQA_evaluator_data/more_dev_gpt2_predictions/" + prediction.split('_prediction')[0]+".jsonl")
        #     score = protoQA_evaluate("/home/qic69/Desktop/CFC/ProtoQA_evaluator_data/gt_answer_dev_converted.jsonl",
        #                      "/home/qic69/Desktop/CFC/ProtoQA_evaluator_data/more_dev_gpt2_predictions/" + prediction.split('_prediction')[0]+".jsonl")
        #     break
        def timeout_handler(signum, frame):
            raise TimeoutError

        # Register the signal function handler
        signal.signal(signal.SIGALRM, timeout_handler)
        csv_file_path = '/home/qic69/Desktop/CFC/CFC_dev_protoQA_scores_wn.csv'
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Prediction File',	"Max Answers - 10"]) # Add more score headers if needed
        
            for prediction, embedding in matches.items():
                start = time.time()
                print(f"Prediction File: {prediction} -> Embedding File: {embedding}")
                prediction_file_path = "/home/qic69/Desktop/CFC/ProtoQA_evaluator_data/more_dev_gpt2_predictions/" + prediction.split('_prediction')[0] + ".jsonl"

                signal.alarm(600)

                try:
                    score = protoQA_evaluate("/home/qic69/Desktop/CFC/ProtoQA_evaluator_data/gt_answer_dev_converted.jsonl", prediction_file_path)
                except KeyboardInterrupt:
                    break
                except TimeoutError:
                    print(f"Timeout occurred for {prediction_file_path}")
                    score = -2
                except Exception as e:
                    print(f"Error processing {prediction_file_path}: {e}")
                    score = -1
                finally:
                    # Disable the alarm
                    signal.alarm(0)
                row = [prediction_file_path] + [score]
                print(row)
                csv_writer.writerow(row)
                print("Time used: ", time.time() - start)

    print("Time used: ", time.time() - s)