#!/bin/bash


python evaluator.py -log --sample_method "wrong_score" --clustering_algo "gt" --similarity_function "wordnet" --starting_component 7 --assignment "cosine" --hac_linkage_function "single" --do_protoQA --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "gt" --similarity_function "fasttext" --starting_component 7 --assignment "cosine" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "gt" --similarity_function "fasttext" --starting_component 7 --assignment "linear regression" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"

python evaluator.py -log --sample_method "wrong_score" --clustering_algo "gmeans" --similarity_function "wordnet" --starting_component 4 --assignment "cosine" --hac_linkage_function "single" --do_protoQA --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "gmeans" --similarity_function "fasttext" --starting_component 7 --assignment "cosine" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "gmeans" --similarity_function "fasttext" --starting_component 7 --assignment "linear regression" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"

python evaluator.py -log --sample_method "wrong_score" --clustering_algo "xmeans" --similarity_function "wordnet" --starting_component 14 --assignment "cosine" --hac_linkage_function "single" --do_protoQA --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "xmeans" --similarity_function "fasttext" --starting_component 14 --assignment "cosine" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "xmeans" --similarity_function "fasttext" --starting_component 14 --assignment "linear regression" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"

python evaluator.py -log --sample_method "wrong_score" --clustering_algo "hac" --similarity_function "wordnet" --starting_component 0.4123 --assignment "cosine" --hac_linkage_function "centroid" --do_protoQA --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "hac" --similarity_function "fasttext" --starting_component 0.215 --assignment "cosine" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "hac" --similarity_function "fasttext" --starting_component 0.298 --assignment "linear regression" --hac_linkage_function "ward" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"


python evaluator.py -log --sample_method "wrong_score" --clustering_algo "gt" --similarity_function "wordnet" --starting_component 7 --assignment "cosine" --hac_linkage_function "single" --do_protoQA --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "gt" --similarity_function "fasttext" --starting_component 7 --assignment "cosine" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "gt" --similarity_function "fasttext" --starting_component 7 --assignment "linear regression" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"

python evaluator.py -log --sample_method "wrong_score" --clustering_algo "gmeans" --similarity_function "wordnet" --starting_component 4 --assignment "cosine" --hac_linkage_function "single" --do_protoQA --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "gmeans" --similarity_function "fasttext" --starting_component 7 --assignment "cosine" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "gmeans" --similarity_function "fasttext" --starting_component 7 --assignment "linear regression" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"

python evaluator.py -log --sample_method "wrong_score" --clustering_algo "xmeans" --similarity_function "wordnet" --starting_component 14 --assignment "cosine" --hac_linkage_function "single" --do_protoQA --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "xmeans" --similarity_function "fasttext" --starting_component 14 --assignment "cosine" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "xmeans" --similarity_function "fasttext" --starting_component 14 --assignment "linear regression" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"

python evaluator.py -log --sample_method "wrong_score" --clustering_algo "hac" --similarity_function "wordnet" --starting_component 0.4123 --assignment "cosine" --hac_linkage_function "centroid" --do_protoQA --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "hac" --similarity_function "fasttext" --starting_component 0.215 --assignment "cosine" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "hac" --similarity_function "fasttext" --starting_component 0.298 --assignment "linear regression" --hac_linkage_function "ward" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"


python evaluator.py -log --sample_method "wrong_score" --clustering_algo "gt" --similarity_function "wordnet" --starting_component 7 --assignment "cosine" --hac_linkage_function "single" --do_protoQA --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "gt" --similarity_function "fasttext" --starting_component 7 --assignment "cosine" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "gt" --similarity_function "fasttext" --starting_component 7 --assignment "linear regression" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"

python evaluator.py -log --sample_method "wrong_score" --clustering_algo "gmeans" --similarity_function "wordnet" --starting_component 4 --assignment "cosine" --hac_linkage_function "single" --do_protoQA --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "gmeans" --similarity_function "fasttext" --starting_component 7 --assignment "cosine" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "gmeans" --similarity_function "fasttext" --starting_component 7 --assignment "linear regression" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"

python evaluator.py -log --sample_method "wrong_score" --clustering_algo "xmeans" --similarity_function "wordnet" --starting_component 14 --assignment "cosine" --hac_linkage_function "single" --do_protoQA --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "xmeans" --similarity_function "fasttext" --starting_component 14 --assignment "cosine" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "xmeans" --similarity_function "fasttext" --starting_component 14 --assignment "linear regression" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"

python evaluator.py -log --sample_method "wrong_score" --clustering_algo "hac" --similarity_function "wordnet" --starting_component 0.4123 --assignment "cosine" --hac_linkage_function "centroid" --do_protoQA --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "hac" --similarity_function "fasttext" --starting_component 0.215 --assignment "cosine" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "hac" --similarity_function "fasttext" --starting_component 0.298 --assignment "linear regression" --hac_linkage_function "ward" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
