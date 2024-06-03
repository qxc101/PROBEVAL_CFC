#!/bin/bash

# Define the range of configuration files

# For diverse sampling:
echo "Starting evaluator.py with config file exp2_r2.json"
python evaluator.py --do_protoQA -tune --wandb_config ".config/exp2_protoqa_r2.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"

echo "Starting evaluator.py with config file exp2_r11.json"
python evaluator.py --do_protoQA -tune --wandb_config ".config/exp2_protoqa_r11.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"

echo "Starting evaluator.py with config file exp2_r20.json"
python evaluator.py --do_protoQA -tune --wandb_config ".config/exp2_protoqa_r20.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"

# For centered sampling 
echo "Starting evaluator.py with config file exp2_r3.json"
python evaluator.py --do_protoQA -tune --wandb_config ".config/exp2_protoqa_r3.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"

echo "Starting evaluator.py with config file exp2_r12.json"
python evaluator.py --do_protoQA -tune --wandb_config ".config/exp2_protoqa_r12.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"

echo "Starting evaluator.py with config file exp2_r21.json"
python evaluator.py --do_protoQA -tune --wandb_config ".config/exp2_protoqa_r21.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
