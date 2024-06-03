#!/bin/bash
start=2
end=27

# Loop through the range and start each process in the background for every third file
# for i in $(seq $start 3 $end)
# do
#     echo "Starting evaluator.py with config file exp2_r${i}.json"
#     python evaluator.py -tune --wandb_config "./config/exp2_r${i}.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets copy.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_annotated_predictions.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_annotated_embeddings.jsonl"
# done

echo "Starting evaluator.py with config file exp2_r17.json"
python evaluator.py -tune --wandb_config "./config/exp2_r17.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets copy.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_annotated_predictions.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_annotated_embeddings.jsonl"

echo "Starting evaluator.py with config file exp2_r23.json"
python evaluator.py -tune --wandb_config "./config/exp2_r23.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets copy.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_annotated_predictions.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_annotated_embeddings.jsonl"

echo "Starting evaluator.py with config file exp2_r26.json"
python evaluator.py -tune --wandb_config "./config/exp2_r26.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets copy.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_annotated_predictions.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_annotated_embeddings.jsonl"