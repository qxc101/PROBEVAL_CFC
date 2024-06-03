#!/bin/bash

# Define the range of configuration files
start=8
end=9

# Loop through the range and start each process in the background
for i in $(seq $start $end)
do
    echo "Starting evaluator.py with config file exp1_r${i}.json"
    python evaluator.py -tune --wandb_config "./config/exp1_r${i}.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_annotated_predictions.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_embeddings.jsonl"
done

