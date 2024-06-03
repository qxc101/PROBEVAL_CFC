#!/bin/bash

# Define the range of configuration files
start=1
end=3

# Loop through the range and start each process in the background
for i in $(seq $start $end)
do
    echo "Starting evaluator.py with config file exp1_r${i}.json"
    python evaluator.py --do_protoQA -tune --wandb_config "./config/exp1_r${i}.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_annotated_predictions.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_embeddings.jsonl"
done

start=10
end=12

# Loop through the range and start each process in the background
for i in $(seq $start $end)
do
    echo "Starting evaluator.py with config file exp1_r${i}.json"
    python evaluator.py --do_protoQA -tune --wandb_config "./config/exp1_r${i}.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_annotated_predictions.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_embeddings.jsonl"
done

start=19
end=21

# Loop through the range and start each process in the background
for i in $(seq $start $end)
do
    echo "Starting evaluator.py with config file exp1_r${i}.json"
    python evaluator.py --do_protoQA -tune --wandb_config "./config/exp1_r${i}.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_annotated_predictions.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_embeddings.jsonl"
done


