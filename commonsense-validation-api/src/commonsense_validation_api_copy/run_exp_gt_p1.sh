#!/bin/bash

# Exp 1 gt-wn with protoQA wn
start=1
end=3
for i in $(seq $start $end)
do
    echo "Starting evaluator.py with config file exp1_gt_r${i}.json"
    python evaluator.py --do_protoQA -tune --wandb_config "./config/exp1_gt_r${i}.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_annotated_predictions.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_embeddings.jsonl"
done

# # Exp 1 gt-cosine and gt-lg
# start=4
# end=9
# for i in $(seq $start $end)
# do
#     echo "Starting evaluator.py with config file exp1_gt_r${i}.json"
#     python evaluator.py -tune --wandb_config "./config/exp1_gt_r${i}.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_annotated_predictions.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/protoqa_dev_embeddings.jsonl"
# done

# Exp 2 gt-wn with protoQA wn
python evaluator.py --do_protoQA -tune --wandb_config "./config/exp2_gt_r2.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets copy.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_annotated_predictions.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_annotated_embeddings.jsonl"


# # Exp 2 gt-cosine and gt-lg
# start=5
# end=8
# for i in $(seq $start 3 $end)
# do
#     echo "Starting evaluator.py with config file exp2_gt_r${i}.json"
#     python evaluator.py -tune --wandb_config "./config/exp2_gt_r${i}.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets copy.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_annotated_predictions.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_annotated_embeddings.jsonl"
# done
