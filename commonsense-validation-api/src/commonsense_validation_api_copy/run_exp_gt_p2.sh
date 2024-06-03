python evaluator.py --do_protoQA -tune --wandb_config "./config/exp2_gt_r2.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"


# Define the range of configuration files
start=5
end=8

# Loop through the range and start each process in the background
for i in $(seq $start 3 $end)
do
    echo "Starting evaluator.py with config file exp2_gt_r${i}.json"
    python evaluator.py -tune --wandb_config "./config/exp2_gt_r${i}.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
done

python evaluator.py --do_protoQA -tune --wandb_config "./config/exp2_gt_r3.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"


# Define the range of configuration files
start=4
end=9

# Loop through the range and start each process in the background
for i in $(seq $start 3 $end)
do
    echo "Starting evaluator.py with config file exp2_gt_r${i}.json"
    python evaluator.py -tune --wandb_config "./config/exp2_gt_r${i}.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
done