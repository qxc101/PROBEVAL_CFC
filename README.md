
# PROBEVAL_CFC

This is the official Git repository for the paper "Every Answer Matters: Evaluating Commonsense with Probabilistic Measures," published at ACL 2024.

## 1. Setting Up the Environment

To set up the conda environment for this project, run the following command:

\`\`\`bash
conda env create -f environment.yml
\`\`\`

## 2. Running PROBEVAL to Get KL Scores

### 2.1 Preparing Your Data

Ensure that your answers are formatted into three files: an embedding file, a prediction file, and a target file, similar to the files in \`/CFC_evaluator_data/Evaluator_Input/\`.

### 2.2 Running the Evaluator

Use the following command to run the evaluator:

\`\`\`bash
python evaluator.py --clustering_algo "gmeans" --similarity_function "fasttext" --starting_component 7 --assignment "linear regression" --hac_linkage_function "single" --target "target file path" --prediction "prediction file path" --embedding "embedding file path"
\`\`\`

Example:

\`\`\`bash
python evaluator.py -log --sample_method "wrong_score" --clustering_algo "gmeans" --similarity_function "fasttext" --starting_component 7 --assignment "linear regression" --hac_linkage_function "single" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_annotated_prediction.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/more_dev_gpt2_predictions/0707modelft_grad32lr1e-06sample200_temp0.7_embeddings_100.jsonl"
\`\`\`

### 2.3 Recommended Settings and Options

For recommended settings and other options to run PROBEVAL, please refer to our paper.

## 3. Getting Correlation Scores

To obtain correlation scores, ensure that the \`annotated_prediction_data\` field is present in your prediction file.

## 4. Tuning Parameters for PROBEVAL

To tune parameters for PROBEVAL, set up a Weights and Biases (wandb) account and use the \`-tune\` flag. Specify the wandb configuration with the \`--wandb_config\` option.

Example:

\`\`\`bash
python evaluator.py --do_protoQA -tune --wandb_config "./config/exp2_gt_r2.json" --target "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_targets copy.jsonl" --prediction "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_annotated_predictions.json" --embedding "/home/qic69/Desktop/CFC/CFC_evaluator_data/Evaluator_Input/all_dev_annotated_embeddings.jsonl"
\`\`\`
