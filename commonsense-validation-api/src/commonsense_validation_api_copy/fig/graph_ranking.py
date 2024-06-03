import os
import wandb
from pathlib import Path
from typing import Callable, List, Union, Optional
from collections import Counter
from scipy.stats.stats import pearsonr, spearmanr
from tqdm import tqdm
import time
from commonsense_validation_api_copy.utilities import save_to_file
from commonsense_validation_api_copy.assignment import wn_automatic_probabilities, human_assignment_probability
from commonsense_validation_api_copy.distributions import clustering_G_probabilities, clustering_hac_G_probabilities
from commonsense_validation_api_copy.correlation_scoring import correlation_scoring
from commonsense_validation_api_copy.get_kl_divergence import KLDivergence
from protoqa_evaluator.data_processing import load_question_answer_clusters_from_jsonl
import protoqa_evaluator.evaluation
from protoqa_evaluator.common_evaluations import exact_match_all_eval_funcs, wordnet_all_eval_funcs
from protoqa_evaluator.scoring import wordnet_score
import json
from typing import NamedTuple, Dict, FrozenSet
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import csv

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# File paths
file_paths = ['/home/qic69/Desktop/CFC/commonsense-validation-api/src/commonsense_validation_api_copy/fig/ranking_missing_answer.csv',
               '/home/qic69/Desktop/CFC/commonsense-validation-api/src/commonsense_validation_api_copy/fig/ranking_wrong_ranking.csv', 
               '/home/qic69/Desktop/CFC/commonsense-validation-api/src/commonsense_validation_api_copy/fig/ranking_wrong_score.csv']

labels = ["missing_answer", "wrong_ranking", "wrong_score"]
colors = ['blue', 'green', 'red']  # Different colors for different files
markers = ['o', 'x']  # Different markers (e.g., circle for auto_scores, cross for protoqa_scores)

# Initialize Min-Max Scaler
scaler = MinMaxScaler()

# Concatenate data from all files for normalization
all_data = pd.concat([pd.read_csv(file).loc[:, ['r2q44_human_score', 'r2q44_auto_score', 'r2q44_protoQA_score']] for file in file_paths])

# Apply normalization on the concatenated data
normalized_data = scaler.fit_transform(all_data)

# Split the normalized data back into separate dataframes
split_indices = [len(pd.read_csv(file)) for file in file_paths]
normalized_dfs = np.split(normalized_data, np.cumsum(split_indices)[:-1])

plt.figure(figsize=(50, 30))

# Plot each file's data
for i, data in enumerate(normalized_dfs):
    # Create a DataFrame from the normalized data
    df = pd.DataFrame(data, columns=['r2q44_human_score', 'r2q44_auto_score', 'r2q44_protoQA_score'])
    
    # Scatter plot for auto_scores
    plt.scatter(df['r2q44_human_score'], df['r2q44_auto_score'], color=colors[i], marker=markers[0], s=400, label=f'Auto Scores ({labels[i]})')

    # Scatter plot for protoqa_scores
    plt.scatter(df['r2q44_human_score'], df['r2q44_protoQA_score'], color=colors[i], marker=markers[1], s=400, label=f'ProtoQA Scores ({labels[i]})')

plt.xlabel('Normalized Human Scores')
plt.ylabel('Normalized Auto and ProtoQA Scores')
plt.title('Normalized Scatter Plot of Scores by Sampling Technique')
plt.legend(fontsize="30")
plt.show()
plt.savefig("normalized_ranking_whole.png")


colors = ['blue', 'green', 'red']  # Different colors for different files
markers = ['o', 'x']  # Different markers for auto_scores and protoqa_scores

# Initialize Min-Max Scaler
scaler = MinMaxScaler()

plt.figure(figsize=(50, 30))

# Process each file individually
for i, file_path in enumerate(file_paths):
    # Load data
    df = pd.read_csv(file_path)

    # Select columns to normalize
    data_to_normalize = df[['r2q44_human_score', 'r2q44_auto_score', 'r2q44_protoQA_score']]

    # Normalize data
    normalized_data = scaler.fit_transform(data_to_normalize)
    normalized_df = pd.DataFrame(normalized_data, columns=['r2q44_human_score', 'r2q44_auto_score', 'r2q44_protoQA_score'])
    
    # Scatter plot for auto_scores
    plt.scatter(normalized_df['r2q44_human_score'], normalized_df['r2q44_auto_score'], color=colors[i], marker=markers[0], s=400, label=f'Auto Scores ({labels[i]})')

    # Scatter plot for protoqa_scores
    plt.scatter(normalized_df['r2q44_human_score'], normalized_df['r2q44_protoQA_score'], color=colors[i], marker=markers[1], s=400, label=f'ProtoQA Scores ({labels[i]})')

plt.xlabel('Normalized Human Scores')
plt.ylabel('Normalized Auto and ProtoQA Scores')
plt.title('Normalized Scatter Plot of Scores by File')
plt.legend(fontsize="30")
plt.show()
plt.savefig("normalized_ranking_individual.png")


def generate_ranking_file(original_file, ranking_file):
    # Read the original CSV file
    df = pd.read_csv(original_file)

    # Rank each column
    for column in df.columns[:3]:
        df[column] = df[column].rank()

    # Save the new DataFrame with rankings to a CSV file
    df.to_csv(ranking_file, index=False)

# Paths to the original files and names for the ranking files
original_files = file_paths
ranking_files = ['/home/qic69/Desktop/CFC/commonsense-validation-api/src/commonsense_validation_api_copy/fig/ranking_missing_answer_rank.csv',
               '/home/qic69/Desktop/CFC/commonsense-validation-api/src/commonsense_validation_api_copy/fig/ranking_wrong_ranking_rank.csv', 
               '/home/qic69/Desktop/CFC/commonsense-validation-api/src/commonsense_validation_api_copy/fig/ranking_wrong_score_rank.csv']

# Generate ranking files
for original, ranking in zip(original_files, ranking_files):
    generate_ranking_file(original, ranking)

