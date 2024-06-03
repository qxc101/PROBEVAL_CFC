import pandas as pd
import numpy as np
from collections import defaultdict
import json
import tqdm
import nltk
from nltk.cluster import KMeansClusterer

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

import fasttext.util

from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV

from scipy.spatial.distance import cosine
from scipy.spatial.distance import *
from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree, fcluster

from bpemb import BPEmb

import torch
import logging
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModel

from commonsense_validation_api.pre_processer import pre_process
from commonsense_validation_api.utilities import *


tokenize = lambda keys: list(map(lambda x: lemmatize_text_for_answer_analysis(x), keys))
join = lambda tokenizedlist: list(map(lambda x: ' '.join(x), tokenizedlist))


def get_unique_elements(x):
    x = np.array(x)
    x = np.unique(x)
    x = x.tolist()
    return x


def get_answer_list(keys):
    tokenized = tokenize(list(keys))
    joined_words = join(tokenized)
    ## We are not removing the duplicates in order to facilitate the categorical distribution calculataion.
    # joined_words = get_unique_elements(joined_words)
    return joined_words


def average_bpe_embeddings(answer, bpemb_en):
    """Computes the average of the word embeddings in an answer

    Parameters
    ----------
    answer: string
        answer string to a question
    bpemb_en: the byte pair encoding object

    Returns
    -------
    answer_embed: array
        array that contains the average values of all the words in the answer string
    """
    if len(bpemb_en.encode(answer)) > 1:
        word_array_list = []
        for word in bpemb_en.encode(answer):
            try:
                word_array_list.append(bpemb_en[word])
            except:
                pass
        answer_embed = np.mean(word_array_list, axis=0)
    else:
        answer_embed = bpemb_en[bpemb_en.encode(answer)].flatten()
    return answer_embed


def average_fast_embeddings(answer, fast):
    """Computes the average of the word embeddings in an answer

    Parameters
    ----------
    answer: string
        answer string to a question
    fast: the fast text object

    Returns
    -------
    answer_embed: array
        array that contains the average values of all the words in the answer string
    """
    word_array_list = []
    for word in answer.split():
        try:
            word_array_list.append(fast.get_word_vector(word))
        except:
            pass
    answer_embed = np.mean(word_array_list, axis=0)
    return answer_embed


def get_embeddings_or_similarity_matrix(embedding_type, need_similarity_mat=False, remove_stop_words=False):
    result = []
    mapping_list = []
    question_list = []
    clustering_list = []

    protoqa_json = pre_process("data/protoqa.jsonl")

    if embedding_type == "bpe":
        bpemb_en = BPEmb(lang="en", vs=200000, dim=300)
    elif embedding_type == "fasttext":
        ft = fasttext.load_model('embeddings/cc.en.300.bin')
        # ft = fasttext.load_model('cc.en.100.bin')
    elif embedding_type == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        bert_model.eval()
    elif embedding_type == "roberta":
        roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        roberta_model = RobertaModel.from_pretrained('roberta-base')
    else:
        sent_model = SentenceTransformer("models/paraphrase-distilroberta-base-v2")
        sent_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-max-tokens")
        sent_model_bert = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-max-tokens")

    for i, example in enumerate(protoqa_json):
        cluster_map = {}

        # Get cluster mappings required for calcuation of rand index
        for i, cluster_id in enumerate(list(example['answers']['clusters'])):
            cluster = get_answer_list(example['answers']['clusters'][cluster_id]['answers'])
            for x in cluster:
                cluster_map[x] = int(cluster_id.split('.')[1])

        # Remove stop words and lemmatize
        if remove_stop_words:
            answer_list = get_answer_list(example['answers']['raw'].keys())
        else:
            answer_list = get_unique_elements(list(example['answers']['raw'].keys()))

        # Get the answers for bert without tokenizing as their tokenizers do this
        answers = list(example['answers']['raw'].keys())

        # Get the question for context for bert embeddings
        question = example['question']['normalized']

        # Get the mapping of the updated answers
        answer_mapping = [answer_list[i] for i in range(len(answer_list))]

        # Get the cluster mapping of the updated answers
        cluster_mapping = [cluster_map[answer_list[i]] for i in range(len(answer_list))]
        clustering_list.append(cluster_mapping)

        num_answer = len(answer_list)
        embeddings = []

        # Get the similarity matrix using BPE
        if embedding_type == 'bpe' and need_similarity_mat:
            sim_matrix = np.zeros((num_answer, num_answer))
            for i in range(num_answer):
                answer_i_embed = average_bpe_embeddings(answer_list[i], bpemb_en)
                for j in range(num_answer):
                    answer_j_embed = average_bpe_embeddings(answer_list[j], bpemb_en)
                    cosine_similarity = 1 - distance.cosine(answer_i_embed, answer_j_embed)
                    sim_matrix[i, j] = cosine_similarity
            result.append(sim_matrix)

        # Get the embeeddings using BPE
        if embedding_type == 'bpe' and not need_similarity_mat:
            example_embeddings = []
            for i in range(num_answer):
                answer_i_embed = average_bpe_embeddings(answer_list[i], bpemb_en)
                example_embeddings.append(answer_i_embed)
            result.append(example_embeddings)

        # Get the embeeddings using FastText
        if embedding_type == 'fasttext' and need_similarity_mat:
            sim_matrix = np.zeros((num_answer, num_answer))
            for i in range(num_answer):
                answer_i_embed = average_fast_embeddings(answer_list[i], ft)
                for j in range(num_answer):
                    answer_j_embed = average_fast_embeddings(answer_list[j], ft)
                    cosine_similarity = 1 - distance.cosine(answer_i_embed, answer_j_embed)
                    sim_matrix[i, j] = cosine_similarity
            result.append(sim_matrix)

        if embedding_type == 'fasttext' and not need_similarity_mat:
            example_embeddings = []
            for i in range(num_answer):
                answer_i_embed = average_fast_embeddings(answer_list[i], ft)
                example_embeddings.append(answer_i_embed)
            result.append(example_embeddings)

        # Get the embeeddings using BERT 
        if embedding_type == 'bert':
            example_embeddings = []
            for answer in answer_list:
                if question.endswith('?') or question.endswith('.'):
                    question = question[:-1]
                text = question + "? " + answer
                inputs = tokenizer.encode_plus(
                    text,
                    None,
                    add_special_tokens=True,
                    max_length=50,
                    pad_to_max_length=True,
                )
                ids = inputs["input_ids"]
                mask = inputs["attention_mask"]
                token_type_ids = inputs["token_type_ids"]

                tokenized_text = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
                answer_start_index = tokenized_text.index('?') + 1
                answer_end_index = len(tokenized_text)

                d = {
                    "ids": torch.tensor(ids, dtype=torch.long),
                    "mask": torch.tensor(mask, dtype=torch.long),
                    "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                }
                ids = torch.unsqueeze(d["ids"], 0)
                token_type_ids = torch.unsqueeze(d["token_type_ids"], 0)
                mask = torch.unsqueeze(d["mask"], 0)
                with torch.no_grad():
                    outputs = bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids)
                    hidden_states = outputs[2][1:]
                token_embeddings = hidden_states[-1]
                token_embeddings = torch.squeeze(token_embeddings, dim=0)
                list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]
                answer_embedding = [list_token_embeddings[j] for j in range(answer_start_index, answer_end_index)]
                answer_embedding = np.mean(answer_embedding, axis=0)
                example_embeddings.append(answer_embedding)
            example_embeddings = example_embeddings
            result.append(example_embeddings)

        # Get the embeeddings using BERT
        if embedding_type == 'roberta':
            example_embeddings = []
            for answer in answer_list:
                if question.endswith('?') or question.endswith('.'):
                    question = question[:-1]
                text = question + "? " + answer
                inputs = roberta_tokenizer.encode_plus(
                    text,
                    None,
                    add_special_tokens=True,
                    max_length=None,
                    pad_to_max_length=True,
                )
                ids = inputs["input_ids"]
                mask = inputs["attention_mask"]

                tokenized_text = roberta_tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
                answer_start_index = tokenized_text.index('?') + 1
                answer_end_index = len(tokenized_text)

                d = {
                    "ids": torch.tensor(ids, dtype=torch.long),
                    "mask": torch.tensor(mask, dtype=torch.long)
                }
                ids = torch.unsqueeze(d["ids"], 0)
                mask = torch.unsqueeze(d["mask"], 0)
                with torch.no_grad():
                    outputs = roberta_model(ids, attention_mask=mask, output_hidden_states=True)
                    hidden_states = outputs[2][1:]

                token_embeddings = hidden_states[-1]
                token_embeddings = torch.squeeze(token_embeddings, dim=0)
                list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]
                answer_embedding = [list_token_embeddings[j] for j in range(answer_start_index, answer_end_index)]
                answer_embedding = np.mean(answer_embedding, axis=0)
                example_embeddings.append(answer_embedding)
            example_embeddings = example_embeddings
            result.append(example_embeddings)

        # Get the embeeddings using Sentence Transformers
        if embedding_type == 'sentence_bert':
            updated_answer_list = []
            for answer in answer_list:
                if question.endswith('?') or question.endswith('.'):
                    question = question[:-1]
                updated_answer_list.append(question + "? " + answer)
            example_embeddings = sent_model.encode(updated_answer_list)
            example_embeddings = example_embeddings
            result.append(example_embeddings)

        mapping_list.append(answers)
        question_list.append(example['question']['normalized'])
    assert len(result) == len(protoqa_json)
    return result, mapping_list, question_list, clustering_list


def get_fasttext_embeddings(
        answers: list,
        ft,
        need_similarity_matrix: bool = False
) -> Union[list, np.ndarray]:
    """
    Takes answers of a question and returns the embeddings or the similarity matrix

    :param answers: answer strings of a question
    :param ft: fasttext object
    :param need_similarity_matrix: flag for the similarity matrix

    :return: similarity matrix or the embeddings_for_question
    """
    num_answer = len(answers)
    answer_list = get_answer_list(answers)
    if need_similarity_matrix:
        sim_matrix = np.zeros((num_answer, num_answer))
        for i in range(num_answer):
            answer_i_embed = average_fast_embeddings(answer_list[i], ft)
            for j in range(num_answer):
                answer_j_embed = average_fast_embeddings(answer_list[j], ft)
                cosine_similarity = 1 - distance.cosine(answer_i_embed, answer_j_embed)
                sim_matrix[i, j] = cosine_similarity
        return sim_matrix
    else:
        embeddings_for_question = []
        for i in range(num_answer):
            answer_i_embed = average_fast_embeddings(answer_list[i], ft)
            embeddings_for_question.append(answer_i_embed.tolist())
        return embeddings_for_question


def get_fasttext_embeddings_from_file(filename):
    with open(filename, 'r') as f:
        data = f.read()
    reduced_embeddings = json.loads(data)
    return reduced_embeddings
