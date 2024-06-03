from typing import Optional
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

# nltk.download('stopwords')
stemmer = SnowballStemmer("english")
from typing import *

# nltk.download('stopwords')
tokenizer = nltk.tokenize.WhitespaceTokenizer()
wnl = WordNetLemmatizer()

mystop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
                "these",
                "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
                "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
                "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
                "after", "above", "below", "to", "from", "in", "out", "on", "off", "over", "under",
                "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
                "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
                "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


def normalized_text(text):
    if text.endswith('.'):
        text = text[:-1]
    return text.lower().strip()


def lemmatize_text_keep_sw(text):
    if text.endswith('.'):
        text = text[:-1]
    return [stemmer.stem(word) for word in tokenizer.tokenize(text)]


def lemmatize_text(text):
    # if len([stemmer.stem(word) for word in tokenizer.tokenize(text) if word not in mystop_words]) == 0:
    #     print(text)
    if text.endswith('.'):
        text = text[:-1]
    result = [stemmer.stem(word) for word in tokenizer.tokenize(text) if word not in mystop_words]

    return [stemmer.stem(word) for word in tokenizer.tokenize(text) if word not in mystop_words]


def lemmatize_text_for_answer_analysis(text):
    # we need a seperate function to return original answer, if the stemmed version has length 0.
    if len(lemmatize_text(text)) == 0:
        return text
    else:
        return lemmatize_text(text)


def reject(word_list, sentence):
    return word_found(word_list, sentence) | word_found_instruction(word_list)


def word_found(word_list, sentence):
    num = 0
    # the intuition is that if the word overlap is one, it's considered as invalid answer only if the answer itself is only one word.
    sentence = " " + sentence
    for w in word_list:
        if ' ' + w + ' ' in sentence:
            num += 1
    if num == 1 and len(word_list) == 1:
        return True
    elif num >= 2:
        print(word_list, sentence)
        return True
    else:
        return False


def word_found_instruction(word_list):
    words = " ".join(word_list)
    if "easily answered" in words:
        return True
    if "relevant question grammatically" in words:
        return True
    if "grammatically incorrect" in words:
        return True
    if "answer question" in words:
        return True
    if "find question valid" in words:
        return True
    if "context sentences" in words:
        return True
    sentence = "The goal is to generate a set of questions that are not explicitly answered in the context sentences but can be answered easily by humans using their commonsense or general understanding of the World. We are trying to assess if our question is valid for the given context based on below mentioned criteria. In this task, you will be answering if the question is valid by a 'Yes', 'No', or 'Maybe'. In case you do not find the question valid, you will be required to write a question that meets the criteria discussed below. If the question seems ok, you will be required to provide a word or short phrase as an answer."
    if words in sentence.lower():
        print(words)
        return True
    return False


def sample_indices(k, min_limit, max_limit):
    # Uniformly sample K numbers from total answers length
    assert max_limit >=min_limit
    if min_limit == 0 and max_limit == 0:
        return np.array([0])
    rng = np.random.default_rng()
    sampled_answers_indices = rng.choice(
        np.arange(min_limit, max_limit), size=k, replace=False)
    
    return sampled_answers_indices


def save_to_file(filename, json_data):
    f = open(filename, 'w')
    f.write(json.dumps(json_data))
    f.close()


def gmm_kl(gmm_p, gmm_q, n_samples=10 ** 5):
    X = gmm_p.sample(n_samples)[0]
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    return log_p_X.mean() - log_q_X.mean()


def gmm_js(gmm_p, gmm_q, n_samples=10 ** 5):
    X = gmm_p.sample(n_samples)[0]
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y = gmm_q.sample(n_samples)[0]
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    return np.sqrt((log_p_X.mean() - (log_mix_X.mean() - np.log(2))
                    + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2)


def plot_kl_scors(
        filename: str,
        best_component: Optional[str] = None
):
    kl_scores = [json.loads(line) for line in open(filename)]
    human_kl_scores = []
    auto_kl_scores = []
    for k, v in kl_scores[0][best_component].items():
        for x in v['auto']:
            auto_kl_scores.append(x)
        for x in v['human']:
            human_kl_scores.append(x)
    plt.scatter(human_kl_scores, auto_kl_scores, alpha=0.2)
    plt.xlabel('Human Kl Scores')
    plt.ylabel('GT + Wordnet Scores')
    plt.title('Human Kl Scores vs GT + Wordnet Scores')


def get_kl_scores_cluster_wise(
        dist1: Dict[str, float],
        dist2: Dict[str, float]
) -> (List[float], List[float]):
    dist1_scores, dist2_scores = [], []
    for k, _ in dist1.items():
        dist1_scores.append(dist1[k])
        dist2_scores.append(dist2[k])

    # print(dist1_scores, dist2_scores)
    return dist1_scores, dist2_scores

def sigmoid(input):
    output = []
    for x in input:
        output.append(1/(1 + np.exp(-x)))
    return output

def get_random_distribution(num_of_categories: int) -> np.ndarray:
    # why is the random distribution using (Pranay's code)
    # result = np.random.lognormal(0, 1, num_of_categories)
    # result /= result.sum()

    # Lorraine thinks it should be done this way. 
    # this is a random uniform distribution
    # result = [(1/num_of_categories)]*num_of_categories
    # noise = np.random.normal(0, 1, num_of_categories)
    # result += noise
    # result = sigmoid(result)
    # result /= sum(result)

    # Michael thinks that we don't need randomness
    result = [(1/num_of_categories)]*num_of_categories
    assert abs(sum(result)-1)<1e-10
    return result
