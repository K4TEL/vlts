import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
import sys
from unidecode import unidecode
from collections import Counter
from itertools import combinations

from sklearn.preprocessing import MinMaxScaler
from Levenshtein import distance as lev

from trans import trans


dataset_file_path = 'CogNet-v2.0.tsv'

selected_file_path = "selected.tsv"
data_file_path = "data.tsv"

bigrams_file_path = "bigram_probs.tsv"
bg_score_file_path = "bigram_scores.tsv"
bg_score_img_path = "bigram_scores.png"

lev_score_file_path = "leven_scores.tsv"
lev_score_img_path = "leven_scores.png"

total_score_file_path = "total_scores.tsv"
total_score_img_path = "total_scores.png"

selected_langs = ["eng", "ukr", "ces", "fra", "ita",
                  "deu", "lat", "pol", "rus", "nob"]

langs_map = {
    "eng": "English",
    "ukr": "Ukrainian",
    "ces": "Czech",
    "fra": "French",
    "ita": "Italian",
    "deu": "German",
    "lat": "Latin",
    "pol": "Polish",
    "rus": "Russian",
    "nob": "Norwegian"
}


def read_selected_lines(dataset_file):
    lines = []
    with (open(dataset_file, 'r') as file):
        for i, line in enumerate(file):
            if i == 0:
                header = line.split("\t")
                print(header)

            columns = line.split("\t")
            concept_id = columns[0]
            lang1, lang2 = columns[1], columns[3]
            if lang1 in selected_langs and lang2 in selected_langs:
                # print(i, len(words), line)
                lines.append(columns)

    print(len(lines))
    df = pd.DataFrame(lines)
    df.rename(columns={'translit 2\n': 'translit 2'}, inplace=True)

    def set_to_none_if_no_letters(value):
        if value == "\n":
            return None

        return value[:-1] if any(char.isalpha() for char in value) else None

    df['translit 2'] = df['translit 2'].apply(set_to_none_if_no_letters)
    df.loc[df['translit 1'].notnull(), 'word 1'] = df['translit 1']
    df.loc[df['translit 2'].notnull(), 'word 2'] = df['translit 2']
    df.drop(columns=["translit 1", "translit 2"], inplace=True)

    df = df.groupby('concept id').filter(lambda x: len(x) > 10)

    print(df)
    print(df.info())

    df.to_csv(selected_file_path, sep="\t", index=False)

    return df


def get_words_table(selected_file):
    words = []
    full_words = []

    with (open(selected_file, 'r') as file):
        for i, line in enumerate(file):
            if i == 0:
                header = line.split("\t")
                print(header)

            if i % 1000 == 0:
                print(i, len(words), len(full_words), line)

            columns = line.split("\t")

            concept_id = columns[0]
            lang1, lang2 = columns[1], columns[3]
            word1, word2 = columns[2], columns[4][:-1]

            if len(words) > 0:

                match = [item for item in words if item.get('concept_id') == concept_id]

                updated = []
                for j, w in enumerate(match):

                    if lang1 in w.keys() and lang2 not in w.keys():
                        if w[lang1] == word1:
                            words.remove(w)

                            w[lang2] = word2

                            updated.append(w)

                    elif lang2 in w.keys() and lang1 not in w.keys():
                        if w[lang2] == word2:
                            words.remove(w)

                            w[lang1] = word1

                            updated.append(w)

                if len(updated) == 1:
                    words += updated
                elif len(updated) == 0:
                    dict = {
                        "concept_id": concept_id,
                        lang1: word1,
                        lang2: word2
                    }

                    words.append(dict)
                elif len(updated) > 1:
                    dict = {}
                    for d in updated:
                        dict = {**dict, **d}
                    words.append(dict)

                    # print(updated, dict)

            else:
                dict = {
                    "concept_id": concept_id,
                    lang1: columns[2],
                    lang2: columns[4]
                }

                words.append(dict)

            for j, w in enumerate(words):
                if len(w.keys()) == len(selected_langs):
                    full_words.append(w)
                    words.pop(j)
                    print(w)

    print(len(full_words))
    print(len(words))

    total_words = full_words + words

    for w in total_words:
        for lang in selected_langs:
            if lang not in w.keys():
                w[lang] = ""

    df = pd.DataFrame(total_words)

    df.drop(columns=["lang 1", "lang 2"], inplace=True)

    digit_pattern = r'\d'
    df = df[~df[selected_langs].apply(lambda x: x.astype(str).str.contains(digit_pattern)).any(axis=1)]

    df[selected_langs] = df[selected_langs].apply(lambda x: x.apply(trans))

    df = df.groupby('concept_id').agg(lambda x: ' '.join(set(map(str, x)))).reset_index()
    df[selected_langs] = df[selected_langs].apply(lambda x: x.str.strip())

    df.replace('', np.nan, inplace=True)
    df = df.sort_values(by='eng')

    print(df)
    print(df.info())

    df.to_csv(data_file_path, sep="\t", index=False)

    return df


def bigram_probs(data_file):
    df = pd.read_csv(data_file, sep="\t")
    print(df.info())

    df = df.fillna('')

    bigram_probs = {}

    for lang in selected_langs:
        words = df[lang].tolist()
        words = list(filter(None, words))

        unite = " ".join(words)

        bigrams = [unite[i:i + 2] for i in range(len(unite) - 1)]
        bigram_counts = Counter(bigrams)
        total_bigrams = len(bigrams)
        bigram_probabilities = {bigram: count / total_bigrams for bigram, count in bigram_counts.items()}

        # print(lang, len(words), total_bigrams, len(bigram_probabilities.keys()))
        bigram_probs[lang] = bigram_probabilities

    unique_bigrams_total = []
    for bg_probs in bigram_probs.values():
        unique_bigrams_total += list(bg_probs.keys())
    unique_bigrams_total = list(set(unique_bigrams_total))

    # print(len(unique_bigrams_total))

    df = pd.DataFrame()

    for language, bigram_probabilities in bigram_probs.items():
        for bg in unique_bigrams_total:
            if bg not in bigram_probabilities.keys():
                bigram_probabilities[bg] = 0
        df[language] = pd.Series(bigram_probabilities)

    df = df.fillna(0)

    print(df)
    print(df.info())

    df.to_csv(bigrams_file_path, sep="\t")

    return df


def bigram_scores(bigrams_file):
    df = pd.read_csv(bigrams_file, sep="\t", index_col=0)
    # print(df.info())

    language_pairs = list(combinations(df.columns, 2))
    scores = {}

    for lang1, lang2 in language_pairs:
        abs_diff = df[lang1] * 100 - df[lang2] * 100
        count_nonzero_rows = len(df[(df[lang1] != 0) | (df[lang2] != 0)])
        score = abs_diff.abs().sum()
        # print(score, count_nonzero_rows, score/count_nonzero_rows)
        scores[f'{lang1}_{lang2}'] = score / count_nonzero_rows

    score_matrix = pd.DataFrame(index=df.columns, columns=df.columns)

    for lang1, lang2 in language_pairs:
        score_matrix.loc[lang1, lang2] = scores[f'{lang1}_{lang2}']
        score_matrix.loc[lang2, lang1] = scores[f'{lang1}_{lang2}']

    score_matrix.fillna(0, inplace=True)

    min_val, max_val = score_matrix.min().min(), score_matrix.max().max()
    score_matrix = (score_matrix - min_val) / (max_val - min_val) * 100

    sum_sorted_order = score_matrix.sum(axis=1).sort_values(ascending=True).index
    score_matrix = score_matrix.loc[sum_sorted_order]
    score_matrix = score_matrix[sum_sorted_order]

    print(score_matrix)

    score_matrix.to_csv(bg_score_file_path, sep="\t")

    score_matrix.rename(columns=langs_map, index=langs_map, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(score_matrix, cmap='RdYlGn_r', interpolation='none')

    for i in range(len(score_matrix.columns)):
        for j in range(len(score_matrix.index)):
            text = ax.text(j, i, f'{score_matrix.iloc[i, j]:.2f}',
                           ha='center', va='center', color='black')

    plt.colorbar(im)
    plt.xticks(range(len(score_matrix.columns)), score_matrix.columns, rotation=45)
    plt.yticks(range(len(score_matrix.index)), score_matrix.index)
    plt.title('Normalized Mean Bigram Differences')
    # plt.show()
    plt.savefig(bg_score_img_path)

    return score_matrix


def leven_scores(data_file):
    df = pd.read_csv(data_file, sep="\t", index_col=0)
    # print(df)
    print(df.info())

    df.fillna("", inplace=True)

    language_pairs = list(combinations(df.columns, 2))
    scores = {}

    for lang1, lang2 in language_pairs:
        leven = 0
        cnt = 0
        for i, k in zip(df[lang1].tolist(), df[lang2].tolist()):
            if len(i) > 0 and len(k) > 0:
                words1 = i.split(" ")
                words2 = k.split(" ")
                if len(words1) == 1 and len(words2) == 1:
                    leven += lev(i, k)
                elif len(words1) > 1 and len(words2) == 1:
                    words = list(filter(None, words1))
                    w_num = len(words)
                    l = 0
                    for w in words:
                        l += lev(w, k)
                    leven += l / w_num
                elif len(words1) == 1 and len(words2) > 1:
                    words = list(filter(None, words2))
                    w_num = len(words)
                    l = 0
                    for w in words:
                        l += lev(i, w)
                    leven += l / w_num
                else:
                    words1 = list(filter(None, words1))
                    words2 = list(filter(None, words2))
                    w_num = len(words1) + len(words2)
                    l = 0
                    for w in words1:
                        for v in words2:
                            l += lev(w, v)
                    leven += l / w_num

                cnt += 1

        # print(leven, cnt, leven/cnt)

        scores[f'{lang1}_{lang2}'] = leven / cnt

    score_matrix = pd.DataFrame(index=df.columns, columns=df.columns)

    for lang1, lang2 in language_pairs:
        score_matrix.loc[lang1, lang2] = scores[f'{lang1}_{lang2}']
        score_matrix.loc[lang2, lang1] = scores[f'{lang1}_{lang2}']

    score_matrix.fillna(0, inplace=True)

    min_val, max_val = int(score_matrix[score_matrix > 0].min().min()), score_matrix.max().max()
    min_val, max_val = score_matrix.min().min(), score_matrix.max().max()
    # print(min_val, max_val)
    score_matrix = (score_matrix - min_val) / (max_val - min_val) * 100
    # score_matrix[score_matrix < 0] = 0

    sum_sorted_order = score_matrix.sum(axis=1).sort_values(ascending=True).index
    score_matrix = score_matrix.loc[sum_sorted_order]
    score_matrix = score_matrix[sum_sorted_order]

    print(score_matrix)

    score_matrix.to_csv(lev_score_file_path, sep="\t")

    score_matrix.rename(columns=langs_map, index=langs_map, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(score_matrix, cmap='RdYlGn_r', interpolation='none')

    for i in range(len(score_matrix.columns)):
        for j in range(len(score_matrix.index)):
            text = ax.text(j, i, f'{score_matrix.iloc[i, j]:.2f}',
                           ha='center', va='center', color='black')

    plt.colorbar(im)
    plt.xticks(range(len(score_matrix.columns)), score_matrix.columns, rotation=45)
    plt.yticks(range(len(score_matrix.index)), score_matrix.index)
    plt.title('Normalized Mean Levenstein Distance')
    # plt.show()
    plt.savefig(lev_score_img_path)

    return score_matrix


def total_scores(leven_scores, bigram_scores):
    lev_matrix = pd.read_csv(leven_scores, sep="\t", index_col=0)
    bg_matrix = pd.read_csv(bigram_scores, sep="\t", index_col=0)

    lev_matrix = lev_matrix.sort_index(axis=0).sort_index(axis=1)
    bg_matrix = bg_matrix.sort_index(axis=0).sort_index(axis=1)

    # print(lev_matrix)
    # print(bg_matrix)

    total_matrix = (lev_matrix + bg_matrix) / 2

    # print(total_matrix)

    sum_sorted_order = total_matrix.sum(axis=1).sort_values(ascending=True).index
    total_matrix = total_matrix.loc[sum_sorted_order]
    total_matrix = total_matrix[sum_sorted_order]

    print(total_matrix)

    total_matrix.to_csv(total_score_file_path, sep="\t")

    total_matrix.rename(columns=langs_map, index=langs_map, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(total_matrix, cmap='RdYlGn_r', interpolation='none')

    for i in range(len(total_matrix.columns)):
        for j in range(len(total_matrix.index)):
            text = ax.text(j, i, f'{total_matrix.iloc[i, j]:.2f}',
                           ha='center', va='center', color='black')

    plt.colorbar(im)
    plt.xticks(range(len(total_matrix.columns)), total_matrix.columns, rotation=45)
    plt.yticks(range(len(total_matrix.index)), total_matrix.index)
    plt.title('Total Difference Score')
    # plt.show()
    plt.savefig(total_score_img_path)

    return total_matrix


# read_selected_lines(dataset_file_path)
#
# get_words_table(selected_file_path)

bigram_probs(data_file_path)

bigram_scores(bigrams_file_path)

leven_scores(data_file_path)

total_scores(lev_score_file_path, bg_score_file_path)
