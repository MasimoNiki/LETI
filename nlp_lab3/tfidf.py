"""TF-IDF внутри документа: одна строка соответствует одному предложению."""
from collections import Counter
import numpy as np


def build_tfidf(tokenized_sentences):
    vocabulary = sorted({w for words in tokenized_sentences for w in words})
    word_to_index = {word: i for i, word in enumerate(vocabulary)}
    counts = np.zeros((len(tokenized_sentences), len(vocabulary)))
    for i, words in enumerate(tokenized_sentences):
        for word, count in Counter(words).items():
            counts[i, word_to_index[word]] = count
    df = (counts > 0).sum(axis=0)
    idf = np.log((len(tokenized_sentences) + 1) / (df + 1)) + 1
    matrix = counts * idf
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms, vocabulary, idf
