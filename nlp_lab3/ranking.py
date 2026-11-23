"""Взвешенный LexRank и структурные признаки документа."""
import numpy as np
from preprocessing import russian_ratio, tokenize_sentence


def lexrank(similarity, threshold=0.1, damping=0.85):
    n = len(similarity)
    if n == 0:
        return np.array([])
    graph = similarity.copy()
    np.fill_diagonal(graph, 0)
    graph[graph < threshold] = 0
    transition = np.zeros_like(graph)
    for i, row in enumerate(graph):
        total = row.sum()
        transition[i] = row / total if total > 0 else np.ones(n) / n
    scores = np.ones(n) / n
    for _ in range(200):
        new_scores = (1 - damping) / n + damping * transition.T @ scores
        if np.abs(new_scores - scores).sum() < 1e-8:
            scores = new_scores
            break
        scores = new_scores
    return scores


def sentence_quality(sentence, russian_document=True, language_filter=True):
    words = tokenize_sentence(sentence, use_lemmatization=False, remove_stopwords=False)
    if not words:
        return 0.0
    if language_filter and russian_document and russian_ratio(sentence) < 0.45:
        return 0.0
    quality = 1.0
    if len(words) < 4:
        quality *= 0.35
    if sentence.rstrip().endswith('?'):
        quality *= 0.65
    if sentence.startswith(('«', '"', '—', '- ')):
        quality *= 0.8
    return quality


def rank_sentences(sentences, matrix, config):
    n = len(sentences)
    if n == 0:
        return np.array([]), {}, np.zeros((0, 0))
    similarity = np.clip(matrix @ matrix.T, 0, 1)
    russian_document = russian_ratio(' '.join(sentences)) >= 0.5
    quality = np.array([sentence_quality(s, russian_document, config['language_filter'])
                        for s in sentences])

    graph = lexrank(similarity, config['similarity_threshold'], config['damping'])
    graph = graph / graph.max() if graph.max() else graph

    eligible = [i for i in range(n) if quality[i] >= 0.6 and np.any(matrix[i])]
    if not eligible:
        eligible = [i for i in range(n) if quality[i] > 0]
    lead_indices = eligible[:config['lead_sentences']]
    lead = np.zeros(n)
    if lead_indices:
        # Ранние содержательные предложения имеют больший вес в опорном блоке.
        anchor = np.zeros(matrix.shape[1])
        for order, i in enumerate(lead_indices):
            anchor += matrix[i] / (order + 1)
        norm = np.linalg.norm(anchor)
        if norm:
            lead = np.clip(matrix @ (anchor / norm), 0, 1)

    position = 1 / np.sqrt(np.arange(n) + 1)
    total_weight = sum(config[k] for k in ('graph_weight', 'lead_weight', 'position_weight'))
    scores = (config['graph_weight'] * graph + config['lead_weight'] * lead
              + config['position_weight'] * position) / total_weight
    scores *= quality
    features = {'graph': graph, 'lead': lead, 'position': position, 'quality': quality}
    return scores, features, similarity
