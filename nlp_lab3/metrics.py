"""ROUGE-1/2/L для русского текста. Без лемматизации и удаления стоп-слов."""
import re
from collections import Counter


def tokenize_for_rouge(text):
    return re.findall(r'[a-zа-я0-9]+', text.lower().replace('ё', 'е'))


def overlap_score(overlap, reference_count, prediction_count):
    precision = overlap / prediction_count if prediction_count else 0.0
    recall = overlap / reference_count if reference_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1}


def rouge_n(reference, prediction, n):
    ref = Counter(tuple(reference[i:i+n]) for i in range(len(reference) - n + 1))
    pred = Counter(tuple(prediction[i:i+n]) for i in range(len(prediction) - n + 1))
    overlap = sum((ref & pred).values())
    return overlap_score(overlap, sum(ref.values()), sum(pred.values()))


def rouge_l(reference, prediction):
    previous = [0] * (len(prediction) + 1)
    for word in reference:
        current = [0]
        for j, other in enumerate(prediction, start=1):
            current.append(previous[j-1] + 1 if word == other
                           else max(previous[j], current[-1]))
        previous = current
    return overlap_score(previous[-1], len(reference), len(prediction))


def compute_rouge(reference, prediction):
    ref, pred = tokenize_for_rouge(reference), tokenize_for_rouge(prediction)
    return {'rouge1': rouge_n(ref, pred, 1), 'rouge2': rouge_n(ref, pred, 2),
            'rougeL': rouge_l(ref, pred)}


def average_rouge(scores):
    result = {}
    for name in ('rouge1', 'rouge2', 'rougeL'):
        result[name] = {}
        for metric in ('precision', 'recall', 'f1'):
            result[name][metric] = sum(s[name][metric] for s in scores) / len(scores) if scores else 0.0
    return result
