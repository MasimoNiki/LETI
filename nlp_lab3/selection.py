"""Отбор по полезности, длине и новизне с точным символьным бюджетом."""
import re
from collections import Counter
from textwrap import shorten
import numpy as np

from compression import compressed_versions, balance_quotes
from preprocessing import tokenize_sentence, has_predicate

DEPENDENT_START = re.compile(
    r'^(?:он|она|оно|они|его|ее|их|это|эти|этот|эта|такой|такая|там|'
    r'после этого|об этом|к этому|в связи с этим|соответствующее заявление|'
    r'по его словам|по ее словам|по их словам)\b', re.I)


def vectorize(text, vocabulary, idf, config):
    words = tokenize_sentence(text, config['use_lemmatization'], config['remove_stopwords'])
    counts = Counter(words)
    vector = np.array([counts.get(w, 0) for w in vocabulary], dtype=float) * idf
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def make_candidates(sentences, scores, vocabulary, idf, features, config):
    candidates = []
    for i, sentence in enumerate(sentences):
        if scores[i] <= 0:
            continue
        original_words = tokenize_sentence(sentence, False, False)
        for text, rule in compressed_versions(sentence, config['compression']):
            words = tokenize_sentence(text, False, False)
            meaningful_short = len(words) >= 2 and has_predicate(text)
            if len(words) < config['min_words'] and not meaningful_short:
                continue
            if len(words) <= 6 and not has_predicate(text):
                continue
            # Слишком короткий вариант не получает весь балл длинного оригинала.
            retention = (len(words) / max(1, len(original_words))) ** 0.5
            score = float(scores[i]) * min(1.0, retention)
            candidates.append({
                'text': text, 'source_index': i, 'rule': rule, 'score': score,
                'vector': vectorize(text, vocabulary, idf, config),
                'dependent_start': bool(DEPENDENT_START.search(text.lower().replace('ё', 'е'))),
            })
    return candidates


def select_summary(sentences, scores, candidates, config):
    limit = config['max_chars']
    selected = []
    used_chars = 0
    used_sources = set()
    best_score = max((c['score'] for c in candidates), default=0.0)
    steps = []

    while True:
        best = None
        best_priority = -float('inf')
        best_gain = 0.0
        for candidate in candidates:
            if candidate['source_index'] in used_sources:
                continue
            # Контекст нужен и первому кандидату: порядок отбора затем меняется.
            if (candidate['source_index'] > 0 and candidate['dependent_start']
                    and candidate['source_index'] - 1 not in used_sources):
                continue
            extra = len(candidate['text']) + (1 if selected else 0)
            if used_chars + extra > limit:
                continue
            if candidate['score'] < config['min_relative_score'] * best_score:
                continue
            redundancy = max((float(candidate['vector'] @ c['vector'])
                              for c in selected), default=0.0)
            if selected and redundancy >= config['duplicate_threshold']:
                continue
            relevance = candidate['score']
            if not selected and candidate['dependent_start']:
                relevance *= 0.65
            gain = relevance - config['redundancy_weight'] * redundancy
            if gain < config['min_gain'] * best_score:
                continue
            priority = gain / (max(1, len(candidate['text'])) / 100) ** config['length_penalty']
            if priority > best_priority:
                best, best_priority, best_gain = candidate, priority, gain
        if best is None:
            break
        selected.append(best)
        used_sources.add(best['source_index'])
        used_chars += len(best['text']) + (1 if len(selected) > 1 else 0)
        steps.append({'source_number': best['source_index'] + 1,
                      'gain': round(best_gain, 6), 'text': best['text']})

    fallback = False
    if not selected and sentences:
        # Это аварийный путь для текста без подходящего целого кандидата.
        # Сначала выбирается содержательное предложение по рейтингу, не начало документа.
        pool = [c for c in candidates if not c['dependent_start'] or c['source_index'] == 0]
        if not pool:
            pool = candidates
        if pool:
            best = max(pool, key=lambda c: c['score'])
            text, index = best['text'], best['source_index']
        else:
            index = int(np.argmax(scores))
            text = balance_quotes(sentences[index])
        if tokenize_sentence(text, False, False):
            fragment = shorten(text, width=limit, placeholder='…',
                               break_long_words=False, break_on_hyphens=False)
            selected = [{'text': fragment, 'source_index': index,
                         'rule': 'запасное сокращение по границам слов',
                         'score': float(scores[index])}]
            fallback = True

    selected.sort(key=lambda c: c['source_index'])
    summary = ' '.join(c['text'] for c in selected)
    if len(summary) > limit:
        raise RuntimeError('Внутренняя ошибка: превышен лимит реферата.')
    details = [{'source_number': c['source_index'] + 1, 'text': c['text'],
                'rule': c['rule'], 'score': round(c['score'], 6)} for c in selected]
    return summary, details, steps, fallback
