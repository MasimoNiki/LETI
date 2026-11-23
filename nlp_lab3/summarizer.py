"""Публичный интерфейс: summarize(text) и summarize_many(texts)."""
from config import get_config
from preprocessing import prepare_text
from tfidf import build_tfidf
from ranking import rank_sentences
from selection import make_candidates, select_summary


def summarize(text, config=None, explain=False):
    if not isinstance(text, str):
        raise ValueError('Каждый документ должен быть строкой.')
    config = get_config(config)
    sentences, tokens = prepare_text(text, config['use_lemmatization'], config['remove_stopwords'])
    matrix, vocabulary, idf = build_tfidf(tokens)
    scores, features, similarity = rank_sentences(sentences, matrix, config)
    candidates = make_candidates(sentences, scores, vocabulary, idf, features, config)
    summary, selected, steps, fallback = select_summary(sentences, scores, candidates, config)
    if not explain:
        return summary
    rows = []
    for i, sentence in enumerate(sentences):
        row = {'number': i + 1, 'text': sentence, 'length': len(sentence),
               'score': round(float(scores[i]), 6)}
        for key, values in features.items():
            row[key] = round(float(values[i]), 6)
        rows.append(row)
    details = {'summary': summary, 'length': len(summary), 'selected': selected,
               'selection_order': steps, 'fallback_used': fallback,
               'vocabulary_size': len(vocabulary), 'sentences': rows, 'config': config}
    return summary, details


def summarize_many(texts, config=None):
    if not isinstance(texts, list) or any(not isinstance(t, str) for t in texts):
        raise ValueError('Вход должен быть JSON-массивом строк.')
    config = get_config(config)
    return [summarize(text, config) for text in texts]
