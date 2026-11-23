"""Настройки одной итоговой системы. Веса — стартовые, не доказанный оптимум."""
import json
from pathlib import Path

DEFAULT_CONFIG = {
    'max_chars': 300,
    'use_lemmatization': True,
    'remove_stopwords': True,
    'graph_weight': 0.25,
    'lead_weight': 0.35,
    'position_weight': 0.40,
    'lead_sentences': 3,
    'similarity_threshold': 0.10,
    'damping': 0.85,
    'redundancy_weight': 0.30,
    'duplicate_threshold': 0.85,
    'min_relative_score': 0.25,
    'min_gain': 0.12,
    'length_penalty': 0.20,
    'min_words': 4,
    'compression': True,
    'language_filter': True,
}


def get_config(overrides=None):
    config = DEFAULT_CONFIG.copy()
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError('Настройки должны быть JSON-объектом.')
        unknown = set(overrides) - set(config)
        if unknown:
            raise ValueError('Неизвестные настройки: ' + ', '.join(sorted(unknown)))
        config.update(overrides)
    for key in ('use_lemmatization', 'remove_stopwords', 'compression', 'language_filter'):
        if type(config[key]) is not bool:
            raise ValueError(f'{key}: ожидается true или false.')
    for key in ('max_chars', 'lead_sentences', 'min_words'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key}: ожидается положительное целое число.')
    if config['max_chars'] > 300:
        raise ValueError('По условию лабораторной max_chars не больше 300.')
    weights = ('graph_weight', 'lead_weight', 'position_weight')
    bounded = ('similarity_threshold', 'damping', 'redundancy_weight',
               'duplicate_threshold', 'min_relative_score', 'min_gain', 'length_penalty')
    for key in weights + bounded:
        value = config[key]
        if type(value) not in (float, int) or not 0 <= value <= 1:
            raise ValueError(f'{key}: ожидается число от 0 до 1.')
    if sum(config[key] for key in weights) <= 0:
        raise ValueError('Хотя бы один вес ранжирования должен быть положительным.')
    if config['damping'] >= 1:
        raise ValueError('damping должен быть меньше 1 для сходимости.')
    return config


def load_config(path=None):
    if path is None:
        return get_config()
    return get_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))
