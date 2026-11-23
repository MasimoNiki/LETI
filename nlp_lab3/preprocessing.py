"""Подготовка русского текста. Исходные предложения сохраняются для вывода."""
import re
from functools import lru_cache

from pymorphy3 import MorphAnalyzer
from razdel import sentenize, tokenize

MORPH = MorphAnalyzer()
STOP_WORDS = set('''и а но или либо в во на с со к ко у о об обо от до из изо
по для при же бы ли ведь вот это этот тот такой свой который как что чтобы
он она оно они мы вы я ты его ее их наш ваш мой твой себя также кроме
между над под перед после'''.split())
RUSSIAN_WORD = re.compile(r'[а-яё]+(?:-[а-яё]+)*', re.I)


def clean_text(text):
    text = text.replace('\u00a0', ' ').replace('\u200b', '')
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


@lru_cache(maxsize=50000)
def analyze_word(word):
    parsed = MORPH.parse(word)[0]
    return parsed.normal_form, parsed.tag.POS


@lru_cache(maxsize=50000)
def lemmatize_word(word):
    if RUSSIAN_WORD.fullmatch(word):
        word = analyze_word(word.lower())[0]
    return word.lower().replace('ё', 'е')


def has_predicate(text):
    for token in tokenize(text):
        if RUSSIAN_WORD.fullmatch(token.text):
            if analyze_word(token.text.lower())[1] in {'VERB', 'PRTS', 'PRED'}:
                return True
    return False


def split_sentences(text):
    text = clean_text(text)
    # Защищаем точку в коротком названии: «Ай-Петри. Крым».
    def protect_title(match):
        inside = match.group(1)
        if (len(inside.split()) <= 6 and not has_predicate(inside)
                and re.search(r'\.\s+[А-ЯЁ]', inside)):
            return '«' + inside.replace('.', '\u2024') + '»'
        return match.group(0)

    protected = re.sub(r'«([^«»\n]{1,100})»', protect_title, text)
    sentences = []
    for part in sentenize(protected):
        sentence = re.sub(r'\s+', ' ', part.text).replace('\u2024', '.').strip()
        sentence = re.sub(r'\s+([,;:!?])', r'\1', sentence)
        if sentence:
            sentences.append(sentence)
    return sentences


def tokenize_sentence(sentence, use_lemmatization=True, remove_stopwords=True):
    words = []
    for token in tokenize(sentence):
        word = token.text.lower()
        if not any(char.isalnum() for char in word):
            continue
        if use_lemmatization:
            word = lemmatize_word(word)
        else:
            word = word.replace('ё', 'е')
        if remove_stopwords and word in STOP_WORDS:
            continue
        words.append(word)
    return words


def prepare_text(text, use_lemmatization=True, remove_stopwords=True):
    sentences = split_sentences(text)
    tokens = [tokenize_sentence(s, use_lemmatization, remove_stopwords)
              for s in sentences]
    return sentences, tokens


def russian_ratio(text):
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum('а' <= char.lower() <= 'я' or char.lower() == 'ё'
               for char in letters) / len(letters)
