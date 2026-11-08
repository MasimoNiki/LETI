"""Словарный индекс из UniMorph и SynTagRus (без морфоанализаторов)."""
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_FILES = [f"ru_syntagrus-ud-train-{part}.conllu" for part in "abc"]


def normalize(word):
    return word.lower().replace("ё", "е")


def normalize_pos(tag):
    return {
        "N": "NOUN", "V": "VERB",
        "V.PTCP": "VERB", "V.CVB": "VERB",
    }.get(tag, tag)


def load_dictionary(data_dir=DATA_DIR):
    """form -> набор (lemma, POS); сохраняем все альтернативы."""
    result = defaultdict(set)

    with (data_dir / "rus.tsv").open(encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 3:
                continue
            lemma, form, tags = cols
            if lemma and form and tags:
                pos = normalize_pos(tags.split(";")[0])
                result[normalize(form)].add((normalize(lemma), pos))

    for filename in TRAIN_FILES:
        with (data_dir / filename).open(encoding="utf-8") as f:
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) != 10 or not cols[0].isdigit():
                    continue
                form, lemma, pos = cols[1:4]
                if not lemma or lemma == "_" or pos == "PUNCT":
                    continue
                result[normalize(form)].add((normalize(lemma), pos))

    return result
