"""Лемматизация неизвестных слов на основе статистики суффиксных замен."""
from collections import defaultdict


class OOVModel:
    def __init__(self, train_path):
        # suffix -> (old_ending, new_ending, POS) -> множество лемм
        examples = defaultdict(lambda: defaultdict(set))

        with open(train_path, encoding="utf-8") as f:
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) != 3:
                    continue
                lemma, form, pos = cols
                i = 0
                while i < min(len(form), len(lemma)) and form[i] == lemma[i]:
                    i += 1

                old, new = form[i:], lemma[i:]
                if i < 2 or len(old) > 5 or len(new) > 5:
                    continue

                rule = (old, new, pos)
                for n in (3, 4, 5):
                    if len(form) >= n and len(old) <= n:
                        examples[form[-n:]][rule].add(lemma)

        # Сохраняем только количества, множества после обучения не нужны
        self.rules = {
            suffix: {rule: len(lemmas) for rule, lemmas in variants.items()}
            for suffix, variants in examples.items()
        }

    def predict(self, word, limit=5, min_support=10):
        """Список (lemma, POS, support), отсортированный по поддержке."""
        for n in (5, 4, 3):
            if len(word) < n:
                continue
            variants = {}
            for (old, new, pos), support in self.rules.get(word[-n:], {}).items():
                if support < min_support or not word.endswith(old):
                    continue
                stem = word[:-len(old)] if old else word
                lemma = stem + new
                key = (lemma, pos)
                variants[key] = max(support, variants.get(key, 0))

            if variants:
                ranked = sorted(variants.items(), key=lambda item: (-item[1], item[0]))
                return [(lemma, pos, count) for (lemma, pos), count in ranked[:limit]]
        return []
