"""Простая контекстная модель: частоты POS-переходов и разборов слов."""
from collections import Counter, defaultdict
from math import log, log1p

from dictionary import TRAIN_FILES, normalize


class ContextModel:
    def __init__(self, data_dir):
        self.transitions = defaultdict(Counter)
        self.lexicon_freq = Counter()
        self.form_freq = Counter()
        self.tags = set()

        for filename in TRAIN_FILES:
            previous = "<START>"
            with (data_dir / filename).open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        if previous != "<START>":
                            self.transitions[previous]["<END>"] += 1
                        previous = "<START>"
                        continue

                    cols = line.rstrip("\n").split("\t")
                    if len(cols) != 10 or not cols[0].isdigit():
                        continue
                    form, lemma, pos = cols[1:4]
                    if pos == "PUNCT" or lemma == "_":
                        continue

                    word, lemma = normalize(form), normalize(lemma)
                    self.transitions[previous][pos] += 1
                    self.lexicon_freq[(word, lemma, pos)] += 1
                    self.form_freq[word] += 1
                    self.tags.add(pos)
                    previous = pos

            if previous != "<START>":
                self.transitions[previous]["<END>"] += 1

        self.transition_totals = {
            prev: sum(next_tags.values())
            for prev, next_tags in self.transitions.items()
        }

    def transition_score(self, previous, current):
        counts = self.transitions.get(previous, {})
        alpha = 0.5  # сглаживание для редких POS-сочетаний
        k = len(self.tags) + 1
        total = self.transition_totals.get(previous, 0)
        return log((counts.get(current, 0) + alpha) / (total + alpha * k))

    def candidate_score(self, word, lemma, pos, support, source, num_candidates):
        if source == "OOV":
            # По умолчанию более распространённые OOV-правила предпочтительнее
            return 0.45 * log1p(support)

        # Среди словарных разборов учитываем частоты из размеченного корпуса
        total = self.form_freq[word]
        if not total:
            return 0.0
        count = self.lexicon_freq[(word, lemma, pos)]
        return 0.7 * log((count + 0.5) / (total + 0.5 * num_candidates))

    def decode(self, words, candidate_lists):
        """Алгоритм Витерби — наиболее вероятная последовательность разборов."""
        if not words:
            return []

        scores = {"<START>": 0.0}
        paths = {"<START>": []}

        for word, variants in zip(words, candidate_lists):
            next_scores, next_paths = {}, {}
            for candidate in variants:
                lemma, pos, source, support = candidate
                emission = self.candidate_score(
                    word, lemma, pos, support, source, len(variants)
                )
                best_score, best_path = float("-inf"), None

                for prev, previous_score in scores.items():
                    prev_pos = prev[1] if isinstance(prev, tuple) else prev
                    score = previous_score + self.transition_score(prev_pos, pos) + emission
                    if score > best_score:
                        best_score = score
                        best_path = paths[prev]

                # Состояния: POS+лемма, а не только POS
                key = (lemma, pos, source, support)
                next_scores[key] = best_score
                next_paths[key] = best_path + [candidate]

            scores, paths = next_scores, next_paths

        final = max(
            scores,
            key=lambda state: scores[state] + self.transition_score(state[1], "<END>"),
        )
        return paths[final]
