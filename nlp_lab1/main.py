"""Лемматизация текста с POS-тегами."""
import argparse
import json
import re
from pathlib import Path

from context_model import ContextModel
from dictionary import DATA_DIR, load_dictionary, normalize
from oov_model import OOVModel


def tokenize(line):
    # В задании разрешены только , . ? !
    return re.sub(r"[,.!?]", " ", line).split()


def main():
    parser = argparse.ArgumentParser(description="Лемматизация текста и определение POS")
    parser.add_argument("--input", default="input.txt", help="Входной файл UTF-8")
    parser.add_argument("--output", default="results/output.txt", help="Результат UTF-8")
    parser.add_argument("--details", help="Необязательный JSONL с кандидатами")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Загружаем словарь...")
    dictionary = load_dictionary()
    print("Уникальных словоформ:", len(dictionary))

    print("Извлекаем OOV-правила...")
    oov = OOVModel(DATA_DIR / "train.tsv")

    print("Подготавливаем контекстную модель...")
    context = ContextModel(DATA_DIR)

    outputs, logs = [], []
    for line in input_path.read_text(encoding="utf-8-sig").splitlines():
        tokens = tokenize(line)
        words = [normalize(token) for token in tokens]
        candidates = []

        for token, word in zip(tokens, words):
            known = dictionary.get(word)
            if known:
                candidates.append([(lemma, pos, "DICT", 0) for lemma, pos in sorted(known)])
                continue

            predicted = oov.predict(word)
            if predicted:
                candidates.append([(lemma, pos, "OOV", support)
                                   for lemma, pos, support in predicted])
            else:
                # Последний запасной вариант для нестандартных и неизменяемых слов
                pos = "PROPN" if token[:1].isupper() else "X"
                candidates.append([(word, pos, "FALLBACK", 0)])

        chosen = context.decode(words, candidates)
        outputs.append(" ".join(
            f"{token}{{{lemma}={pos}}}"
            for token, (lemma, pos, source, support) in zip(tokens, chosen)
        ))

        if args.details:
            logs.append({"tokens": [
                {"token": token, "chosen": list(picked),
                 "candidates": [list(c) for c in variants]}
                for token, picked, variants in zip(tokens, chosen, candidates)
            ]})

    output_path.write_text("\n".join(outputs) + "\n", encoding="utf-8")
    print("Результат:", output_path.resolve())

    if args.details:
        details_path = Path(args.details)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("w", encoding="utf-8") as f:
            for record in logs:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("Разборы:", details_path.resolve())


if __name__ == "__main__":
    main()
