"""Build a held-out sentence benchmark from the UD Russian SynTagRus test split.

No third-party packages; never trains on test labels.
"""
import argparse
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/"
    "UD_Russian-SynTagRus/master/ru_syntagrus-ud-test.conllu"
)
PUNCT = {",", ".", "!", "?"}
POS = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "SCONJ", "SYM", "VERB", "X",
}


def download_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(DEFAULT_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with path.open("wb") as f:
                while chunk := response.read(1024 * 1024):
                    f.write(chunk)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise SystemExit(
            f"Не удалось скачать корпус: {exc}\n"
            f"Скачай вручную {DEFAULT_URL}\n"
            f"и сохрани как {path}"
        ) from exc


def read_conllu(path):
    sentence = []
    sent_id = None
    bad_id = False

    def produce():
        if sentence and not bad_id:
            return sent_id, list(sentence)
        return None

    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                result = produce()
                if result:
                    yield result
                sentence, sent_id, bad_id = [], None, False
                continue
            if line.startswith("# sent_id = "):
                sent_id = line.partition("=")[2].strip()
            elif not line.startswith("#"):
                cols = line.split("\t")
                if len(cols) != 10:
                    bad_id = True
                    continue
                if not cols[0].isdigit():
                    # Multiword IDs (2-3), empty nodes (3.1), etc.
                    bad_id = True
                    continue
                _, form, lemma, upos = cols[:4]
                sentence.append({"form": form, "lemma": lemma, "pos": upos})
    result = produce()
    if result:
        yield result


def make_text(tokens):
    text = ""
    for tok in tokens:
        form = tok["form"]
        if form in PUNCT:
            text = text.rstrip() + form + " "
        else:
            text += form + " "
    return text.strip()


def select_sentences(path, count, seed, min_words, max_words):
    eligible = []
    seen = set()
    for source_id, tokens in read_conllu(path):
        words = [t for t in tokens if t["pos"] != "PUNCT"]
        if not min_words <= len(words) <= max_words:
            continue
        if not all(
            (t["form"] in PUNCT and t["pos"] == "PUNCT")
            or (
                t["pos"] in POS and t["form"].isalpha()
                and t["lemma"].isalpha() and "=" not in t["lemma"]
            )
            for t in tokens
        ):
            continue
        text = make_text(tokens)
        forms = re.sub(r"[,.!?]", " ", text).split()
        if forms != [t["form"] for t in words]:
            continue
        key = text.casefold().replace("ё", "е")
        if key in seen:
            continue
        seen.add(key)
        eligible.append((source_id, text, words))
    rng = random.Random(seed)
    if len(eligible) < count:
        raise SystemExit(
            f"Подходящих предложений {len(eligible)}, запрошено {count}. "
            "Уменьши --sentences или измени ограничения длины."
        )
    # Deterministic random sample; sorted to make output stable for comparisons.
    picked = sorted(rng.sample(range(len(eligible)), count))
    return [eligible[i] for i in picked], len(eligible)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/ru_syntagrus-ud-test.conllu")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--sentences", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1629)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=35)
    parser.add_argument("--out-dir", default="tests")
    args = parser.parse_args()
    source = Path(args.source)
    if not source.exists():
        if not args.download:
            raise SystemExit(
                f"Нет файла {source}. Запусти ещё раз с --download, "
                "либо скачай test.conllu вручную."
            )
        print("Скачиваем официальный test-сплит SynTagRus...")
        download_file(source)

    selected, eligible_count = select_sentences(
        source, args.sentences, args.seed, args.min_words, args.max_words
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inputs = []
    gold = []
    for n, (source_id, text, words) in enumerate(selected, start=1):
        inputs.append(text)
        gold.append({
            "id": f"corpus-{n:04d}",
            "source_sent_id": source_id,
            "text": text,
            "tokens": words,
        })

    (out / "corpus_input.txt").write_text("\n".join(inputs) + "\n", encoding="utf-8")
    with (out / "corpus_gold.jsonl").open("w", encoding="utf-8") as f:
        for row in gold:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "source": "UD Russian SynTagRus test split",
        "url": DEFAULT_URL,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seed": args.seed,
        "sentence_count": len(selected),
        "token_count": sum(len(row["tokens"]) for row in gold),
        "eligible_sentences": eligible_count,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "license": "CC BY-NC-SA 4.0 (verify with source repository for version)",
    }
    (out / "corpus_info.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Создан тест: {len(selected)} предложений, {metadata['token_count']} токенов")
    print(f"Ввод: {out / 'corpus_input.txt'}")
    print(f"Эталон: {out / 'corpus_gold.jsonl'}")
    print(f"Метаданные: {out / 'corpus_info.json'}")


if __name__ == "__main__":
    main()
