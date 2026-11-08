"""Evaluate token{lemma=UPOS} outputs against per-sentence JSONL gold.

Normalizes case and е/ё for lemma matching, as required by the lab.
Does not depend on model internals; optional details.jsonl adds source breakdown.
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PRED_RE = re.compile(r"^([^\s{}]+)\{([^\s{}=]+)=([^\s{}=]+)\}$")


def normalize(s):
    return s.casefold().replace("ё", "е")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(n, d):
    return round(n * 100 / d, 2) if d else 0.0


def score(gold_path, output_path, details_path=None, out_dir="results/benchmark"):
    gold = read_jsonl(gold_path)
    predicted = Path(output_path).read_text(encoding="utf-8-sig").splitlines()
    if len(predicted) != len(gold):
        raise ValueError(
            f"Несовпадение количества строк: gold={len(gold)}, "
            f"output={len(predicted)}. Проверь пустые строки во входе."
        )
    details = read_jsonl(details_path) if details_path else None
    if details is not None and len(details) != len(gold):
        raise ValueError(f"В details {len(details)} строк, ожидалось {len(gold)}")

    total = 0
    lemma_ok = pos_ok = pair_ok = 0
    by_pos = defaultdict(lambda: Counter())
    by_source = defaultdict(lambda: Counter())
    confusions = Counter()
    errors = []

    for i, (truth, line) in enumerate(zip(gold, predicted)):
        tokens = truth["tokens"]
        chunks = line.split()
        if len(chunks) != len(tokens):
            raise ValueError(
                f"Строка {i+1}: ожидается {len(tokens)} токенов, "
                f"получено {len(chunks)}. Строка: {line[:160]!r}"
            )
        detail_tokens = details[i].get("tokens", []) if details else None
        if detail_tokens is not None and len(detail_tokens) != len(tokens):
            raise ValueError(f"Строка {i+1}: неверное число токенов в details")

        for j, (wanted, chunk) in enumerate(zip(tokens, chunks), start=1):
            match = PRED_RE.fullmatch(chunk)
            if not match:
                raise ValueError(
                    f"Неверный формат в строке {i+1}, токен {j}: {chunk!r}"
                )
            form, lemma, pos = match.groups()
            if form != wanted["form"]:
                raise ValueError(
                    f"Строка {i+1}, токен {j}: ожидался {wanted['form']!r}, "
                    f"получен {form!r}. Ввод/вывод не совпадают."
                )
            gold_lemma, gold_pos = wanted["lemma"], wanted["pos"]
            a = normalize(lemma) == normalize(gold_lemma)
            b = pos == gold_pos
            c = a and b
            total += 1
            lemma_ok += a
            pos_ok += b
            pair_ok += c
            by_pos[gold_pos].update(total=1, lemma=int(a), pos=int(b), pair=int(c))
            confusions[(gold_pos, pos)] += 1

            source = "не указан"
            if detail_tokens is not None:
                dt = detail_tokens[j - 1]
                if dt.get("token") != form:
                    raise ValueError(f"Строка {i+1}, токен {j}: details не совпал")
                chosen = dt.get("chosen", [])
                if len(chosen) < 3 or normalize(chosen[0]) != normalize(lemma) or chosen[1] != pos:
                    raise ValueError(f"Строка {i+1}, токен {j}: chosen не совпал")
                source = chosen[2]
                by_source[source].update(total=1, lemma=int(a), pos=int(b), pair=int(c))

            if not c:
                errors.append({
                    "sent_id": truth.get("id", i + 1),
                    "token_no": j,
                    "form": form,
                    "gold_lemma": gold_lemma,
                    "pred_lemma": lemma,
                    "gold_pos": gold_pos,
                    "pred_pos": pos,
                    "source": source,
                })

    metrics = {
        "sentences": len(gold),
        "tokens": total,
        "lemma_accuracy": pct(lemma_ok, total),
        "pos_accuracy": pct(pos_ok, total),
        "pair_accuracy": pct(pair_ok, total),
        "lemma_correct": lemma_ok,
        "pos_correct": pos_ok,
        "pair_correct": pair_ok,
        "errors_count": len(errors),
        "by_pos": {
            pos: {
                "tokens": count["total"],
                "lemma_accuracy": pct(count["lemma"], count["total"]),
                "pos_accuracy": pct(count["pos"], count["total"]),
                "pair_accuracy": pct(count["pair"], count["total"]),
            }
            for pos, count in sorted(by_pos.items())
        },
        "by_source": {
            source: {
                "tokens": count["total"],
                "lemma_accuracy": pct(count["lemma"], count["total"]),
                "pos_accuracy": pct(count["pos"], count["total"]),
                "pair_accuracy": pct(count["pair"], count["total"]),
            }
            for source, count in sorted(by_source.items())
        },
        "common_pos_confusions": [
            {"gold": x, "pred": y, "count": n}
            for (x, y), n in confusions.most_common() if x != y
        ][:15],
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (out / "errors.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=(
            "sent_id", "token_no", "form", "gold_lemma", "pred_lemma",
            "gold_pos", "pred_pos", "source"
        ), delimiter="\t")
        writer.writeheader()
        writer.writerows(errors)

    lines = [
        "# Оценка лемматизатора",
        "",
        f"Предложений: **{metrics['sentences']}**",
        f"Токенов: **{total}**",
        f"Lemma accuracy: **{metrics['lemma_accuracy']:.2f}%** ({lemma_ok}/{total})",
        f"POS accuracy: **{metrics['pos_accuracy']:.2f}%** ({pos_ok}/{total})",
        f"Pair accuracy: **{metrics['pair_accuracy']:.2f}%** ({pair_ok}/{total})",
        "",
        "## Результаты по частям речи",
        "",
        "| POS (эталон) | Токенов | Лемма | POS | Пара |",
        "|---|---:|---:|---:|---:|",
    ]
    for pos, row in sorted(metrics["by_pos"].items(), key=lambda x: -x[1]["tokens"]):
        lines.append(
            f"| {pos} | {row['tokens']} | {row['lemma_accuracy']:.2f}% | "
            f"{row['pos_accuracy']:.2f}% | {row['pair_accuracy']:.2f}% |"
        )
    if metrics["by_source"]:
        lines.extend(["", "## Источник разбора (из details.jsonl)", "",
                      "| Источник | Токенов | Лемма | POS | Пара |",
                      "|---|---:|---:|---:|---:|"])
        for source, row in metrics["by_source"].items():
            lines.append(
                f"| {source} | {row['tokens']} | {row['lemma_accuracy']:.2f}% | "
                f"{row['pos_accuracy']:.2f}% | {row['pair_accuracy']:.2f}% |"
            )
    lines.extend(["", "## Ограничения", "",
                  "Оценка проводится на токенах. Леммы сопоставляются без учёта регистра и различия е/ё.",
                  "Теги POS сравниваются по стандарту Universal Dependencies без автоматического объединения классов.",
                  "Знакомые словоформы в тесте возможны — это не OOV-only benchmark.",
                  "Разметка корпуса и словарная нормализация могут иметь отдельные конвенционные различия.", ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Предложений: {len(gold)}, токенов: {total}")
    print(f"Леммы: {metrics['lemma_accuracy']:.2f}%")
    print(f"POS:    {metrics['pos_accuracy']:.2f}%")
    print(f"Пары:   {metrics['pair_accuracy']:.2f}%")
    print(f"Ошибки: {len(errors)}; отчёт: {out / 'report.md'}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--details", default=None)
    parser.add_argument("--out-dir", default="results/benchmark")
    args = parser.parse_args()
    score(args.gold, args.output, args.details, args.out_dir)


if __name__ == "__main__":
    main()
