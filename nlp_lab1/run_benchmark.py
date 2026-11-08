"""Проверка точности по корпусу и ручным примерам."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from evaluate_benchmark import score

ROOT = Path(__file__).resolve().parent


def call(*args):
    print("\n>", sys.executable, " ".join(str(x) for x in args), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def save_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Сборка теста и оценка лемматизатора")
    p.add_argument("--sentences", type=int, default=500)
    p.add_argument("--seed", type=int, default=1629)
    p.add_argument("--download", action="store_true")
    p.add_argument("--rebuild", action="store_true", help="Создать корпусный тест заново")
    p.add_argument("--source", default="data/ru_syntagrus-ud-test.conllu")
    args = p.parse_args()

    test_dir = ROOT / "tests"
    required = (test_dir / "corpus_input.txt", test_dir / "corpus_gold.jsonl")
    if args.rebuild or args.download or not all(path.exists() for path in required):
        command = [
            "make_testset.py", "--source", args.source, "--sentences", str(args.sentences),
            "--seed", str(args.seed),
        ]
        if args.download:
            command.append("--download")
        call(*command)
    elif args.sentences != 500 or args.seed != 1629:
        raise SystemExit("Для изменения числа предложений или seed используй --rebuild")

    result_dir = ROOT / "results" / "benchmark"
    result_dir.mkdir(parents=True, exist_ok=True)

    corpus_input = (test_dir / "corpus_input.txt").read_text(encoding="utf-8").splitlines()
    manual_input = (test_dir / "manual_input.txt").read_text(encoding="utf-8").splitlines()
    if not corpus_input:
        raise SystemExit("Нет предложений корпуса для тестирования")

    combined_input = result_dir / "_combined_input.txt"
    combined_output = result_dir / "_combined_output.txt"
    combined_details = result_dir / "_combined_details.jsonl"
    save_lines(combined_input, corpus_input + manual_input)

    call(
        "main.py", "--input", combined_input, "--output", combined_output,
        "--details", combined_details,
    )

    all_outputs = combined_output.read_text(encoding="utf-8").splitlines()
    all_details = combined_details.read_text(encoding="utf-8").splitlines()
    if len(all_outputs) != len(corpus_input) + len(manual_input):
        raise ValueError("Число строк вывода не совпало с числом тестовых предложений")
    if len(all_details) != len(all_outputs):
        raise ValueError("Число строк деталей не совпало с числом тестовых предложений")

    n = len(corpus_input)
    sections = {
        "corpus": (all_outputs[:n], all_details[:n]),
        "manual": (all_outputs[n:], all_details[n:]),
    }
    for name, (outputs, details) in sections.items():
        print(f"\n=== {name.upper()} ===", flush=True)
        output_path = result_dir / f"{name}_output.txt"
        details_path = result_dir / f"{name}_details.jsonl"
        save_lines(output_path, outputs)
        save_lines(details_path, details)
        score(
            test_dir / f"{name}_gold.jsonl", output_path, details_path,
            result_dir / name,
        )
    print("\nГотово. Отчёты: results/benchmark/corpus/report.md и manual/report.md")


if __name__ == "__main__":
    main()
