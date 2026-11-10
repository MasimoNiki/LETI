"""Запуск: python main.py. Все зависимости входят в стандартную библиотеку."""

import argparse
import json
import sys
from pathlib import Path

from lexer import QueryError
from parser import parse_query
from render import html_report, text_result


BASE = Path(__file__).resolve().parent


def analyze(text, line=1):
    record = {"line": line, "query": text, "tree": None, "error": None}
    try:
        record["tree"] = parse_query(text)
    except QueryError as error:
        record["error"] = error.to_dict()
    return record


def read_queries(path):
    # utf-8-sig читает обычный UTF-8 и файлы с BOM, в том числе из Windows.
    # Пустые строки пропускаем, номера исходных строк сохраняем.
    text = path.read_text(encoding="utf-8-sig")
    return [(number, line) for number, line in enumerate(text.splitlines(), 1) if line.strip()]


def save_reports(records, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    text = "\n\n".join(text_result(r) for r in records)
    (out_dir / "results.txt").write_text(text + "\n", encoding="utf-8")
    (out_dir / "report.html").write_text(html_report(records), encoding="utf-8")
    json_records = [
        {"line": r["line"], "query": r["query"], "success": r["tree"] is not None,
         "tree": r["tree"].to_dict() if r["tree"] is not None else None, "error": r["error"]}
        for r in records
    ]
    (out_dir / "results.json").write_text(
        json.dumps(json_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    arg_parser = argparse.ArgumentParser(description="Синтаксический анализ запросов о парфюмерии")
    source = arg_parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="Файл UTF-8: один запрос на строке")
    source.add_argument("--query", help="Один запрос вместо файла")
    source.add_argument("--interactive", action="store_true", help="Ввод запросов с клавиатуры")
    arg_parser.add_argument("--out-dir", type=Path, default=BASE / "results", help="Папка отчётов")
    args = arg_parser.parse_args()

    if args.interactive:
        print("Вводи запросы. Пустая строка или Ctrl+C — выход.")
        records = []
        while True:
            try:
                query = input("Запрос> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query.strip():
                break
            record = analyze(query, len(records) + 1)
            records.append(record)
            print(text_result(record), end="\n\n")
    elif args.query is not None:
        records = [analyze(args.query)]
    else:
        try:
            queries = read_queries(args.input or BASE / "queries.txt")
        except (OSError, UnicodeError) as error:
            arg_parser.error(f"Не удалось прочитать входной файл UTF-8: {error}")
        records = [analyze(query, line) for line, query in queries]

    try:
        save_reports(records, args.out_dir)
    except OSError as error:
        arg_parser.error(f"Не удалось сохранить отчёты: {error}")
    if not args.interactive:
        for record in records:
            print(text_result(record), end="\n\n")
    ok = sum(r["tree"] is not None for r in records)
    print(f"Всего: {len(records)}. Успех: {ok}. Неудача: {len(records) - ok}.")
    print(f"Деревья и ошибки: {(args.out_dir / 'report.html').resolve()}")
    # Ошибочные запросы — ожидаемые входные данные лабораторной, а не сбой программы.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
