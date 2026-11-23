"""Оценка на JSONL/JSON с полями text и summary (эталон)."""
import argparse
import json
import random
import sys
from pathlib import Path
from time import perf_counter

from config import load_config
from metrics import compute_rouge, average_rouge
from summarizer import summarize


ROOT = Path(__file__).resolve().parent


def load_examples(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    rows = json.loads(text) if text.lstrip().startswith('[') else [json.loads(line) for line in text.split("\n") if line.strip()]
    if not isinstance(rows, list) or not rows:
        raise ValueError('Нужен непустой массив примеров или JSONL.')
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) for k in ('text', 'summary')):
            raise ValueError(f'Пример {i + 1}: нужны строковые поля text и summary.')
    return rows


def main():
    parser = argparse.ArgumentParser(description='Оценка рефератов: ROUGE и диагностический отчёт.')
    parser.add_argument('--input', default=str(ROOT / 'data/gazeta_sample20.jsonl'))
    parser.add_argument('--output', default=str(ROOT / 'results/evaluation.json'))
    parser.add_argument('--report', default=str(ROOT / 'results/review.txt'))
    parser.add_argument('--config')
    parser.add_argument('--limit', type=int, default=0, help='0 — все примеры; иначе случайная выборка')
    parser.add_argument('--seed', type=int, default=1629)
    args = parser.parse_args()
    try:
        if args.limit < 0:
            raise ValueError('limit не может быть отрицательным.')
        paths = [Path(p).resolve() for p in (args.input, args.output, args.report, args.config) if p]
        if len(paths) != len(set(paths)):
            raise ValueError('Вход и оба отчёта должны быть разными файлами.')
        data = load_examples(args.input)
        indices = list(range(len(data)))
        if args.limit and args.limit < len(data):
            indices = random.Random(args.seed).sample(indices, args.limit)
        config = load_config(args.config)
        rows, all_scores, old_scores = [], [], []
        started = perf_counter()
        for number, index in enumerate(indices, start=1):
            example = data[index]
            summary, details = summarize(example['text'], config, explain=True)
            scores = compute_rouge(example['summary'], summary)
            all_scores.append(scores)
            row = {'dataset_index': example.get('dataset_index', index),
                   'title': example.get('title', ''), 'url': example.get('url', ''),
                   'text': example['text'], 'reference': example['summary'],
                   'summary': summary, 'rouge': scores, 'details': details}
            if isinstance(example.get('baseline_summary'), str):
                row['baseline_summary'] = example['baseline_summary']
                old_scores.append(compute_rouge(example['summary'], example['baseline_summary']))
            rows.append(row)
            print(f'{number}/{len(indices)}: {row["title"]}', file=sys.stderr)
        result = {
            'count': len(rows), 'seed': args.seed, 'config': config,
            'protocol': 'Словесные ROUGE-1/2/L; lowercase, ё=е; без лемматизации/стоп-слов; macro P/R/F1; полные эталоны.',
            'mean_rouge': average_rouge(all_scores),
            'references_over_300': sum(len(r['reference']) > 300 for r in rows),
            'length_violations': sum(len(r['summary']) > 300 for r in rows),
            'fallback_count': sum(r['details']['fallback_used'] for r in rows),
            'elapsed_seconds': perf_counter() - started,
            'rows': rows,
        }
        if len(old_scores) == len(rows):
            result['stored_baseline_mean_rouge'] = average_rouge(old_scores)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        write_report(result, args.report)
        for name, scores in result['mean_rouge'].items():
            print(f'{name} F1: {scores["f1"]:.4f}')
        print(f'Отчёты: {args.output}, {args.report}')
    except (OSError, ValueError) as error:
        parser.exit(2, f'Ошибка: {error}\n')


def write_report(result, path):
    lines = ['CLASSIC SUMMARIZER: ОЦЕНКА', result['protocol'],
             f'Документов: {result["count"]}; превышений лимита: {result["length_violations"]}',
             'Настройки: ' + json.dumps(result['config'], ensure_ascii=False), '']
    for name, scores in result['mean_rouge'].items():
        lines.append(f'{name} F1: {scores["f1"]:.4f}')
    if 'stored_baseline_mean_rouge' in result:
        lines.append('Сохранённая исходная версия LSA (те же документы):')
        for name, scores in result['stored_baseline_mean_rouge'].items():
            lines.append(f'{name} F1: {scores["f1"]:.4f}')
    for number, row in enumerate(result['rows'], start=1):
        lines += ['', '=' * 80, f'ПРИМЕР {number}: {row["title"]}', row['url'],
                  'ИСХОДНЫЙ ТЕКСТ:', row['text'], '', 'ЭТАЛОН:', row['reference'],
                  '', f'НАШ РЕФЕРАТ ({len(row["summary"])} символов):', row['summary']]
        if 'baseline_summary' in row:
            lines += ['', 'СОХРАНЁННЫЙ LSA:', row['baseline_summary']]
        lines += ['', 'ВЫБРАННЫЕ ФРАГМЕНТЫ:']
        for selected in row['details']['selected']:
            lines.append(f'{selected["source_number"]}. {selected["rule"]}: {selected["text"]}')
        lines += ['', 'ВСЕ ПРЕДЛОЖЕНИЯ:']
        for sentence in row['details']['sentences']:
            lines.append(f'{sentence["number"]}. score={sentence["score"]:.3f}; '
                         f'graph={sentence["graph"]:.3f}; lead={sentence["lead"]:.3f}; '
                         f'position={sentence["position"]:.3f}; quality={sentence["quality"]:.2f} | {sentence["text"]}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
