"""JSON-массив текстов -> JSON-массив рефератов. Пути заданы относительно запуска."""
import argparse
import json
import sys
from pathlib import Path

from config import load_config
from summarizer import summarize


ROOT = Path(__file__).resolve().parent


def write_text_report(texts, diagnostics, path):
    lines = ['РЕЗУЛЬТАТЫ РЕФЕРИРОВАНИЯ', 'Лимит: 300 символов, включая пробелы.', '']
    for number, (text, details) in enumerate(zip(texts, diagnostics), start=1):
        lines += ['=' * 70, f'ДОКУМЕНТ {number}', 'ИСХОДНЫЙ ТЕКСТ:', text,
                  '', f'РЕФЕРАТ ({details["length"]} символов):', details['summary'],
                  '', 'ВЫБРАННЫЕ ПРЕДЛОЖЕНИЯ:']
        for item in details['selected']:
            lines.append(f'{item["source_number"]}. {item["rule"]}: {item["text"]}')
        lines += [f'Запасное усечение: {"да" if details["fallback_used"] else "нет"}', '']
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Реферирование до 300 символов.')
    parser.add_argument('--input', default=str(ROOT / 'examples/input.json'), help='JSON-массив строк; - для stdin')
    parser.add_argument('--output', default=str(ROOT / 'results/summaries.json'), help='JSON результата; - для stdout')
    parser.add_argument('--config', help='Необязательный JSON с настройками')
    parser.add_argument('--explain', help='Отдельный JSON с оценками и выбранными фрагментами')
    parser.add_argument('--report', help='Читаемый текстовый отчёт с исходными документами и результатами')
    args = parser.parse_args()
    try:
        # Диагностика не должна перезаписать вход или основной результат.
        paths = [Path(p).resolve() for p in (args.input, args.output, args.explain, args.report, args.config) if p and p != '-']
        if len(paths) != len(set(paths)):
            raise ValueError('Вход, результат и диагностика должны быть разными файлами.')
        raw = sys.stdin.read() if args.input == '-' else Path(args.input).read_text(encoding='utf-8-sig')
        texts = json.loads(raw.lstrip('\ufeff'))
        if not isinstance(texts, list) or any(not isinstance(t, str) for t in texts):
            raise ValueError('Ожидается JSON-массив строк, например ["Первый текст", "Второй текст"].')
        config = load_config(args.config)
        summaries, diagnostics = [], []
        for text in texts:
            if args.explain or args.report:
                summary, details = summarize(text, config, explain=True)
                diagnostics.append(details)
            else:
                summary = summarize(text, config)
            summaries.append(summary)
        result = json.dumps(summaries, ensure_ascii=False, indent=2)
        if args.output == '-':
            print(result)
        else:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(result + '\n', encoding='utf-8')
            print(f'Готово: {len(summaries)} рефератов. Файл: {path}', file=sys.stderr)
        if args.explain:
            path = Path(args.explain)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if args.report:
            write_text_report(texts, diagnostics, args.report)
    except (OSError, ValueError) as error:
        parser.exit(2, f'Ошибка: {error}\n')


if __name__ == '__main__':
    main()
