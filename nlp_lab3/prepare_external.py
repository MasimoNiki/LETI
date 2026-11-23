"""Воспроизводимая новая выборка. Код суммаризатора и его веса не изменяются."""
import argparse
import hashlib
import json
import random
import re
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'gazeta_test.jsonl': 'https://www.dropbox.com/scl/fi/mxjt7x0b416uf188stp6o/gazeta_test.jsonl?rlkey=n61ftoywfnqvg9mpmpypojqot&dl=1',
    'russian_XLSum_v2.0.tar.bz2': 'https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/data/russian_XLSum_v2.0.tar.bz2',
}
ALGORITHM_FILES = ['preprocessing.py', 'tfidf.py', 'ranking.py', 'compression.py',
                   'selection.py', 'summarizer.py', 'config.py', 'config.json', 'metrics.py']


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(text):
    return ' '.join(re.findall(r'[a-zа-я0-9]+', text.lower().replace('ё', 'е')))


def shingles(text):
    words = fingerprint(text).split()
    return {tuple(words[i:i+5]) for i in range(len(words) - 4)}


def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as file:
        return [json.loads(line) for line in file if line.strip()]


def download(url, path):
    if path.exists():
        return
    temporary = path.with_suffix(path.suffix + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'lab3-evaluation/1.0'})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file.write(chunk)
    temporary.replace(path)


def sample_rows(rows, count, seed, seen_texts, seen_urls, seen_shingles, dataset):
    indexed = list(enumerate(rows))
    random.Random(seed).shuffle(indexed)
    selected = []
    rejected = {'empty_or_invalid': 0, 'duplicate': 0, 'near_duplicate': 0}
    for index, row in indexed:
        text, summary = row.get('text'), row.get('summary')
        if not isinstance(text, str) or not isinstance(summary, str) or not text.strip() or not summary.strip():
            rejected['empty_or_invalid'] += 1
            continue
        key = fingerprint(text)
        url = row.get('url', '')
        if key in seen_texts or (url and url in seen_urls):
            rejected['duplicate'] += 1
            continue
        current = shingles(text)
        duplicate = False
        if current:
            for previous in seen_shingles:
                # Проверяем только пары, у которых вообще возможен Jaccard >= 0.9.
                if previous and min(len(current), len(previous)) / max(len(current), len(previous)) >= 0.9:
                    overlap = len(current & previous)
                    if overlap / (len(current) + len(previous) - overlap) >= 0.9:
                        duplicate = True
                        break
        if duplicate:
            rejected['near_duplicate'] += 1
            continue
        selected.append({'dataset': dataset, 'source_split': 'test', 'dataset_index': index,
                         'source_id': str(row.get('id', index)), 'url': url,
                         'title': row.get('title', ''), 'text': text, 'summary': summary})
        seen_texts.add(key)
        if url:
            seen_urls.add(url)
        seen_shingles.append(current)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f'{dataset}: удалось выбрать только {len(selected)} документов')
    return selected, rejected


def main():
    parser = argparse.ArgumentParser(description='Новые test-выборки Gazeta и XL-Sum без изменения модели.')
    parser.add_argument('--cache', default=str(ROOT / 'data' / 'external_cache'))
    parser.add_argument('--count', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20261123)
    parser.add_argument('--project', default=str(ROOT), help='Корень исходного проекта с прежними 120 статьями')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('count должен быть положительным')
    project, cache = Path(args.project), Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    hashes = {name: sha256_file(project / name) for name in ALGORITHM_FILES}
    old_rows = []
    for name in ('gazeta_sample20.jsonl', 'gazeta_check100.jsonl'):
        old_rows.extend(read_jsonl(project / 'data' / name))
    seen_texts = {fingerprint(row['text']) for row in old_rows}
    seen_urls = {row['url'] for row in old_rows if row.get('url')}
    seen_shingles = [shingles(row['text']) for row in old_rows]
    for name, url in SOURCES.items():
        print('Источник:', name, flush=True)
        download(url, cache / name)
    gazeta = read_jsonl(cache / 'gazeta_test.jsonl')
    # Читаем только JSONL test из архива, ничего из архива не исполняем и не извлекаем на диск.
    with tarfile.open(cache / 'russian_XLSum_v2.0.tar.bz2', 'r:bz2') as archive:
        matches = [m for m in archive.getmembers() if m.isfile() and m.name.endswith('russian_test.jsonl')]
        if len(matches) != 1:
            raise ValueError('В архиве XL-Sum не найден единственный russian_test.jsonl')
        with archive.extractfile(matches[0]) as file:
            xlsum = [json.loads(line.decode('utf-8')) for line in file if line.strip()]
    manifest = {'seed': args.seed, 'count_per_source': args.count,
                'excluded_previous_documents': len(old_rows), 'algorithm_sha256': hashes,
                'config': json.loads((project / 'config.json').read_text(encoding='utf-8')),
                'sources': {name: {'url': url, 'sha256': sha256_file(cache / name)} for name, url in SOURCES.items()},
                'selection': 'Случайный порядок с фиксированным seed; исключены пустые, совпадающие URL/нормализованные тексты и близкие тексты (Jaccard 5-грамм >=0.9). По длине и качеству рефератов отбор не выполняется.',
                'datasets': {}}
    for dataset, rows, seed in [('gazeta', gazeta, args.seed), ('xlsum_ru', xlsum, args.seed + 1)]:
        selected, rejected = sample_rows(rows, args.count, seed, seen_texts, seen_urls, seen_shingles, dataset)
        path = ROOT / 'data' / f'external_{dataset}{args.count}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in selected) + '\n', encoding='utf-8')
        manifest['datasets'][dataset] = {'source_rows': len(rows), 'count': len(selected),
                                        'indices': [r['dataset_index'] for r in selected],
                                        'rejected_before_filling_sample': rejected,
                                        'file': str(path.relative_to(ROOT)), 'sha256': sha256_file(path)}
        print('Готово:', path.name, flush=True)
    path = ROOT / 'data' / 'external_manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
