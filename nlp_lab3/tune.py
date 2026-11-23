"""Небольшой подбор весов на валидации; итоговая программа использует один конфиг."""
import argparse
import json
import random
from pathlib import Path

from config import get_config
from evaluate import load_examples
from metrics import average_rouge, compute_rouge
from preprocessing import prepare_text
from ranking import rank_sentences
from selection import make_candidates, select_summary
from tfidf import build_tfidf


def main():
    parser = argparse.ArgumentParser(description='Подбор на валидационной, НЕ тестовой выборке.')
    parser.add_argument('--input', required=True, help='Gazeta JSONL/JSON, часть validation')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1629)
    parser.add_argument('--output', default='results/tuning.json')
    parser.add_argument('--best-config', default='results/best_config.json')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit должен быть положительным')
    paths = [Path(p).resolve() for p in (args.input, args.output, args.best_config)]
    if len(paths) != len(set(paths)):
        parser.error('Вход и оба результата должны быть разными файлами')
    data = load_examples(args.input)
    indices = random.Random(args.seed).sample(range(len(data)), min(args.limit, len(data)))
    prepared = []
    for index in indices:
        example = data[index]
        sentences, tokens = prepare_text(example['text'])
        matrix, vocabulary, idf = build_tfidf(tokens)
        prepared.append((example, sentences, matrix, vocabulary, idf))
    # Проверяем небольшое число осмысленных комбинаций, не сотни случайных правил.
    weight_sets = [(0.25, 0.35, 0.40), (0.40, 0.30, 0.30),
                   (0.20, 0.50, 0.30), (0.30, 0.20, 0.50)]
    experiments = []
    for graph_weight, lead_weight, position_weight in weight_sets:
        for redundancy in (0.20, 0.35):
            config = get_config({'graph_weight': graph_weight, 'lead_weight': lead_weight,
                                 'position_weight': position_weight, 'redundancy_weight': redundancy})
            results = []
            for example, sentences, matrix, vocabulary, idf in prepared:
                scores, features, similarity = rank_sentences(sentences, matrix, config)
                candidates = make_candidates(sentences, scores, vocabulary, idf, features, config)
                summary, selected, steps, fallback = select_summary(sentences, scores, candidates, config)
                results.append(compute_rouge(example['summary'], summary))
            metrics = average_rouge(results)
            # Критерий фиксируем до сравнения: среднее трёх F1.
            objective = sum(metrics[k]['f1'] for k in metrics) / 3
            experiments.append({'config': config, 'mean_rouge': metrics, 'objective': objective})
            print(f'{len(experiments)}/8: среднее F1 = {objective:.4f}')
    experiments.sort(key=lambda row: row['objective'], reverse=True)
    result = {'seed': args.seed, 'indices': indices, 'count': len(indices),
              'criterion': 'mean(ROUGE-1 F1, ROUGE-2 F1, ROUGE-L F1)',
              'note': 'Это результат подбора, независимую оценку выполняйте на другом наборе.',
              'experiments': experiments}
    for path, content in ((args.output, result), (args.best_config, experiments[0]['config'])):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(content, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Сохранён конфиг:', args.best_config)


if __name__ == '__main__':
    main()
