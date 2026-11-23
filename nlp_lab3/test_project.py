"""Проверки ограничений задания и нескольких найденных ошибок."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from compression import compressed_versions
from config import get_config
from evaluate import load_examples
from metrics import compute_rouge
from preprocessing import prepare_text, split_sentences, tokenize_sentence
from ranking import lexrank
from selection import select_summary
from summarizer import summarize, summarize_many


class ProjectTests(unittest.TestCase):
    def test_empty_batch_and_documents(self):
        self.assertEqual(summarize_many([]), [])
        self.assertEqual(summarize_many(['', '  \n ', '... !!!']), ['', '', ''])

    def test_input_validation(self):
        for texts in ('text', {}, ['text', 3], [None]):
            with self.assertRaises(ValueError):
                summarize_many(texts)

    def test_length_limits_and_long_token(self):
        documents = ['слово ' * 300, 'А' * 1000,
                     'Учёные проверили новую технологию ' * 100 + '.',
                     'Станция работает. Насос не работает.']
        for limit in (1, 30, 150, 300):
            for text in documents:
                result = summarize(text, {'max_chars': limit})
                self.assertLessEqual(len(result), limit)

    def test_length_cannot_exceed_assignment(self):
        for value in (301, 0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                get_config({'max_chars': value})

    def test_configuration_validation(self):
        for overrides in ({'unknown': 1}, {'damping': 1}, {'lead_weight': float('nan')},
                          {'compression': 'yes'}):
            with self.assertRaises(ValueError):
                get_config(overrides)

    def test_batch_order_and_independence(self):
        texts = ['Завод не будет закрыт. Рабочие продолжат работу.',
                 'Учёные создали аккумулятор. Он заряжается за десять минут.']
        self.assertEqual(summarize_many(texts), [summarize(t) for t in texts])
        self.assertEqual(summarize_many(texts[::-1]), summarize_many(texts)[::-1])

    def test_lemma_and_negation(self):
        words = tokenize_sentence('Новые аккумуляторы не работают без зарядки.')
        self.assertIn('аккумулятор', words)
        self.assertIn('не', words)
        self.assertIn('без', words)

    def test_title_with_period(self):
        text = 'Похитили картину «Ай-Петри. Крым» из музея. Полиция начала поиск.'
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertIn('Ай-Петри. Крым', sentences[0])

    def test_multisentence_quote_is_not_merged(self):
        text = '«Мы закончили работу. Завтра мы начнем испытания», — сказал инженер.'
        self.assertGreaterEqual(len(split_sentences(text)), 2)

    def test_compression_protects_conditions(self):
        text = 'Завод, работающий без лицензии с 2020 года, прекратил производство.'
        for version, rule in compressed_versions(text):
            self.assertIn('без лицензии с 2020 года', version)

    def test_compression_of_filler(self):
        text = 'Кроме того, исследователи проверили новый аккумулятор в лаборатории.'
        versions = [s for s, _ in compressed_versions(text)]
        self.assertIn('Исследователи проверили новый аккумулятор в лаборатории.', versions)
        text = 'При этом заболевании требуется дополнительное обследование пациента.'
        self.assertEqual(compressed_versions(text)[0][0], text)
        self.assertEqual(len(compressed_versions(text)), 1)

    def test_compression_removes_both_commas(self):
        text = 'Инженер, работающий в местной лаборатории, проверил новый двигатель.'
        versions = [s for s, _ in compressed_versions(text)]
        self.assertIn('Инженер проверил новый двигатель.', versions)

    def test_graph_with_no_edges(self):
        scores = lexrank(np.eye(3))
        self.assertTrue(np.isfinite(scores).all())
        self.assertAlmostEqual(float(scores.sum()), 1)
        np.testing.assert_allclose(scores, np.ones(3) / 3)

    def test_rouge_known_values(self):
        score = compute_rouge('Учёные создали новый аккумулятор.',
                              'Учёные создали мощный аккумулятор.')
        self.assertAlmostEqual(score['rouge1']['f1'], 0.75)
        self.assertAlmostEqual(score['rouge2']['f1'], 1 / 3)
        self.assertAlmostEqual(score['rougeL']['f1'], 0.75)
        self.assertEqual(compute_rouge('', '')['rouge1']['f1'], 0)
        self.assertEqual(compute_rouge('кот кот', 'кот')['rouge1']['recall'], 0.5)

    def test_jsonl_unicode_separator(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data.jsonl'
            example = {'text': 'Строка\u2028продолжается.', 'summary': 'Реферат.'}
            path.write_text(json.dumps(example, ensure_ascii=False) + '\n', encoding='utf-8')
            self.assertEqual(load_examples(path), [example])

    def test_details_match_output(self):
        summary, details = summarize('Завод построили в городе. Завод начнет работу завтра.', explain=True)
        self.assertEqual(summary, ' '.join(s['text'] for s in details['selected']))
        self.assertEqual(details['length'], len(summary))
        numbers = [s['source_number'] for s in details['selected']]
        self.assertEqual(numbers, sorted(set(numbers)))

    def test_dependent_candidate_waits_for_previous_sentence(self):
        sentences = ['Инженеры создали двигатель.', 'Он работает тише прежней модели.']
        candidates = [
            {'text': sentences[0], 'source_index': 0, 'rule': 'исходное', 'score': 0.7,
             'vector': np.array([1., 0.]), 'dependent_start': False},
            {'text': sentences[1], 'source_index': 1, 'rule': 'исходное', 'score': 1.0,
             'vector': np.array([0., 1.]), 'dependent_start': True},
        ]
        summary, selected, steps, _ = select_summary(sentences, np.array([0.7, 1.]), candidates, get_config())
        self.assertEqual(steps[0]['source_number'], 1)
        self.assertEqual([c['source_number'] for c in selected], [1, 2])
        self.assertEqual(summary, ' '.join(sentences))

    def test_context_sentence_does_not_fit_with_dependent_one(self):
        sentences = ['Инженеры создали двигатель.', 'Об этом сообщили разработчики.']
        candidates = [
            {'text': sentences[0], 'source_index': 0, 'rule': 'исходное', 'score': 0.6,
             'vector': np.array([1., 0.]), 'dependent_start': False},
            {'text': sentences[1], 'source_index': 1, 'rule': 'исходное', 'score': 1.,
             'vector': np.array([0., 1.]), 'dependent_start': True},
        ]
        summary, selected, _, _ = select_summary(sentences, np.array([0.6, 1.]), candidates,
                                                 get_config({'max_chars': 35}))
        self.assertEqual(summary, sentences[0])

    def test_cli_bom_report_and_foreign_working_directory(self):
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as folder:
            output, report = Path(folder) / 'output.json', Path(folder) / 'report.txt'
            run = subprocess.run([sys.executable, str(root / 'main.py'), '--input', '-',
                                  '--output', str(output), '--report', str(report)], cwd=folder,
                                 input='\ufeff' + json.dumps(['Договор расторгнут.'], ensure_ascii=False),
                                 text=True, encoding='utf-8', capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            summary = json.loads(output.read_text(encoding='utf-8'))[0]
            self.assertIn(summary, report.read_text(encoding='utf-8'))

    def test_cli_report_cannot_overwrite_input(self):
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.json'
            content = '["Договор расторгнут."]'
            path.write_text(content, encoding='utf-8')
            run = subprocess.run([sys.executable, str(root / 'main.py'), '--input', str(path),
                                  '--report', str(path)], text=True, encoding='utf-8', capture_output=True)
            self.assertEqual(run.returncode, 2)
            self.assertEqual(path.read_text(encoding='utf-8'), content)

    def test_cli_array_contract(self):
        root = Path(__file__).resolve().parent
        run = subprocess.run([sys.executable, str(root / 'main.py'), '--input', '-', '--output', '-'],
                             input=json.dumps(['Договор расторгнут.', ''], ensure_ascii=False),
                             text=True, encoding='utf-8', capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        output = json.loads(run.stdout)
        self.assertEqual(len(output), 2)
        self.assertEqual(output[1], '')
        self.assertTrue(all(isinstance(s, str) and len(s) <= 300 for s in output))

    def test_cli_rejects_non_array(self):
        root = Path(__file__).resolve().parent
        run = subprocess.run([sys.executable, str(root / 'main.py'), '--input', '-', '--output', '-'],
                             input='{"text":"document"}', text=True, capture_output=True)
        self.assertEqual(run.returncode, 2)
        self.assertEqual(run.stdout, '')


if __name__ == '__main__':
    unittest.main(verbosity=2)
