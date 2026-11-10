"""Проверки языка, диагностики, деревьев и обработки файлов."""

import itertools
import json
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from lexer import QueryError, tokenize
from main import analyze
from parser import parse_query
from render import html_report, svg_tree


CASES = json.loads((BASE / "tests" / "cases.json").read_text(encoding="utf-8"))


class LanguageTests(unittest.TestCase):
    def test_valid_queries_and_all_leaves(self):
        for query in CASES["valid"]:
            with self.subTest(query=query):
                tree = parse_query(query)
                # Дерево должно сохранять все токены в исходном порядке и регистре.
                self.assertEqual(
                    [(t.text, t.start) for t in tree.leaves()],
                    [(t.text, t.start) for t in tokenize(query) if t.kind != "EOF"],
                )

    def test_invalid_queries_and_exact_diagnostics(self):
        for case in CASES["invalid"]:
            with self.subTest(query=case["query"]):
                result = analyze(case["query"])
                self.assertIsNone(result["tree"])
                error = result["error"]
                self.assertEqual(error["actual"], case["actual"])
                self.assertEqual(error["token_number"], case["token"])
                self.assertEqual(error["stage"], case.get("stage", "Синтаксическая"))
                self.assertTrue(error["expected"])
                pos = error["column"] - 1
                if case["actual"] == "конец запроса":
                    self.assertEqual(pos, len(case["query"]))
                else:
                    self.assertTrue(case["query"][pos:].startswith(case["actual"]))

    def test_optional_filter_combinations(self):
        # Все 16 сочетаний четырёх необязательных блоков.
        filters = ['бренда "Dior"', "без ванили", "для лета", "дешевле 5000 рублей"]
        for mask in itertools.product((False, True), repeat=4):
            query = " ".join(["найди ароматы"] + [f for f, use in zip(filters, mask) if use])
            with self.subTest(query=query):
                parse_query(query)

    def test_inverted_filter_order_is_rejected(self):
        filters = ['бренда "Dior"', "без ванили", "для лета", "дешевле 5000 рублей"]
        for i in range(len(filters)):
            for j in range(i + 1, len(filters)):
                query = f"найди ароматы {filters[j]} {filters[i]}"
                with self.subTest(query=query), self.assertRaises(QueryError):
                    parse_query(query)

    def test_empty_and_unfinished_input(self):
        for query in ("", "   ", "найди", 'найди ароматы бренда', "найди ароматы от 5 до"):
            with self.subTest(query=query), self.assertRaises(QueryError):
                parse_query(query)

    def test_quotes_and_line_break(self):
        for query in ('найди ароматы бренда "   "', 'найди ароматы бренда "Dior\nParis"'):
            with self.subTest(query=query), self.assertRaises(QueryError):
                parse_query(query)

    def test_unknown_character_not_silently_dropped(self):
        for query in ("найди ароматы!", "найди ароматы 💐", "найди ароматы дешевле 5.5 рублей"):
            with self.subTest(query=query), self.assertRaises(QueryError):
                parse_query(query)

    def test_tabs_and_multiword_brand(self):
        tree = parse_query('Найди\tароматы бренда "Jo Malone London"')
        self.assertEqual(tree.leaves()[-1].text, '"Jo Malone London"')
        self.assertEqual(tree.leaves()[1].start, 6)

    def test_basic_tree_matches_rules(self):
        tree = parse_query("найди ароматы с нотой кедра")
        self.assertEqual([n.label for n in tree.children], ["Команда", "Объект", "Условие нот"])
        notes = tree.children[2]
        self.assertEqual([c.text for c in notes.children[:2]], ["с", "нотой"])
        self.assertEqual(notes.children[2].label, "Нота")

    def test_syntax_does_not_check_real_world_meaning(self):
        # Не существующий в каталоге бренд и обратный диапазон синтаксически допустимы.
        parse_query('найди ароматы бренда "Выдуманный" от 9000 до 100 рублей')


class OutputTests(unittest.TestCase):
    def test_html_escapes_user_input(self):
        query = 'найди ароматы бренда "<script>alert(1)</script> & perfume"'
        record = analyze(query)
        self.assertIsNotNone(record["tree"])
        report = html_report([record])
        self.assertNotIn("<script>alert(1)</script>", report)
        self.assertIn("&lt;script&gt;", report)
        ET.fromstring(svg_tree(record["tree"]))
        error_report = html_report([analyze("найди ароматы <script>")])
        self.assertNotIn("<script></pre>", error_report)

    def test_cli_bom_crlf_and_error_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            input_file = folder / "input.txt"
            input_file.write_bytes("\ufeffнайди ароматы\r\n\r\nнайди духи\r\nпокажи все ароматы\r\n".encode("utf-8"))
            run = subprocess.run(
                [sys.executable, str(BASE / "main.py"), "--input", str(input_file),
                 "--out-dir", str(folder / "output")],
                cwd=directory, capture_output=True, text=True, encoding="utf-8",
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            records = json.loads((folder / "output" / "results.json").read_text(encoding="utf-8"))
            self.assertEqual([r["success"] for r in records], [True, False, True])
            self.assertEqual([r["line"] for r in records], [1, 3, 4])
            self.assertIn("<svg", (folder / "output" / "report.html").read_text(encoding="utf-8"))
            self.assertTrue((folder / "output" / "results.txt").is_file())


if __name__ == "__main__":
    unittest.main()
