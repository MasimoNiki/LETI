"""Разбиение строки на слова, числа и названия в кавычках."""

import re
from dataclasses import dataclass


@dataclass
class Token:
    kind: str
    text: str
    start: int                 # Индекс символа с нуля.
    end: int

    @property
    def value(self):
        return self.text.casefold()


class QueryError(Exception):
    """Одна ошибка разбора с позицией в исходной строке."""

    def __init__(self, stage, position, token_index, actual, expected, message):
        super().__init__(message)
        self.stage = stage
        self.position = position
        self.token_index = token_index
        self.actual = actual
        self.expected = expected
        self.message = message

    def to_dict(self):
        return {
            "stage": self.stage,
            "column": self.position + 1,
            "token_number": self.token_index + 1,
            "actual": self.actual,
            "expected": self.expected,
            "message": self.message,
        }


WORD = re.compile(r"[a-zA-Zа-яА-ЯёЁ]+(?:-[a-zA-Zа-яА-ЯёЁ]+)*")
NUMBER = re.compile(r"[0-9]+")


def tokenize(text):
    tokens = []
    pos = 0
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue

        if tokens and tokens[-1].end == pos:
            raise QueryError(
                "Лексическая", pos, len(tokens), text[pos],
                ["пробел между токенами"],
                "Между словами, числами и строками в кавычках нужен пробел.",
            )

        start = pos
        if text[pos] == '"':
            end = text.find('"', pos + 1)
            if end == -1:
                raise QueryError(
                    "Лексическая", start, len(tokens), text[start:],
                    ['закрывающая кавычка "'], "Не закрыто название в кавычках.",
                )
            if not text[pos + 1:end].strip():
                raise QueryError(
                    "Лексическая", start, len(tokens), text[start:end + 1],
                    ["непустая строка в кавычках"], "Название бренда не может быть пустым.",
                )
            if "\n" in text[pos + 1:end] or "\r" in text[pos + 1:end]:
                raise QueryError(
                    "Лексическая", start, len(tokens), text[start:end + 1],
                    ["название в одной строке"], "Перевод строки внутри названия запрещён.",
                )
            pos = end + 1
            kind = "STRING"
        else:
            match = NUMBER.match(text, pos) or WORD.match(text, pos)
            if match is None:
                raise QueryError(
                    "Лексическая", pos, len(tokens), text[pos],
                    ["слово", "целое неотрицательное число", 'строка в кавычках "..."'],
                    "Недопустимый символ. Пунктуация вне кавычек не поддерживается.",
                )
            pos = match.end()
            kind = "NUMBER" if text[start].isdigit() else "WORD"
        tokens.append(Token(kind, text[start:pos], start, pos))
    tokens.append(Token("EOF", "", len(text), len(text)))
    return tokens
