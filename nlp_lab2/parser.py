"""Рекурсивный спуск: функции соответствуют правилам grammar.ebnf."""

from dataclasses import dataclass, field

from lexer import QueryError, Token, tokenize


COMMANDS = ("найди", "покажи", "выведи")
FAMILIES = ("древесные", "цветочные", "цитрусовые", "восточные", "свежие")
NOTES = ("бергамота", "кедра", "ванили", "розы", "жасмина", "мускуса", "ветивера", "сандала")
SEASONS = ("зимы", "весны", "лета", "осени")


@dataclass
class Node:
    label: str
    children: list = field(default_factory=list)

    def leaves(self):
        """Все слова запроса слева направо, включая служебные слова."""
        result = []
        for child in self.children:
            result.extend(child.leaves() if isinstance(child, Node) else [child])
        return result

    def to_dict(self):
        return {
            "label": self.label,
            "children": [
                child.to_dict() if isinstance(child, Node)
                else {"token": child.text, "kind": child.kind, "column": child.start + 1}
                for child in self.children
            ],
        }


class Parser:
    def __init__(self, text):
        self.text = text
        self.tokens = tokenize(text)
        self.pos = 0

    @property
    def current(self):
        return self.tokens[self.pos]

    def fail(self, expected, message="Токен не соответствует правилу грамматики."):
        token = self.current
        raise QueryError(
            "Синтаксическая", token.start, self.pos,
            token.text if token.kind != "EOF" else "конец запроса",
            list(expected), message,
        )

    def take(self):
        token = self.current
        self.pos += 1
        return token

    def expect(self, *words):
        if self.current.kind != "WORD" or self.current.value not in words:
            self.fail(words)
        return self.take()

    def parse_number(self):
        if self.current.kind != "NUMBER":
            self.fail(["целое неотрицательное число"])
        return Node("Число", [self.take()])

    def parse_note(self):
        return Node("Нота", [self.expect(*NOTES)])

    def parse_note_list(self, plural=False):
        label = "Список нот" if plural else "Перечень нот"
        node = Node(label, [self.parse_note()])
        # После «с нотами» требуются минимум две ноты.
        if plural:
            node.children.extend([self.expect("и"), self.parse_note()])
        while self.current.value == "и":
            node.children.extend([self.take(), self.parse_note()])
        return node

    def parse_notes(self):
        node = Node("Условие нот", [self.expect("с", "без")])
        if node.children[0].value == "без":
            node.children.append(self.parse_note_list())
        else:
            marker = self.expect("нотой", "нотами")
            node.children.append(marker)
            node.children.append(
                self.parse_note() if marker.value == "нотой"
                else self.parse_note_list(plural=True)
            )
        return node

    def parse_brand(self):
        marker = self.expect("бренда")
        if self.current.kind != "STRING":
            self.fail(['название бренда в двойных кавычках: "Dior"'])
        return Node("Бренд", [marker, Node("Название бренда", [self.take()])])

    def parse_season(self):
        return Node("Сезон", [
            self.expect("для"), Node("Название сезона", [self.expect(*SEASONS)])
        ])

    def parse_price(self):
        marker = self.expect("дешевле", "дороже", "от")
        node = Node("Цена", [marker, self.parse_number()])
        if marker.value == "от":
            node.children.extend([self.expect("до"), self.parse_number()])
        node.children.append(self.expect("рублей"))
        return node

    def parse(self):
        root = Node("Запрос", [Node("Команда", [self.expect(*COMMANDS)])])
        if self.current.value == "все":
            root.children.append(self.take())
        if self.current.value in FAMILIES:
            root.children.append(Node("Семейство", [self.take()]))
        root.children.append(Node("Объект", [self.expect("ароматы")]))

        # Проверка необязательных частей в порядке из грамматики.
        if self.current.value == "бренда":
            root.children.append(self.parse_brand())
        if self.current.value in ("с", "без"):
            root.children.append(self.parse_notes())
        if self.current.value == "для":
            root.children.append(self.parse_season())
        if self.current.value in ("дешевле", "дороже", "от"):
            root.children.append(self.parse_price())

        if self.current.kind != "EOF":
            # Допустимые продолжения зависят от уже разобранных частей.
            labels = {child.label for child in root.children if isinstance(child, Node)}
            expected = []
            if not labels & {"Бренд", "Условие нот", "Сезон", "Цена"}:
                expected.append("бренда")
            if not labels & {"Условие нот", "Сезон", "Цена"}:
                expected.extend(["с", "без"])
            if not labels & {"Сезон", "Цена"}:
                expected.append("для")
            if "Цена" not in labels:
                expected.extend(["дешевле", "дороже", "от"])
            expected.append("конец запроса")
            self.fail(expected, "Лишний токен, повтор условия или нарушение порядка частей.")
        return root


def parse_query(text):
    return Parser(text).parse()
