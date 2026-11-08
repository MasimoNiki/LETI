
from pathlib import Path
from random import Random

DATA_DIR = Path("data")
SEED = 1629

TRAIN_FILES = [
    "ru_syntagrus-ud-train-a.conllu",
    "ru_syntagrus-ud-train-b.conllu",
    "ru_syntagrus-ud-train-c.conllu",
]


def normalize(word):
    return word.lower().replace("ё", "е")


def load_data():
    records = set()

    # UniMorph
    with open(DATA_DIR / "rus.tsv", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue

            lemma, form, tags = parts
            pos = tags.split(";")[0]

            pos = {
                "N": "NOUN",
                "V": "VERB",
                "V.PTCP": "VERB",
                "V.CVB": "VERB"
            }.get(pos, pos)

            records.add((normalize(lemma), normalize(form), pos))

    # SynTagRus
    for filename in TRAIN_FILES:
        with open(DATA_DIR / filename, encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("\t")

                if len(parts) != 10 or not parts[0].isdigit():
                    continue

                form, lemma, pos = parts[1:4]

                if pos == "PUNCT":
                    continue

                records.add((normalize(lemma), normalize(form), pos))

    # Исключаем пустые и неалфавитные записи
    records = {
        row for row in records
        if row[0].isalpha() and row[1].isalpha()
    }

    return records


def save_tsv(filename, records):
    with open(DATA_DIR / filename, "w", encoding="utf-8") as file:
        for row in sorted(records):
            file.write("\t".join(row) + "\n")


# Загружаем данные
records = load_data()

# Получаем список уникальных лемм
lemmas = sorted({lemma for lemma, form, pos in records})

# Перемешиваем с фиксированным seed
Random(SEED).shuffle(lemmas)

# 80% лемм отправляем в обучение
split = int(len(lemmas) * 0.8)
train_lemmas = set(lemmas[:split])

train = set()
valid = set()

for row in records:
    if row[0] in train_lemmas:
        train.add(row)
    else:
        valid.add(row)

# Оставляем в OOV-тесте только действительно новые словоформы
train_forms = {form for lemma, form, pos in train}
valid_oov = {row for row in valid if row[1] not in train_forms}

# Сохраняем результаты
save_tsv("train.tsv", train)
save_tsv("valid.tsv", valid)
save_tsv("valid_oov.tsv", valid_oov)

print("Всего лемм:", len(lemmas))
print("Всего записей:", len(records))
print("Обучающих записей:", len(train))
print("Проверочных записей:", len(valid))
print("OOV-записей:", len(valid_oov))
print("Уникальных OOV-слов:", len({row[1] for row in valid_oov}))
