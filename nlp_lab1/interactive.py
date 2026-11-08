"""Ввод предложений через терминал."""
import re

from context_model import ContextModel
from dictionary import DATA_DIR, load_dictionary, normalize
from oov_model import OOVModel

PUNCTUATION = re.compile(r"[,.!?]")
EXIT_WORDS = {"выход", "exit", "quit"}


def lemmatize_sentence(sentence, dictionary, oov, context):
    """Возвращает результат в формате токен{лемма=POS}."""
    tokens = PUNCTUATION.sub(" ", sentence).split()
    if not tokens:
        return ""

    words = [normalize(token) for token in tokens]
    candidate_lists = []

    for token, word in zip(tokens, words):
        known = dictionary.get(word)
        if known:
            candidate_lists.append([
                (lemma, pos, "DICT", 0)
                for lemma, pos in sorted(known)
            ])
            continue

        predicted = oov.predict(word)
        if predicted:
            candidate_lists.append([
                (lemma, pos, "OOV", support)
                for lemma, pos, support in predicted
            ])
        else:
            pos = "PROPN" if token[:1].isupper() else "X"
            candidate_lists.append([(word, pos, "FALLBACK", 0)])

    chosen = context.decode(words, candidate_lists)
    return " ".join(
        f"{token}{{{lemma}={pos}}}"
        for token, (lemma, pos, _, _) in zip(tokens, chosen)
    )


def main():
    print("Загружаем словарь и модели (один раз при запуске)...")
    dictionary = load_dictionary()
    oov = OOVModel(DATA_DIR / "train.tsv")
    context = ContextModel(DATA_DIR)
    print(f"Готово: {len(dictionary):,} словоформ. Пиши предложения по одному в строке.")
    print("Для завершения введи 'выход' или нажми Ctrl+C.\n")

    while True:
        try:
            sentence = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nЗавершение.")
            break

        if sentence.lower() in EXIT_WORDS:
            print("Завершение.")
            break
        if not sentence:
            continue

        try:
            print(lemmatize_sentence(sentence, dictionary, oov, context))
        except Exception as exc:
            print(f"Ошибка обработки: {exc}")


if __name__ == "__main__":
    main()
