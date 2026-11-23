"""Осторожное удаление отдельных вставок; новые слова не генерируются."""
import re
from preprocessing import has_predicate, tokenize_sentence

FILLER = re.compile(
    r'^(?:(?:Кроме того|Вместе с тем|При этом|В частности)\s*,\s*|'
    r'(?:Стоит отметить|Необходимо отметить|Следует отметить),?\s+что\s+)', re.I)
PARTICIPLE = re.compile(
    r',\s*(?:работающ\w*|занимающ\w*|расположенн\w*|находящ\w*|базирующ\w*)'
    r'[^,;.!?]{5,120},', re.I)
PROTECTED = re.compile(
    r'\d|\b(?:не|ни|без|якобы|предположительно|следств\w*|верси\w*|слов\w*)\b', re.I)


def balance_quotes(text):
    # Оборванную цитату оставляем обычным фрагментом, не придумывая её границы.
    if text.count('«') != text.count('»'):
        text = text.replace('«', '').replace('»', '')
    if text.count('"') % 2:
        text = text.replace('"', '')
    return text.strip()


def compressed_versions(sentence, enabled=True):
    original = balance_quotes(sentence)
    versions = [(original, 'исходное')]
    if not enabled:
        return versions
    shortened = FILLER.sub('', original)
    if shortened and shortened != original:
        shortened = shortened[0].upper() + shortened[1:]
    def remove_insert(match):
        return match.group(0) if PROTECTED.search(match.group(0)) else ' '
    shortened = PARTICIPLE.sub(remove_insert, shortened)
    shortened = re.sub(r'\s+', ' ', shortened).strip()
    if shortened != original and len(tokenize_sentence(shortened, False, False)) >= 4:
        versions.append((shortened, 'сокращено по правилам'))
    # До точки с запятой иногда находится самостоятельное предложение.
    if ';' in shortened:
        prefix, tail = shortened.split(';', 1)
        if (len(prefix) >= 50 and has_predicate(prefix)
                and not re.search(r'\b(?:но|однако|не|лишь|только)\b', tail, re.I)):
            prefix = prefix.rstrip(' ,:') + '.'
            versions.append((balance_quotes(prefix), 'самостоятельная часть до ;'))
    unique = []
    seen = set()
    for text, rule in versions:
        if text and text not in seen:
            unique.append((text, rule))
            seen.add(text)
    return unique
