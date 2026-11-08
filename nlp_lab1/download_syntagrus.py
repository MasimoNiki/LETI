
from pathlib import Path
from urllib.request import urlretrieve

base_url = (
    "https://raw.githubusercontent.com/"
    "UniversalDependencies/UD_Russian-SynTagRus/master/"
)

filenames = [
    "ru_syntagrus-ud-train-a.conllu",
    "ru_syntagrus-ud-train-b.conllu",
    "ru_syntagrus-ud-train-c.conllu",
]

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

for name in filenames:
    path = data_dir / name

    if not path.exists():
        print("Скачиваем:", name)
        urlretrieve(base_url + name, path)

    print(f"Готово: {name} ({path.stat().st_size / 1024**2:.1f} МБ)")
