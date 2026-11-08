
from pathlib import Path
from urllib.request import urlretrieve

url = "https://raw.githubusercontent.com/unimorph/rus/master/rus"

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

file_path = data_dir / "rus.tsv"

if not file_path.exists():
    print("Скачиваем UniMorph...")
    urlretrieve(url, file_path)

print(f"Размер: {file_path.stat().st_size / 1024**2:.2f} МБ")
print("Готово!")
