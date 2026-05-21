from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from common import REPO_ROOT, slugify


KNOWN_URLS = {
    "Airfoil": "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
    "Boston housing": "http://lib.stat.cmu.edu/datasets/boston",
    "Wine red": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "Wine white": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
    "Yacht hydrodynamics": "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
    "Energy": "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
    "Concrete compressive strength": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
    "Dow chemical": "http://gpbenchmarks.org/wp-content/uploads/2020/07/SymbolicRegressionCompetitionData.xls",
}

GOOGLE_DRIVE_FILES = {
    "Tower": [
        ("towerData-train-0.csv", "1pl0YQt9sIKkmYG34Tobn5GpUkRv4GK9I"),
        ("towerData-test-0.csv", "13BzLFTEsICbfDbfteAMQaKJ7JENoqkOG"),
    ],
}
DOWNLOAD_ALIASES = {
    "Energy cooling": "Energy",
    "Energy heating": "Energy",
}


def download(name: str, url: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(url).suffix or ".data"
    out = raw_dir / f"{slugify(name)}{suffix}"
    if out.exists():
        return out
    print(f"downloading {name}: {url}")
    urllib.request.urlretrieve(url, out)
    return out


def download_google_drive_file(file_id: str, out: Path) -> None:
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as response, out.open("wb") as f:
        shutil.copyfileobj(response, f)


def download_tower(raw_dir: Path) -> Path:
    out = raw_dir / f"{slugify('Tower')}.csv"
    if out.exists():
        return out
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("downloading Tower from the paper dataset folder")
    with out.open("wb") as combined:
        for file_name, file_id in GOOGLE_DRIVE_FILES["Tower"]:
            part = raw_dir / file_name
            if not part.exists():
                download_google_drive_file(file_id, part)
            data = part.read_bytes()
            combined.write(data)
            if data and not data.endswith(b"\n"):
                combined.write(b"\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public datasets used by the paper reproduction harness.")
    parser.add_argument("--datasets", nargs="*", default=[*KNOWN_URLS, "Tower"])
    args = parser.parse_args()

    raw_dir = REPO_ROOT / "data" / "raw"
    for name in args.datasets:
        name = DOWNLOAD_ALIASES.get(name, name)
        if name in GOOGLE_DRIVE_FILES:
            download_tower(raw_dir)
            continue
        if name not in KNOWN_URLS:
            print(f"[skip] no public URL configured for {name}; place a CSV under data/raw/{slugify(name)}.csv")
            continue
        download(name, KNOWN_URLS[name], raw_dir)


if __name__ == "__main__":
    main()
