"""Внесение результатов из Kaggle в проект.

Источник — архив results_*.zip (его создает src.kaggle_run) или папка, в которую он распакован.
Папки запусков копируются в models/, строки experiments.csv — в reports/experiments.csv
(без повторов; путь к весам переписывается на локальный).

Пример:
    python -m src.import_kaggle C:/Users/.../results_20261008-101500.zip
    python -m src.import_kaggle kaggle/results
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd


def open_source(path: Path) -> tuple[list[str], callable]:
    """Список файлов источника (пути через «/») и функция чтения файла по такому пути."""
    if path.is_dir():
        return [f.relative_to(path).as_posix() for f in sorted(path.rglob("*")) if f.is_file()], lambda n: (path / n).read_bytes()
    z = zipfile.ZipFile(path)
    return [n for n in z.namelist() if not n.endswith("/")], z.read


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="архив results_*.zip или папка с его содержимым")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--experiments", type=Path, default=Path("reports/experiments.csv"))
    args = ap.parse_args()

    names, read = open_source(args.source)
    runs = sorted({n.split("/")[1] for n in names if n.startswith("models/") and n.count("/") >= 2})
    if not runs:
        raise SystemExit(f"в {args.source} нет папок запусков (models/<запуск>/…)")
    new = [r for r in runs if not (args.models_dir / r).exists()]
    for n in names:
        if n.startswith("models/") and n.split("/")[1] in new:
            target = args.models_dir / Path(*n.split("/")[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(read(n))
    exp = pd.read_csv(io.BytesIO(read("experiments.csv"))) if "experiments.csv" in names else pd.DataFrame()

    added = 0
    if len(exp):
        exp["run_dir"] = exp["weights"].map(lambda w: Path(str(w).replace("\\", "/")).parent.name)
        exp["weights"] = exp["run_dir"].map(lambda r: (args.models_dir / r / "best.pt").as_posix())
        old = pd.read_csv(args.experiments) if args.experiments.exists() else None
        known = set(old["run_dir"]) if old is not None and "run_dir" in old.columns else set()
        fresh = exp[~exp["run_dir"].isin(known)]
        added = len(fresh)
        args.experiments.parent.mkdir(parents=True, exist_ok=True)
        table = fresh if old is None else pd.concat([old, fresh], ignore_index=True)
        table.to_csv(args.experiments, index=False, encoding="utf-8")
    skipped = [r for r in runs if r not in new]
    no_weights = [r for r in new if not (args.models_dir / r / "best.pt").exists()]
    print(f"запусков в источнике: {len(runs)}; внесено новых: {len(new)}; уже были: {len(skipped)}; строк добавлено в {args.experiments}: {added}")
    if no_weights:
        print(f"без весов (архив собран с --no-weights): {no_weights}")


if __name__ == "__main__":
    main()
