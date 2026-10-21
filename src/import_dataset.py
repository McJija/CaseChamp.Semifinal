"""Подготовка датасета для обучения из архива brain_mri_dataset.zip (исходные снимки и обводки).

Читает dataset.csv, прогоняет каждый снимок через ту же подготовку, что при обучении (src/preprocess.py), и пишет
data/processed/pool.csv и кэш data/processed/cache/pool_256.*. После этого обучение запускается как обычно:
    python -m src.train --config configs/run1_cls.yaml
Оценка по этому датасету — с обеими папками снимков, указывающими на него:
    python -m src.evaluate --split test --data-root <папка> --fig-root <папка> --no-extra
Вместо этого можно скачать готовый кэш brain_mri_cache256.zip и распаковать его в корень проекта.

Пример:
    python -m src.import_dataset ../brain_mri_dataset
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.dataset import build_cache


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="распакованная папка brain_mri_dataset")
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--force", action="store_true", help="перезаписать существующий pool.csv")
    args = ap.parse_args()
    d = pd.read_csv(args.dataset / "dataset.csv", keep_default_na=False, dtype=str)
    # те же колонки, что у pool.csv из src/build_pool.py; root — figshare / case, как в исходном пуле
    pool = pd.DataFrame({
        "root": d["source"].map(lambda s: "figshare" if s == "figshare" else "case"), "path": d["image"], "mask": d["mask"],
        "has_mask": d["mask"] != "", "cls": d["class"], "split": d["split"], "source": d["source"],
        "fig_id": d["figshare_id"].replace("", "-1").astype(int), "pid_base": d["figshare_pid"], "group": d["patient_group"],
        "case_path": "", "nt_finding": d["nt_finding"], "nt_mode": d["nt_mode"]})
    out_csv = args.out / "pool.csv"
    if out_csv.exists() and not args.force:
        raise SystemExit(f"{out_csv} уже есть — добавьте --force, чтобы перезаписать")
    args.out.mkdir(parents=True, exist_ok=True)
    build_cache(pool, image_root=None, cache_dir=args.out / "cache", name="pool", size=args.size,
                roots={"figshare": args.dataset, "case": args.dataset})
    pool.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"{out_csv}: {len(pool)} снимков (train {int((pool.split == 'train').sum())}, val {int((pool.split == 'val').sum())}, "
          f"test {int((pool.split == 'test').sum())}); кэш — {args.out / 'cache'}")


if __name__ == "__main__":
    main()
