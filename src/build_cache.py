"""Сборка кэша препроцессинга для рабочих снимков (primary) из splits.csv.

Кэш `cls_<size>`: <cache_dir>/cls_<size>.npy (uint8, N×size×size), .csv (пути, split, класс,
геометрия препроцессинга), .json (версия). Его достаточно для обучения классификатора —
исходные снимки при обучении не нужны (в Kaggle можно загрузить только кэш и splits.csv).

Пример:
    python -m src.build_cache --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.dataset import DataConfig, build_cache


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()
    cfg = DataConfig.from_yaml(args.config, data_root=args.data_root, processed_dir=args.processed_dir, cache_dir=args.cache_dir)

    s = pd.read_csv(Path(cfg.processed_dir) / "splits.csv")
    table = s[s["primary"] & s["split"].isin(["train", "val", "test"])][
        ["path", "split", "cls", "patient_group_id", "group_exact_id", "source"]].sort_values("path")
    t = time.perf_counter()
    files = build_cache(table, cfg.data_root, cfg.cache_dir, "cls", cfg.size)
    size_mb = files["images"].stat().st_size / 2**20
    print(f"снимков: {len(table)} ({table['split'].value_counts().to_dict()}); {size_mb:.0f} МБ; "
          f"{time.perf_counter() - t:.0f} с\nкэш: {files['images']}")


if __name__ == "__main__":
    main()
