"""Сборка архивов для загрузки в Kaggle.

  kaggle/code_bundle.zip — код ровно из последнего коммита (src/, configs/, requirements-kaggle.txt)
                           и файл COMMIT с его хешем. Собирать заново после каждого коммита.
  kaggle/data_bundle.zip — pool.csv и кэш препроцессинга pool_256.* в одной папке (ключ --data),
                           только части train и val. Собирать заново только после изменения пула.
Исходные снимки и тестовая часть в архивы не входят.

Пример:
    python -m src.make_kaggle_bundle            # только код
    python -m src.make_kaggle_bundle --data     # код и данные
"""
from __future__ import annotations

import argparse
import compileall
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

CODE_PATHS = ["src", "configs", "requirements-kaggle.txt"]
SPLITS = ["train", "val"]  # части пула, которые уходят в Kaggle
REPO = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except FileNotFoundError:
        raise SystemExit("git не найден: запускайте из Git Bash (в PowerShell git не в PATH)") from None


def build_code(out: Path) -> str:
    dirty = git("status", "--porcelain", "--", *CODE_PATHS)
    if dirty:
        raise SystemExit("в коде есть незакоммиченные правки — сначала сделайте коммит:\n" + dirty)
    if not compileall.compile_dir(str(REPO / "src"), quiet=1, force=True):
        raise SystemExit("в src/ есть файлы с синтаксическими ошибками — архив не собран")
    commit = git("rev-parse", "HEAD")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "archive", "--format=zip", "-o", str(out.resolve()), "HEAD", *CODE_PATHS], cwd=REPO, check=True)
    with zipfile.ZipFile(out, "a", zipfile.ZIP_DEFLATED) as z:
        z.writestr("COMMIT", commit + "\n")
    return commit


def build_data(out: Path, processed: Path, cache_dir: Path, size: int) -> dict:
    files = {"pool.csv": processed / "pool.csv", **{f"pool_{size}{s}": cache_dir / f"pool_{size}{s}" for s in (".npy", "_mask.npy", ".csv", ".json")}}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise SystemExit(f"нет файлов: {missing} — сначала python -m src.build_pool")
    pool = pd.read_csv(files["pool.csv"], keep_default_na=False, dtype=str)
    cache = pd.read_csv(files[f"pool_{size}.csv"], keep_default_na=False, dtype=str)
    if not pool["path"].isin(cache["path"]).all() or len(pool) != len(cache):
        raise SystemExit("кэш не соответствует pool.csv — пересоберите: python -m src.build_pool")
    # тестовая часть в Kaggle не отправляется: там только обучение и выбор по val (правило 5)
    keep = np.flatnonzero(cache["split"].isin(SPLITS).to_numpy())
    cache_info = {**json.loads(files[f"pool_{size}.json"].read_text(encoding="utf-8")), "n": len(keep)}
    info = {"created": datetime.now().strftime("%Y-%m-%d %H:%M"), "pool_rows": len(keep),
            "pool_md5": hashlib.md5(files["pool.csv"].read_bytes()).hexdigest(),
            "split": cache.iloc[keep]["split"].value_counts().to_dict(), "test_included": False, "cache": cache_info}
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        z.writestr("pool.csv", pool[pool["split"].isin(SPLITS)].to_csv(index=False, lineterminator="\n"))
        z.writestr(f"pool_{size}.csv", cache.iloc[keep].to_csv(index=False, lineterminator="\n"))
        z.writestr(f"pool_{size}.json", json.dumps(cache_info))
        for suffix in (".npy", "_mask.npy"):
            arr = np.load(files[f"pool_{size}{suffix}"], mmap_mode="r")[keep]
            with z.open(f"pool_{size}{suffix}", "w", force_zip64=True) as f:
                np.save(f, arr)
            del arr
        z.writestr("data_info.json", json.dumps(info, ensure_ascii=False, indent=1))
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="store_true", help="собрать и архив данных")
    ap.add_argument("--out-dir", type=Path, default=Path("kaggle"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/processed/cache"))
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    code = args.out_dir / "code_bundle.zip"
    commit = build_code(code)
    with zipfile.ZipFile(code) as z:
        n = sum(not name.endswith("/") for name in z.namelist())
    print(f"код: {code} — {n} файлов, {code.stat().st_size / 1024:.0f} КБ, коммит {commit}")
    if args.data:
        data = args.out_dir / "data_bundle.zip"
        info = build_data(data, args.processed, args.cache_dir, args.size)
        print(f"данные: {data} — {data.stat().st_size / 2**20:.0f} МБ, снимков {info['pool_rows']}, части {info['split']} (test не входит), "
              f"md5 исходного pool.csv {info['pool_md5'][:12]}")


if __name__ == "__main__":
    main()
