"""Заморозка системы: выбранные веса и правило «опухоль есть» → models/final/.

Копируются best.pt классификатора и сегментатора (основной seed 42) и configs/presence.yaml; рядом пишется
manifest.json: откуда взяты файлы, их md5, коммит, дата. После заморозки модели и порог не меняются.
Повторная заморозка поверх существующей — только с --force.

Пример:
    python -m src.freeze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

FINAL = Path("models/final")


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def one(models_dir: Path, pattern: str) -> Path:
    runs = sorted(models_dir.glob(pattern))
    if len(runs) != 1:
        raise ValueError(f"шаблону {pattern!r} соответствует {len(runs)} папок, нужна одна")
    return runs[0] / "best.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classifier", default="run1_cls_s42_*")
    ap.add_argument("--segmenter", default="run1_seg_neg_s42_*")
    ap.add_argument("--presence", type=Path, default=Path("configs/presence.yaml"))
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--out", type=Path, default=FINAL)
    ap.add_argument("--force", action="store_true", help="перезаписать существующую заморозку")
    args = ap.parse_args()

    files = {"classifier.pt": one(args.models_dir, args.classifier), "segmenter.pt": one(args.models_dir, args.segmenter),
             "presence.yaml": args.presence}
    if (args.out / "manifest.json").exists() and not args.force:
        raise SystemExit(f"{args.out} уже заморожена (manifest.json есть); перезапись — только с --force")
    rule = yaml.safe_load(args.presence.read_text(encoding="utf-8"))
    if not (args.classifier == rule["classifier"] and args.segmenter == rule["segmenter"]):
        raise SystemExit(f"порог в {args.presence} подобран для {rule['classifier']} + {rule['segmenter']}, а замораживаются другие модели")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        shutil.copy2(src, args.out / name)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    manifest = {"frozen_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "git_head_at_freeze": commit,
                "files": {name: {"source": src.as_posix(), "md5": md5(args.out / name)} for name, src in files.items()},
                "cls_threshold": rule["cls_threshold"], "min_area_px": rule["min_area_px"]}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, v in manifest["files"].items():
        print(f"{name}: {v['source']} (md5 {v['md5'][:12]})")
    print(f"порог {rule['cls_threshold']}, площадь > {rule['min_area_px']} px; заморожено в {args.out}")


if __name__ == "__main__":
    main()
