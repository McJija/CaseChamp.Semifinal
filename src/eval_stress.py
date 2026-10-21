"""Стресс-тест shortcut для классификаторов (определения — configs/eval_definitions.yaml).

Снимки опухолей приводятся к техническому профилю файлов «нет опухоли» (маленький резкий сжатый файл),
снимки «нет опухоли» — к профилю файлов опухолей (большой мягкий файл); затем обычный препроцессинг
и модель. Считается, у какой доли снимков ответ «опухоль есть / нет» сменился. Решение — по argmax.

Результат: reports/stress/stress_<split>.csv (строка на модель), stress_<split>_images.csv (строка на
снимок и модель) и сводка в консоль. По умолчанию — val; test — только при финальной оценке.

Пример:
    python -m src.eval_stress --runs "run1_cls_s*" "run1_cls_nodr_s*"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.dataset import PreprocCache, to_tensor
from src.io_utils import read_image
from src.labels import CLASSES, NO_TUMOR_IDX
from src.models import build_model
from src.preprocess import preprocess
from src.stress import DEFS, to_no_tumor_profile, to_tumor_profile


@torch.no_grad()
def predict(model, images: np.ndarray, batch: int = 16) -> np.ndarray:
    """Вероятности классов для массива снимков uint8 N×S×S."""
    out = []
    for i in range(0, len(images), batch):
        x = torch.stack([to_tensor(im) for im in images[i:i + batch]])
        out.append(torch.softmax(model(x).float(), 1).numpy())
    return np.concatenate(out)


def transformed(table: pd.DataFrame, roots: dict, cache: PreprocCache, defs: dict, size: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Исходные (из кэша) и преобразованные снимки в координатах модели; число снимков, где кэш не совпал с пересчетом."""
    base, moved, stale = [], [], 0
    for r in table.itertuples():
        gray = read_image(Path(roots[r.root]) / r.path)
        base.append(cache.image(r.path))
        stale += int(not np.array_equal(preprocess(gray, size)[0], base[-1]))
        other = to_tumor_profile(gray, defs["no_tumor_to_tumor_profile"]) if r.cls == "no_tumor" else to_no_tumor_profile(gray, defs["tumor_to_no_tumor_profile"])
        moved.append(preprocess(other, size)[0])
    return np.stack(base), np.stack(moved), stale


def measures(f: pd.DataFrame) -> dict:
    """Меры стресс-теста по таблице снимков одной модели (колонки cls, tumor0, tumor1, p0, p1)."""
    t, n = f[f["cls"] != "no_tumor"], f[f["cls"] == "no_tumor"]
    ft = float((t["tumor0"] & ~t["tumor1"]).mean())   # опухоль: был ответ «опухоль», стал «нет опухоли»
    fn = float((~n["tumor0"] & n["tumor1"]).mean())   # нет опухоли: был ответ «нет опухоли», стал «опухоль»
    return {"flip_rate_tumor": ft, "flip_rate_no_tumor": fn, "balanced_flip_rate": (ft + fn) / 2,
            "delta_p_tumor": float((t["p1"] - t["p0"]).mean()), "delta_p_no_tumor": float((n["p1"] - n["p0"]).mean()),
            "recall_before": float(t["tumor0"].mean()), "recall_after": float(t["tumor1"].mean()),
            "specificity_before": float((~n["tumor0"]).mean()), "specificity_after": float((~n["tumor1"]).mean())}


def bootstrap(frames: list[pd.DataFrame], key: str, n_boot: int, seed: int, other: list[pd.DataFrame] | None = None) -> tuple[float, float]:
    """95% интервал меры `key`, усредненной по моделям `frames` (или разницы со средним по `other`); единица — group."""
    rng = np.random.default_rng(seed)
    groups = frames[0]["group"].to_numpy()
    uniq, inv = np.unique(groups, return_inverse=True)
    rows = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[g] for g in rng.integers(0, len(uniq), len(uniq))])
        v = np.mean([measures(f.iloc[idx])[key] for f in frames])
        if other:
            v -= np.mean([measures(f.iloc[idx])[key] for f in other])
        vals.append(v)
    return float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="шаблоны папок запусков внутри --models-dir; первый шаблон сравнивается со вторым")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/processed/cache"))
    ap.add_argument("--data-root", type=Path, default=Path("mrt"))
    ap.add_argument("--fig-root", type=Path, default=Path("data/external/figshare/png"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/stress"))
    ap.add_argument("--limit", type=int, default=None, help="первые N снимков каждого класса (отладка)")
    args = ap.parse_args()

    defs_all = load_config(DEFS)
    defs, unc = defs_all["stress_test"], defs_all["uncertainty"]
    pool = pd.read_csv(args.processed / "pool.csv", keep_default_na=False)
    table = pool[pool["split"] == args.split].reset_index(drop=True)
    if args.limit:
        table = table.groupby("cls").head(args.limit).reset_index(drop=True)
    size = 256
    cache = PreprocCache(args.cache_dir, "pool", size)
    base, moved, stale = transformed(table, {"figshare": args.fig_root, "case": args.data_root}, cache, defs, size)
    print(f"снимков {len(table)} ({args.split}): опухолей {int((table['cls'] != 'no_tumor').sum())}, «нет опухоли» {int((table['cls'] == 'no_tumor').sum())}; "
          f"кэш не совпал с пересчетом препроцессинга у {stale} снимков", flush=True)

    torch.set_num_threads(4)
    sets, summary, per_image = [], [], []
    for pattern in args.runs:
        frames = []
        for run in sorted(args.models_dir.glob(pattern)):
            ck = torch.load(run / "best.pt", weights_only=False, map_location="cpu")
            if ck["kind"] != "classifier":
                continue
            model = build_model(ck["kind"], ck["encoder"], pretrained=False)
            model.load_state_dict(ck["model"])
            model.eval()
            p0, p1 = predict(model, base), predict(model, moved)
            f = table[["path", "cls", "source", "group", "nt_mode", "nt_finding"]].copy()
            f["p0"], f["p1"] = 1 - p0[:, NO_TUMOR_IDX], 1 - p1[:, NO_TUMOR_IDX]
            f["tumor0"], f["tumor1"] = p0.argmax(1) != NO_TUMOR_IDX, p1.argmax(1) != NO_TUMOR_IDX
            f["pred0"], f["pred1"] = [CLASSES[k] for k in p0.argmax(1)], [CLASSES[k] for k in p1.argmax(1)]
            frames.append(f)
            per_image.append(f.assign(run=run.name))
            summary.append({"run": run.name, "config": ck["config"]["run"], "seed": ck["seed"], "commit": ck["commit"][:7], "split": args.split, **measures(f)})
            m = summary[-1]
            print(f"{run.name}: опухоли → «нет опухоли» у {m['flip_rate_tumor']:.3f}, «нет опухоли» → опухоль у {m['flip_rate_no_tumor']:.3f}, "
                  f"среднее двух {m['balanced_flip_rate']:.3f}; recall {m['recall_before']:.3f} → {m['recall_after']:.3f}, "
                  f"specificity {m['specificity_before']:.3f} → {m['specificity_after']:.3f}", flush=True)
        sets.append((pattern, frames))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).round(4).to_csv(args.out_dir / f"stress_{args.split}.csv", index=False, encoding="utf-8")
    pd.concat(per_image).round(5).to_csv(args.out_dir / f"stress_{args.split}_images.csv", index=False, encoding="utf-8")

    print(f"\nсреднее по seed и 95% интервал (бутстреп по {unc['bootstrap_unit']}, {unc['n_boot']} повторов):")
    for pattern, frames in sets:
        if not frames:
            continue
        for key in ("balanced_flip_rate", "flip_rate_tumor", "flip_rate_no_tumor"):
            mean = np.mean([measures(f)[key] for f in frames])
            lo, hi = bootstrap(frames, key, unc["n_boot"], unc["seed"])
            print(f"  {pattern:22s} {key:20s} {mean:.3f} [{lo:.3f}; {hi:.3f}]")
    if len(sets) >= 2 and sets[0][1] and sets[1][1]:
        (pa, fa), (pb, fb) = sets[0], sets[1]
        for key in ("balanced_flip_rate", "flip_rate_tumor", "flip_rate_no_tumor"):
            diff = np.mean([measures(f)[key] for f in fa]) - np.mean([measures(f)[key] for f in fb])
            lo, hi = bootstrap(fa, key, unc["n_boot"], unc["seed"], other=fb)
            print(f"  разница {pa} − {pb}: {key:20s} {diff:+.3f} [{lo:+.3f}; {hi:+.3f}]")
    print(f"\nтаблицы: {args.out_dir / f'stress_{args.split}.csv'}, {args.out_dir / f'stress_{args.split}_images.csv'}")


if __name__ == "__main__":
    main()
