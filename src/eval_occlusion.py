"""Тест с закрашиванием опухоли для классификаторов (определения — configs/eval_definitions.yaml).

На снимках с маской область опухоли закрашивается по окружающим тканям (в координатах модели, после
препроцессинга), снимок снова подается в модель. Контроль — закрасить участок той же формы вне опухоли.
Считается по снимкам, где исходный ответ — «опухоль»: доля сменивших ответ на «нет опухоли» и среднее
падение вероятности опухоли. Основная мера — разница долей на снимках, у которых есть контрольный участок:
чем она больше, тем сильнее ответ зависит именно от опухоли.

Результат: reports/stress/occlusion_<split>.csv (строка на модель), occlusion_<split>_images.csv
(строка на снимок и модель) и сводка в консоль. По умолчанию — val; test — только при финальной оценке.

Пример:
    python -m src.eval_occlusion --runs "run1_cls_s4*" "run1_cls_nodr_s4*"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.dataset import PreprocCache
from src.eval_stress import predict
from src.labels import CLASSES, NO_TUMOR_IDX
from src.models import build_model
from src.stress import DEFS, control_region, occlude, tumor_region


def measures(f: pd.DataFrame) -> dict:
    """Меры по таблице снимков одной модели (колонки tumor0, tumor_occ, tumor_ctl, p0, p_occ, p_ctl, has_control)."""
    a = f[f["tumor0"]]                     # исходный ответ — «опухоль»
    p = a[a["has_control"]]                # парная выборка: есть контрольный участок
    return {"n_answer_tumor": len(a), "n_paired": len(p),
            "flip_rate_all": float((~a["tumor_occ"]).mean()), "drop_p_all": float((a["p0"] - a["p_occ"]).mean()),
            "flip_rate": float((~p["tumor_occ"]).mean()), "flip_rate_control": float((~p["tumor_ctl"]).mean()),
            "flip_rate_minus_control": float((~p["tumor_occ"]).mean() - (~p["tumor_ctl"]).mean()),
            "drop_p": float((p["p0"] - p["p_occ"]).mean()), "drop_p_control": float((p["p0"] - p["p_ctl"]).mean())}


def bootstrap(frames: list[pd.DataFrame], key: str, n_boot: int, seed: int) -> tuple[float, float]:
    """95% интервал меры `key`, усредненной по моделям; единица повторной выборки — group."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(frames[0]["group"].to_numpy(), return_inverse=True)
    rows = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[g] for g in rng.integers(0, len(uniq), len(uniq))])
        vals.append(np.mean([measures(f.iloc[idx])[key] for f in frames]))
    return float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="шаблоны папок запусков внутри --models-dir")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/processed/cache"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/stress"))
    ap.add_argument("--limit", type=int, default=None, help="первые N снимков каждого класса (отладка)")
    args = ap.parse_args()

    defs_all = load_config(DEFS)
    defs, unc = defs_all["occlusion"], defs_all["uncertainty"]
    pool = pd.read_csv(args.processed / "pool.csv", keep_default_na=False)
    table = pool[(pool["split"] == args.split) & (pool["has_mask"].astype(str).str.lower() == "true")].reset_index(drop=True)
    if args.limit:
        table = table.groupby("cls").head(args.limit).reset_index(drop=True)
    cache = PreprocCache(args.cache_dir, "pool", 256)

    base, occ, ctl, how, area = [], [], [], [], []
    for r in table.itertuples():
        img, mask = cache.image(r.path), cache.mask(r.path)
        region = tumor_region(mask, defs)
        cand, name = control_region(img, mask, defs)
        base.append(img)
        occ.append(occlude(img, region, defs))
        ctl.append(occlude(img, cand, defs) if cand is not None else img)
        how.append(name)
        area.append(int(region.sum()))
    base, occ, ctl = np.stack(base), np.stack(occ), np.stack(ctl)
    table["control"], table["has_control"], table["region_px"] = how, [h != "нет" for h in how], area
    print(f"снимков с маской {len(table)} ({args.split}); контрольный участок найден у {int(table['has_control'].sum())}: "
          f"{table.groupby('cls')['has_control'].agg(lambda x: f'{int(x.sum())} из {len(x)}').to_dict()}", flush=True)

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
            p0, p1, p2 = predict(model, base), predict(model, occ), predict(model, ctl)
            f = table[["path", "cls", "group", "control", "has_control", "region_px"]].copy()
            for tag, p in (("0", p0), ("_occ", p1), ("_ctl", p2)):
                f["p" + tag] = 1 - p[:, NO_TUMOR_IDX]
                f["tumor" + tag] = p.argmax(1) != NO_TUMOR_IDX
                f["pred" + tag] = [CLASSES[k] for k in p.argmax(1)]
            frames.append(f)
            per_image.append(f.assign(run=run.name))
            m = measures(f)
            summary.append({"run": run.name, "config": ck["config"]["run"], "seed": ck["seed"], "commit": ck["commit"][:7], "split": args.split, **m})
            print(f"{run.name}: ответ «опухоль» у {m['n_answer_tumor']} из {len(f)}; после закрашивания опухоли ответ сменился у {m['flip_rate_all']:.3f} (все) "
                  f"и {m['flip_rate']:.3f} (с контролем, {m['n_paired']} шт.), при закрашенном контроле — у {m['flip_rate_control']:.3f}; "
                  f"падение p: {m['drop_p']:.3f} против {m['drop_p_control']:.3f}", flush=True)
        sets.append((pattern, frames))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).round(4).to_csv(args.out_dir / f"occlusion_{args.split}.csv", index=False, encoding="utf-8")
    pd.concat(per_image).round(5).to_csv(args.out_dir / f"occlusion_{args.split}_images.csv", index=False, encoding="utf-8")

    print(f"\nсреднее по seed и 95% интервал (бутстреп по {unc['bootstrap_unit']}, {unc['n_boot']} повторов):")
    for pattern, frames in sets:
        for key in ("flip_rate_minus_control", "flip_rate", "flip_rate_control", "flip_rate_all", "drop_p", "drop_p_control"):
            if frames:
                lo, hi = bootstrap(frames, key, unc["n_boot"], unc["seed"])
                print(f"  {pattern:22s} {key:24s} {np.mean([measures(f)[key] for f in frames]):.3f} [{lo:.3f}; {hi:.3f}]")
    print(f"\nтаблицы: {args.out_dir / f'occlusion_{args.split}.csv'}, {args.out_dir / f'occlusion_{args.split}_images.csv'}")


if __name__ == "__main__":
    main()
