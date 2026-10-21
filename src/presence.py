"""Решение «опухоль есть» по двум моделям и подбор порога на val.

Правило (решение команды 2026-10-14, записано до оценки на test): опухоль есть, если сегментатор
нашел область больше min_area пикселей ИЛИ вероятность опухоли у классификатора не ниже порога.
Тип опухоли — самый вероятный из трех у классификатора. Порог подбирается только на val: наибольшая
доля верных «нет опухоли» при recall «опухоль есть» не ниже recall_floor; при равенстве — больший
recall, затем меньший порог. Кандидаты — середины между соседними значениями вероятности.

Результат: configs/presence.yaml (его читает src/inference.py) и сводка в консоль.

Пример:
    python -m src.presence
    python -m src.presence --classifier "run1_cls_s42_*" --segmenter "run1_seg_neg_s42_*" --check "run1_seg_neg_s43_*" "run1_seg_neg_s44_*"
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

OUT = Path("configs/presence.yaml")


def decide(p_tumor, mask_area, threshold: float, min_area: int):
    """Опухоль есть: область больше min_area ИЛИ p_tumor ≥ threshold (работает и с числами, и с массивами)."""
    return (np.asarray(mask_area) > min_area) | (np.asarray(p_tumor) >= threshold)


def choose_threshold(p: np.ndarray, found: np.ndarray, tumor: np.ndarray, recall_floor: float) -> float:
    vals = np.unique(p[~found])
    cands = np.concatenate([[0.0], (vals[:-1] + vals[1:]) / 2, [1.0 + 1e-9]])
    best, best_key = None, None
    for t in cands:
        yes = found | (p >= t)
        recall, spec = yes[tumor].mean(), (~yes[~tumor]).mean()
        if recall < recall_floor:
            continue
        key = (spec, recall, -t)
        if best_key is None or key > best_key:
            best, best_key = float(t), key
    if best is None:
        raise ValueError(f"ни один порог не дает recall ≥ {recall_floor}")
    return best


def load(cls_pat: str, seg_pat: str, models_dir: Path, processed: Path) -> pd.DataFrame:
    def one(pat: str) -> Path:
        runs = sorted(models_dir.glob(pat))
        if len(runs) != 1:
            raise ValueError(f"шаблону {pat!r} соответствует {len(runs)} папок, нужна одна")
        return runs[0] / "val_predictions.csv"
    c = pd.read_csv(one(cls_pat))
    s = pd.read_csv(one(seg_pat))[["path", "pred_area"]]
    pool = pd.read_csv(processed / "pool.csv", keep_default_na=False)[["path", "nt_mode", "nt_finding"]]
    d = c.merge(s, on="path", validate="1:1").merge(pool, on="path", how="left")
    d["p_tumor"] = 1 - d["p_no_tumor"]
    return d


def summary(d: pd.DataFrame, threshold: float, min_area: int) -> dict:
    yes = decide(d["p_tumor"], d["pred_area"], threshold, min_area)
    tu, nt = (d["cls"] != "no_tumor").to_numpy(), (d["cls"] == "no_tumor").to_numpy()
    src, mode = d["source"].to_numpy(), d["nt_mode"].to_numpy()
    cls_yes = (d["p_tumor"] >= 0.5).to_numpy()
    found = (d["pred_area"] > min_area).to_numpy()
    out = {"n_tumor": int(tu.sum()), "n_no_tumor": int(nt.sum()),
           "recall": round(float(yes[tu].mean()), 4), "tumors_found": int(yes[tu].sum()),
           "specificity": round(float((~yes[nt]).mean()), 4), "no_tumor_correct": int((~yes[nt]).sum())}
    for s in ("figshare", "figshare_crop", "other"):
        m = tu & (src == s)
        out[f"found_{s}"] = f"{int(yes[m].sum())}/{int(m.sum())}"
    for md in sorted(set(mode[nt])):
        m = nt & (mode == md)
        out[f"no_tumor_correct_{md}"] = f"{int((~yes[m]).sum())}/{int(m.sum())}"
    out["classifier_alone"] = {"recall": round(float(cls_yes[tu].mean()), 4), "no_tumor_correct": int((~cls_yes[nt]).sum())}
    out["flag_models_disagree"] = {"no_tumor": int((cls_yes != found)[nt].sum()), "tumor": int((cls_yes != found)[tu].sum())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classifier", default="run1_cls_s42_*")
    ap.add_argument("--segmenter", default="run1_seg_neg_s42_*")
    ap.add_argument("--check", nargs="*", default=["run1_seg_neg_s43_*", "run1_seg_neg_s44_*"],
                    help="другие seed сегментатора: тот же порог, только для отчета об устойчивости")
    ap.add_argument("--recall-floor", type=float, default=0.985)
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    d = load(args.classifier, args.segmenter, args.models_dir, args.processed)
    tumor = (d["cls"] != "no_tumor").to_numpy()
    t = choose_threshold(d["p_tumor"].to_numpy(), (d["pred_area"] > args.min_area).to_numpy(), tumor, args.recall_floor)
    main_sum = summary(d, t, args.min_area)
    checks = {pat: summary(load(args.classifier, pat, args.models_dir, args.processed), t, args.min_area) for pat in args.check}
    rule = {"rule": "опухоль есть, если площадь маски сегментатора > min_area_px ИЛИ p_tumor классификатора ≥ cls_threshold",
            "cls_threshold": round(t, 6), "min_area_px": args.min_area, "recall_floor": args.recall_floor,
            "chosen_on": "val", "classifier": args.classifier, "segmenter": args.segmenter,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "val": main_sum, "val_other_segmenter_seeds": checks}
    args.out.write_text("# Порог решения «опухоль есть» по двум моделям (src/presence.py). Подобран только на val.\n"
                        + yaml.safe_dump(rule, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"порог классификатора {t:.6f} (min_area {args.min_area} px, recall ≥ {args.recall_floor}); записан в {args.out}")
    for name, s in [(args.segmenter, main_sum), *checks.items()]:
        print(f"  {name}: recall {s['recall']} ({s['tumors_found']}/{s['n_tumor']}), верно «нет опухоли» {s['no_tumor_correct']}/{s['n_no_tumor']}, "
              f"other {s['found_other']}, " + ", ".join(f"{k} {v}" for k, v in s.items() if k.startswith("no_tumor_correct_"))
              + f" | классификатор отдельно: {s['classifier_alone']} | модели расходятся: {s['flag_models_disagree']}")


if __name__ == "__main__":
    main()
