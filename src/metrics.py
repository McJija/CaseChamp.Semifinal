"""Метрики классификации и сегментации и правила выбора эпохи.

Правила выбора эпохи (записаны до обучения):
  классификатор — лучший macro-F1 среди эпох с recall «опухоль есть» не ниже порога (0,95);
                  если таких эпох нет — среди всех;
  сегментатор   — лучший Dice;
  многозадачная — лучший macro-F1 среди эпох, где Dice не хуже лучшего более чем на допуск (0,02).
При равенстве берется более ранняя эпоха.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.labels import CLASSES, NO_TUMOR_IDX


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% доверительный интервал Уилсона для доли k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def classification_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """labels — индексы классов, probs — вероятности B×4. «Опухоль есть» = предсказан не класс «нет опухоли»."""
    pred = probs.argmax(1)
    tumor = labels != NO_TUMOR_IDX
    out = {
        "acc": float((pred == labels).mean()),
        "macro_f1": float(f1_score(labels, pred, labels=list(range(len(CLASSES))), average="macro", zero_division=0)),
    }
    per = f1_score(labels, pred, labels=list(range(len(CLASSES))), average=None, zero_division=0)
    out.update({f"f1_{c}": float(v) for c, v in zip(CLASSES, per)})
    n_t, n_n = int(tumor.sum()), int((~tumor).sum())
    k_t, k_n = int((pred[tumor] != NO_TUMOR_IDX).sum()), int((pred[~tumor] == NO_TUMOR_IDX).sum())
    out["recall_tumor"] = k_t / n_t if n_t else float("nan")
    out["specificity"] = k_n / n_n if n_n else float("nan")
    out["spec_lo"], out["spec_hi"] = wilson(k_n, n_n)
    out["n_tumor"], out["n_no_tumor"] = n_t, n_n
    out["auc_tumor"] = float(roc_auc_score(tumor, 1 - probs[:, NO_TUMOR_IDX])) if n_t and n_n else float("nan")
    return out


def dice_iou(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """Dice и IoU двух бинарных масок одного снимка."""
    inter = float((pred & true).sum())
    s = float(pred.sum() + true.sum())
    dice = 2 * inter / s if s else 1.0
    union = s - inter
    return dice, (inter / union if union else 1.0)


def segmentation_metrics(frame: pd.DataFrame, min_area_px: int) -> dict:
    """frame — по строке на снимок: cls, has_mask, source, dice, iou, pred_area."""
    m = frame[frame["has_mask"]]
    out = {"dice": float(m["dice"].mean()) if len(m) else float("nan"),
           "iou": float(m["iou"].mean()) if len(m) else float("nan"), "n_masks": len(m)}
    for c in CLASSES:
        x = m[m["cls"] == c]
        if len(x):
            out[f"dice_{c}"] = float(x["dice"].mean())
    neg = frame[frame["cls"] == "no_tumor"]
    if len(neg):  # доля снимков без опухоли, на которых нарисована маска больше порога
        k = int((neg["pred_area"] > min_area_px).sum())
        out["false_mask_rate"] = k / len(neg)
        out["false_mask_lo"], out["false_mask_hi"] = wilson(k, len(neg))
        out["n_no_tumor_seg"] = len(neg)
    ext = frame[(frame["cls"] != "no_tumor") & ~frame["has_mask"]]
    for src, x in ext.groupby("source"):  # опухоли без масок: найдена ли область (внешняя проверка)
        out[f"found_rate_{src}"] = float((x["pred_area"] > min_area_px).mean())
        out[f"n_{src}_unmasked"] = len(x)
    return out


# ---------------------------------------------------------------- выбор эпохи


def _best(rows: list[dict], key: str) -> int:
    """Эпоха с наибольшим key; при равенстве — более ранняя."""
    return max(rows, key=lambda r: (r[key], -r["epoch"]))["epoch"]


def select_epoch(history: list[dict], kind: str, recall_floor: float = 0.95, dice_tolerance: float = 0.02) -> int:
    if kind == "segmenter":
        return _best(history, "dice")
    if kind == "classifier":
        ok = [r for r in history if r["recall_tumor"] >= recall_floor]
        return _best(ok or history, "macro_f1")
    best_dice = max(r["dice"] for r in history)
    ok = [r for r in history if r["dice"] >= best_dice - dice_tolerance]
    # при равном macro-F1 — эпоха с большим Dice, затем более ранняя (так выбор всегда среди хранимых весов)
    return max(ok, key=lambda r: (r["macro_f1"], r["dice"], -r["epoch"]))["epoch"]


def candidate_epochs(history: list[dict], kind: str, recall_floor: float = 0.95) -> set[int]:
    """Эпохи, одна из которых может быть выбрана по итогам обучения, — их веса нужно хранить."""
    if kind == "segmenter":
        return {_best(history, "dice")}
    if kind == "classifier":
        keep = {_best(history, "macro_f1")}
        ok = [r for r in history if r["recall_tumor"] >= recall_floor]
        if ok:
            keep.add(_best(ok, "macro_f1"))
        return keep
    # многозадачная: все эпохи, которые ни одна другая не превосходит сразу и по macro-F1, и по Dice
    keep = set()
    for r in history:
        dominated = any((o["macro_f1"] >= r["macro_f1"] and o["dice"] >= r["dice"])
                        and (o["macro_f1"] > r["macro_f1"] or o["dice"] > r["dice"] or o["epoch"] < r["epoch"])
                        for o in history if o is not r)
        if not dominated:
            keep.add(r["epoch"])
    return keep
