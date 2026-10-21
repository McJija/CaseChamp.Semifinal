"""Контрфактические примеры для обучения (эксперимент: ответ «опухоль есть» не зависел от опухоли).

Для снимка с маской опухоли при обучении случайно выбирается одно из трех:
  «убрать»   — опухоль закрашивается, метка становится «нет опухоли» (маска — пустой);
  «контроль» — закрашивается участок той же формы вне опухоли, метка не меняется;
  иначе снимок остается как есть.
У снимков «нет опухоли» с той же вероятностью, что и «контроль», закрашивается случайный овальный участок.
Так след закрашивания встречается при любой метке и сам по себе ничего не сообщает; отличать «убрать»
от остальных можно только по самой опухоли.

Способы закрашивания здесь — зеркальный перенос ткани с другой стороны головы и cv2.INPAINT_NS — намеренно
не совпадают со способом из теста с закрашиванием (cv2.INPAINT_TELEA, configs/eval_definitions.yaml).

Просмотр примеров (только обучающая часть):
    python -m src.counterfactual --preview
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from src.labels import NO_TUMOR_IDX
from src.stress import _dilate, control_region

CONTROL = {"candidates": ["mirror_horizontal", "mirror_vertical", "rotate_180"], "no_overlap_dilate_px": 8,
           "min_tissue_fraction": 0.8, "tissue_threshold": 20}
FEATHER_PX = 3


def mirror_fill(image: np.ndarray, region: np.ndarray, keep_out: np.ndarray) -> np.ndarray | None:
    """Заполняет region тканью с зеркальной стороны головы; None, если источник выходит за кадр, задевает keep_out или лежит не на ткани."""
    xs = np.nonzero((image > CONTROL["tissue_threshold"]).any(0))[0]
    if xs.size == 0:
        return None
    cx = (xs.min() + xs.max()) / 2
    big = _dilate(region, FEATHER_PX)
    ry, rx = np.nonzero(big)
    sx = np.round(2 * cx - rx).astype(int)
    if (sx < 0).any() or (sx >= image.shape[1]).any() or keep_out[ry, sx].any():
        return None
    if (image[ry, sx] > CONTROL["tissue_threshold"]).mean() < 0.9:  # с той стороны не ткань (например, сагиттальный срез)
        return None
    patch = image.astype(np.float32)
    patch[ry, rx] = image[ry, sx]
    alpha = cv2.GaussianBlur(big.astype(np.float32), (0, 0), FEATHER_PX / 2)
    return np.clip(np.round(alpha * patch + (1 - alpha) * image), 0, 255).astype(np.uint8)


def fill(image: np.ndarray, region: np.ndarray, keep_out: np.ndarray, rng: random.Random) -> tuple[np.ndarray, str]:
    """Закрашивает region одним из двух способов (зеркальный перенос — если возможен и выпал)."""
    if rng.random() < 0.5:
        out = mirror_fill(image, region, keep_out)
        if out is not None:
            return out, "зеркало"
    return cv2.inpaint(image, region.astype(np.uint8), rng.choice((3, 5, 7)), cv2.INPAINT_NS), "ns"


def random_blob(image: np.ndarray, rng: random.Random) -> np.ndarray | None:
    """Случайный овальный участок на ткани (для снимков без маски)."""
    tissue = image > CONTROL["tissue_threshold"]
    ys, xs = np.nonzero(tissue)
    if ys.size == 0:
        return None
    for _ in range(10):
        k = rng.randrange(ys.size)
        blob = np.zeros(image.shape, np.uint8)
        cv2.ellipse(blob, (int(xs[k]), int(ys[k])), (rng.randint(8, 30), rng.randint(8, 30)), rng.uniform(0, 180), 0, 360, 1, -1)
        blob = blob.astype(bool)
        if tissue[blob].mean() >= CONTROL["min_tissue_fraction"]:
            return blob
    return None


def apply(image: np.ndarray, mask: np.ndarray | None, label: int, cf: dict, rng: random.Random | None = None
          ) -> tuple[np.ndarray, np.ndarray | None, int, str]:
    """Возвращает (снимок, маска, метка, что сделано: «убрать» / «контроль» / «нет»)."""
    rng = rng or random
    u = rng.random()
    has_tumor_mask = mask is not None and bool(mask.any())
    if has_tumor_mask:
        d = rng.randint(*cf["dilate_px"])
        tumor = _dilate(mask > 0, d)
        if u < cf["p_remove"]:
            out, _ = fill(image, tumor, tumor, rng)
            return out, np.zeros_like(mask), NO_TUMOR_IDX, "убрать"
        if u < cf["p_remove"] + cf["p_control"]:
            region, _ = control_region(image, mask, {"dilate_px": d, "control": CONTROL})
            if region is not None:
                return fill(image, region, tumor, rng)[0], mask, label, "контроль"
    elif label == NO_TUMOR_IDX and u < cf["p_control"]:
        region = random_blob(image, rng)
        if region is not None:
            return fill(image, region, np.zeros(image.shape, bool), rng)[0], mask, label, "контроль"
    return image, mask, label, "нет"


def preview(out: Path, n: int = 8, seed: int = 42) -> None:
    """Лист примеров по обучающей части: исходный снимок, «убрать» и «контроль» обоими способами."""
    import pandas as pd

    from src.dataset import PreprocCache

    rng = random.Random(seed)
    pool = pd.read_csv("data/processed/pool.csv", keep_default_na=False)
    train = pool[(pool["split"] == "train") & (pool["has_mask"].astype(str).str.lower() == "true")]
    cache = PreprocCache("data/processed/cache", "pool", 256)
    rows, stats = [], {"зеркало": 0, "ns": 0}
    for r in train.groupby("cls").sample(n // 3 + 1, random_state=seed).head(n).itertuples():
        img, mask = cache.image(r.path), cache.mask(r.path)
        tumor = _dilate(mask > 0, 4)
        mirrored = mirror_fill(img, tumor, tumor)
        ns = cv2.inpaint(img, tumor.astype(np.uint8), 5, cv2.INPAINT_NS)
        region, _ = control_region(img, mask, {"dilate_px": 4, "control": CONTROL})
        ctl = fill(img, region, tumor, rng)[0] if region is not None else np.zeros_like(img)
        rows.append(np.hstack([img, mirrored if mirrored is not None else np.zeros_like(img), ns, ctl]))
    for r in train.sample(600, random_state=seed).itertuples():
        img, mask = cache.image(r.path), cache.mask(r.path)
        tumor = _dilate(mask > 0, 4)
        stats[fill(img, tumor, tumor, rng)[1]] += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"примеры: {out} (столбцы: исходный, зеркальный перенос, INPAINT_NS, контроль; черный — способ неприменим)")
    print(f"способ закрашивания опухоли на 600 снимках train: {stats}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("reports/stress/counterfactual_examples.png"))
    args = ap.parse_args()
    if args.preview:
        preview(args.out)
