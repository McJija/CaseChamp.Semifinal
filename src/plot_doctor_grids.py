"""Визуальные листы для врача.

1. classes_overview.png   — по 16 случайных рабочих снимков каждого класса (метки с правками).
2. augmentations.png      — аугментации организаторов: оригинал и 6 копий (BR, DA, HF, VF, RO, SP).
3. no_tumor_mri_type.png  — случайные снимки класса «нет опухоли» с местом для отметки режима МРТ
                            + no_tumor_mri_type.csv для внесения ответов.
Снимки берутся из train и val (test не используется), показаны как есть, без препроцессинга.

Пример:
    python -m src.plot_doctor_grids
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from src.io_utils import read_image

SEED = 42
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
CLS_RU = {"glioma": "Глиома", "meningioma": "Менингиома", "pituitary": "Опухоль гипофиза", "no_tumor": "Опухоли нет"}
AUGS = ["orig", "BR", "DA", "HF", "VF", "RO", "SP"]
AUG_RU = {"orig": "оригинал", "BR": "BR", "DA": "DA", "HF": "HF", "VF": "VF", "RO": "RO", "SP": "SP"}
MRI_OPTIONS = ["T1+К", "T1", "T2", "FLAIR", "КТ", "?"]
INK, MUTED = "#0b0b0b", "#52514e"


def _show(ax, img: np.ndarray) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_axis_off()


def plot_classes(work: pd.DataFrame, data_root: Path, out: Path, n: int = 16) -> None:
    cols, rows_per = 8, n // 8
    fig = plt.figure(figsize=(2.05 * cols, (2.2 * rows_per + 0.5) * len(CLASSES) + 1.1), dpi=100, facecolor="white")
    outer = fig.add_gridspec(len(CLASSES), 1, left=0.008, right=0.992, top=1 - 1.25 / fig.get_figheight(), bottom=0.005, hspace=0.16)
    fig.text(0.008, 1 - 0.18 / fig.get_figheight(),
             f"Обзор классов: по {n} случайных рабочих снимков (train и val, seed {SEED}), показаны как есть",
             fontsize=13, fontweight="bold", va="top", color=INK)
    for k, cls in enumerate(CLASSES):
        sub = work[work["cls"] == cls].sample(n, random_state=SEED)
        inner = outer[k].subgridspec(rows_per, cols, wspace=0.03, hspace=0.14)
        for i, r in enumerate(sub.itertuples(index=False)):
            ax = fig.add_subplot(inner[i // cols, i % cols])
            img = read_image(data_root / r.path)
            _show(ax, img)
            ax.set_title(f"{Path(r.path).name} · {img.shape[1]}×{img.shape[0]}", fontsize=6.5, pad=2, color=MUTED)
            if i == 0:
                ax.text(0, 1.2, f"{CLS_RU[cls]}  (рабочих снимков в train+val: {int((work['cls'] == cls).sum())})",
                        transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=INK, va="bottom")
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def plot_augmentations(s: pd.DataFrame, data_root: Path, out: Path, n_groups: int = 4) -> None:
    g = s[(s["source"] == "aug_GMPN") & (s["split"] == "train")]
    full = g.groupby("base_id")["aug"].nunique()
    full = full[full == 7].index
    first = g[g["base_id"].isin(full)].drop_duplicates("base_id").sort_values("base_id")
    picks = pd.concat([first[first["cls"] == c].sample(1, random_state=SEED) for c in CLASSES if (first["cls"] == c).any()])
    picks = picks.head(n_groups)
    fig, axes = plt.subplots(len(picks), len(AUGS), figsize=(2.25 * len(AUGS), 2.5 * len(picks) + 0.7), dpi=100, facecolor="white")
    for r, base in enumerate(picks["base_id"]):
        for c, aug in enumerate(AUGS):
            row = g[(g["base_id"] == base) & (g["aug"] == aug)].iloc[0]
            img = read_image(data_root / row["path"])
            _show(axes[r, c], img)
            axes[r, c].set_title(f"{AUG_RU[aug]} · {Path(row['path']).name}\n{img.shape[1]}×{img.shape[0]}", fontsize=7.5, pad=3)
    fig.suptitle("Аугментации организаторов: оригинал и 6 копий (суффикс в имени файла)", fontsize=12, fontweight="bold",
                 x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=0.3, h_pad=1.0)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def aug_numbers(s: pd.DataFrame, data_root: Path, n_groups: int = 60) -> pd.DataFrame:
    """Что делает каждая аугментация, в цифрах (по случайным полным группам)."""
    g = s[s["source"] == "aug_GMPN"]
    full = g.groupby("base_id")["aug"].nunique()
    bases = pd.Series(sorted(full[full == 7].index)).sample(n_groups, random_state=SEED)
    rows = []
    side = 224  # копии организаторов часто другого размера, чем оригинал — сравниваем в общем размере
    small = lambda im: cv2.resize(im, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
    corr = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    for base in bases:
        x = g[g["base_id"] == base].set_index("aug")["path"]
        o_raw = read_image(data_root / x["orig"])
        o = small(o_raw)
        for aug in AUGS[1:]:
            a_raw = read_image(data_root / x[aug])
            a = small(a_raw)
            rows.append({
                "aug": aug, "same_size": a_raw.shape == o_raw.shape, "is_224": a_raw.shape == (224, 224),
                "mean_ratio": a_raw.mean() / max(o_raw.mean(), 1e-6),
                "frac_0_255": float(((a_raw <= 2) | (a_raw >= 253)).mean() - ((o_raw <= 2) | (o_raw >= 253)).mean()),
                "eq_orig": corr(a, o), "eq_flipH": corr(a, o[:, ::-1]), "eq_flipV": corr(a, o[::-1, :]),
            })
    t = pd.DataFrame(rows).groupby("aug").agg(
        размер_как_у_оригинала=("same_size", "mean"), размер_224x224=("is_224", "mean"),
        яркость_к_оригиналу=("mean_ratio", "median"),
        прирост_доли_пикселей_0_и_255=("frac_0_255", "median"), corr_с_оригиналом=("eq_orig", "median"),
        corr_с_отражением_по_горизонтали=("eq_flipH", "median"), corr_с_отражением_по_вертикали=("eq_flipV", "median"))
    return t.reindex(AUGS[1:]).round(3)


def plot_mri_type(work: pd.DataFrame, data_root: Path, out: Path, n: int = 48) -> pd.DataFrame:
    pool = work[work["cls"] == "no_tumor"].sort_values("path")
    sub = pool.sample(min(n, len(pool)), random_state=SEED).reset_index(drop=True)
    cols = 8
    rows = int(np.ceil(len(sub) / cols))
    cw, img_h, lab_h, head = 2.3, 2.3, 0.78, 1.25
    fig = plt.figure(figsize=(cw * cols, rows * (img_h + lab_h) + head), dpi=110, facecolor="white")
    fw, fh = fig.get_figwidth(), fig.get_figheight()
    fig.text(0.008, 1 - 0.18 / fh, "Класс «Опухоли нет»: какой режим МРТ на снимке?", fontsize=15, fontweight="bold", va="top", color=INK)
    fig.text(0.008, 1 - 0.58 / fh,
             f"Обведите или отметьте один вариант под каждым снимком:  T1+К — T1 с контрастом,  T1 — T1 без контраста,  T2,  FLAIR,  "
             f"КТ — это не МРТ, а компьютерная томография,  ? — определить нельзя.\nСлучайные {len(sub)} из {len(pool)} уникальных снимков класса (train и val, seed {SEED}). "
             "Если на снимке видна патология — допишите рядом.",
             fontsize=10, va="top", color=MUTED, linespacing=1.5)
    for k, r in enumerate(sub.itertuples(index=False)):
        row, col = divmod(k, cols)
        top = 1 - (head + row * (img_h + lab_h)) / fh
        ax = fig.add_axes((col / cols + 0.04 / fw, top - (img_h - 0.04) / fh, (cw - 0.08) / fw, (img_h - 0.04) / fh))
        _show(ax, read_image(data_root / r.path))
        ax.text(0.03, 0.97, f"{k + 1}", transform=ax.transAxes, fontsize=10, fontweight="bold", color="white", va="top",
                bbox={"facecolor": "black", "edgecolor": "none", "pad": 1.5, "alpha": 0.7})
        # варианты ответа — две строки по три, в фиксированных колонках ячейки
        sq = 0.15
        for j, o in enumerate(MRI_OPTIONS):
            x = col / cols + (0.1 + (j % 3) * 0.74) / fw
            y = top - (img_h + 0.2 + (j // 3) * 0.27) / fh
            fig.add_artist(Rectangle((x, y - sq / (2 * fh)), sq / fw, sq / fh, fill=False, ec=INK, lw=0.9, transform=fig.transFigure))
            fig.text(x + (sq + 0.05) / fw, y, o, fontsize=8.5, va="center", color=INK)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    sub.insert(0, "num", np.arange(1, len(sub) + 1))
    sub["mri_type"] = ""
    sub["comment"] = ""
    return sub[["num", "path", "split", "source", "mri_type", "comment"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("mrt"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/doctor"))
    args = ap.parse_args()

    s = pd.read_csv(args.processed / "splits.csv")
    work = s[s["primary"] & s["split"].isin(["train", "val"])].sort_values("path").reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_classes(work, args.data_root, args.out_dir / "classes_overview.png")
    plot_augmentations(s, args.data_root, args.out_dir / "augmentations.png")
    sheet = plot_mri_type(work, args.data_root, args.out_dir / "no_tumor_mri_type.png")
    sheet.to_csv(args.out_dir / "no_tumor_mri_type.csv", index=False, encoding="utf-8-sig")
    t = aug_numbers(s, args.data_root)
    t.to_csv(args.out_dir / "augmentations_check.csv", encoding="utf-8-sig")
    print(t.to_string())
    print(f"листы: {args.out_dir}")


if __name__ == "__main__":
    main()
