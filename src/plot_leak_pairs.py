"""Сетка случайных пар «test-снимок ↔ найденная копия в train» для ручной проверки утечки.

Выборка — по test-файлам (каждый test-файл не более одного раза), а не по парам:
иначе снимки с десятками копий перевешивали бы. Для выбранного test-файла
показывается одна копия из train — с максимальной корреляцией.
TEST и TRAIN показаны как есть; если копия найдена с поворотом/отражением,
третья панель показывает TRAIN, приведенный к ориентации TEST.

Пример:
    python -m src.plot_leak_pairs --level exact --n 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.imhash import TF_NAMES, TFS
from src.io_utils import read_image

SEED = 42
TF_INV = {"rot90": "rot270", "rot270": "rot90"}  # остальные преобразования обратны сами себе
TF_RU = {
    "id": "без преобразования", "rot90": "поворот 90°", "rot180": "поворот 180°", "rot270": "поворот 270°",
    "flipH": "отражение по горизонтали", "flipV": "отражение по вертикали",
    "transpose": "транспонирование", "antitranspose": "антитранспонирование",
}


def select_pairs(pairs: pd.DataFrame, level: str, n: int) -> pd.DataFrame:
    """n случайных test-файлов с копией в train; для каждого — лучшая (по corr) копия."""
    if level == "exact":
        p = pairs[pairs["is_exact"]]
    else:  # near, но не exact
        p = pairs[pairs["is_near"] & ~pairs["is_exact"]]
    p = p[p["split_i"] != p["split_j"]]
    i_test = (p["split_i"] == "test").to_numpy()
    out = pd.DataFrame({
        "test_path": np.where(i_test, p["path_i"], p["path_j"]),
        "test_cls": np.where(i_test, p["cls_i"], p["cls_j"]),
        "train_path": np.where(i_test, p["path_j"], p["path_i"]),
        "train_cls": np.where(i_test, p["cls_j"], p["cls_i"]),
        "phash_dist": p["phash_dist"].to_numpy(),
        "corr": p["corr"].to_numpy(),
        # В dup_pairs tf приводит снимок j к i. Если test — это j, для train нужно обратное преобразование.
        "tf": [t if it else TF_INV.get(t, t) for t, it in zip(p["tf"], i_test)],
    })
    n_copies = out.groupby("test_path").size().rename("train_copies")
    best = out.sort_values("corr", ascending=False).drop_duplicates("test_path").join(n_copies, on="test_path")
    best = best.sort_values("test_path").reset_index(drop=True)  # фиксируем порядок до выборки
    sel = best.sample(min(n, len(best)), random_state=SEED)
    return sel.sort_values(["test_cls", "test_path"]).reset_index(drop=True), len(best)


def plot_pages(sel: pd.DataFrame, data_root: Path, out_prefix: Path, title: str, per_page: int = 10) -> list[Path]:
    files = []
    pages = int(np.ceil(len(sel) / per_page))
    for pg in range(pages):
        chunk = sel.iloc[pg * per_page : (pg + 1) * per_page]
        rows = int(np.ceil(len(chunk) / 2))
        # Фиксированная сетка: [test, train, train→test, зазор, test, train, train→test];
        # tight_layout не используем — он сжимает оси под длинные подписи.
        fig = plt.figure(figsize=(17.5, 3.5 * rows + 0.5), dpi=100)
        top = 1 - 0.45 / fig.get_figheight()
        gs = fig.add_gridspec(rows, 7, width_ratios=[1, 1, 1, 0.12, 1, 1, 1], left=0.008, right=0.992,
                              top=top - 0.55 / fig.get_figheight(), bottom=0.005, wspace=0.03, hspace=0.26)
        for k, r in enumerate(chunk.itertuples(index=False)):
            row, col = divmod(k, 2)
            num = pg * per_page + k + 1
            test_img = read_image(data_root / r.test_path)
            train_img = read_image(data_root / r.train_path)
            panels = [
                (test_img, f"TEST · {r.test_cls} · {Path(r.test_path).name} · {test_img.shape[1]}×{test_img.shape[0]}"),
                (train_img, f"TRAIN · {r.train_cls} · {Path(r.train_path).name} · {train_img.shape[1]}×{train_img.shape[0]}"),
            ]
            if r.tf != "id":  # третья панель — train в ориентации test, чтобы не поворачивать в уме
                panels.append((TFS[TF_NAMES.index(r.tf)](train_img), "TRAIN, приведен к ориентации TEST"))
            pair_axes = []
            for c, (img, label) in enumerate(panels):
                ax = fig.add_subplot(gs[row, col * 4 + c])
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                ax.set_anchor("N")
                ax.set_axis_off()
                ax.set_title(label, fontsize=7.5, pad=3)
                pair_axes.append(ax)
            pair_axes[0].text(
                0.0, 1.085,
                f"№{num}  pHash={r.phash_dist}, corr={r.corr:.3f}, {TF_RU[r.tf]}; копий в train: {r.train_copies}",
                transform=pair_axes[0].transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom")
        fig.text(0.01, 1 - 0.2 / fig.get_figheight(), f"{title} — стр. {pg + 1}/{pages}", fontsize=11, ha="left", va="top")
        path = out_prefix.with_name(f"{out_prefix.name}_p{pg + 1}.png")
        fig.savefig(path)
        plt.close(fig)
        files.append(path)
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("mrt"))
    ap.add_argument("--pairs", type=Path, default=Path("data/processed/dup_pairs.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/leakage"))
    ap.add_argument("--level", choices=["exact", "near"], default="exact",
                    help="exact — тот же снимок; near — связи near, не вошедшие в exact")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs)
    sel, total = select_pairs(pairs, args.level, args.n)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_dir / f"leak_check_{args.level}"
    for old in args.out_dir.glob(f"{prefix.name}_p*.png"):
        old.unlink()
    title = (f"Проверка утечки, уровень {args.level}: {len(sel)} случайных test-файлов из {total} "
             f"с копией в train (seed {SEED})")
    files = plot_pages(sel, args.data_root, prefix, title)
    sel.insert(0, "num", np.arange(1, len(sel) + 1))
    sel.to_csv(prefix.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print(f"test-файлов с копией ({args.level}): {total}; показано: {len(sel)}")
    print("по классам:", sel["test_cls"].value_counts().to_dict())
    print("файлы:", ", ".join(str(f) for f in files), "+", prefix.with_suffix(".csv"))


if __name__ == "__main__":
    main()
