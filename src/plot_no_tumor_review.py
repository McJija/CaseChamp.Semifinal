"""Листы для врача: проверка всех уникальных снимков класса «нет опухоли».

Под каждым снимком врач отмечает две клетки:
  режим   — T1+К / T1 / T2 / FLAIR / КТ / ?
  находки — Норма / Опухоль / Другое
Берутся все рабочие снимки класса (primary, train + val + test): разметка идет вслепую,
без ответов модели, и нужна для чистки данных во всех частях. Снимки одного пациента
(patient_group_id) идут подряд; нумерация сквозная и совпадает с таблицей ответов.

Результат: reports/doctor/no_tumor_review/page_NN.png и answers.csv.

Пример:
    python -m src.plot_no_tumor_review
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from src.io_utils import read_image

MODE = ["T1+К", "T1", "T2", "FLAIR", "КТ", "?"]
FINDING = ["Норма", "Опухоль", "Другое"]
INK, MUTED, BAND = "#0b0b0b", "#52514e", "#eceae5"
COLS, ROWS = 8, 5
CELL_W, IMG_H, LAB_H, HEAD = 2.4, 2.4, 1.02, 1.62


def build_table(processed: Path, old_sheet: Path) -> pd.DataFrame:
    s = pd.read_csv(processed / "splits.csv")
    d = s[s["primary"] & (s["cls"] == "no_tumor")].sort_values(["patient_group_id", "path"]).reset_index(drop=True)
    d.insert(0, "num", np.arange(1, len(d) + 1))
    d["page"] = (d["num"] - 1) // (COLS * ROWS) + 1
    d["sample48_num"] = ""
    if old_sheet.exists():  # номера на прежнем листе-выборке (48 снимков), чтобы перенести уже данные ответы
        old = pd.read_csv(old_sheet).set_index("path")["num"]
        d["sample48_num"] = d["path"].map(old).map(lambda v: "" if pd.isna(v) else str(int(v)))
    d["mri_type"], d["finding"], d["comment"] = "", "", ""
    return d[["num", "page", "path", "split", "patient_group_id", "sample48_num", "mri_type", "finding", "comment"]]


def plot_page(chunk: pd.DataFrame, page: int, pages: int, total: int, data_root: Path, out: Path) -> None:
    fig = plt.figure(figsize=(CELL_W * COLS, ROWS * (IMG_H + LAB_H) + HEAD + 0.1), dpi=120, facecolor="white")
    fw, fh = fig.get_figwidth(), fig.get_figheight()
    fig.text(0.006, 1 - 0.16 / fh,
             f"Класс «Опухоли нет»: проверка снимков · лист {page} из {pages} · №{chunk['num'].iloc[0]}–{chunk['num'].iloc[-1]} из {total}",
             fontsize=16, fontweight="bold", va="top", color=INK)
    fig.text(0.006, 1 - 0.56 / fh,
             "Под каждым снимком отметьте ДВЕ клетки: одну в белых строках (режим) и одну в серой строке (находки). "
             "Все, что хочется уточнить, пишите прямо на снимке или рядом.\n"
             "Режим:  T1+К — T1 с контрастом,  T1 — T1 без контраста,  T2,  FLAIR,  КТ — компьютерная томография (не МРТ),  "
             "? — определить нельзя.\n"
             "Находки:  Норма — патологии не видно,  Опухоль — на снимке опухоль,  Другое — иная патология (очаги, киста, кровоизлияние и т. п.).",
             fontsize=10.5, va="top", color=MUTED, linespacing=1.55)
    sq = 0.15
    for k, r in enumerate(chunk.itertuples(index=False)):
        row, col = divmod(k, COLS)
        left, top = col * CELL_W, HEAD + row * (IMG_H + LAB_H)  # дюймы от левого верхнего угла листа
        ax = fig.add_axes(((left + 0.05) / fw, 1 - (top + IMG_H) / fh, (CELL_W - 0.1) / fw, (IMG_H - 0.05) / fh))
        ax.imshow(read_image(data_root / r.path), cmap="gray", vmin=0, vmax=255)
        ax.set_axis_off()
        ax.text(0.03, 0.97, f"{r.num}", transform=ax.transAxes, fontsize=11, fontweight="bold", color="white", va="top",
                bbox={"facecolor": "black", "edgecolor": "none", "pad": 1.6, "alpha": 0.75})
        # серая полоса под строку «находки»
        fig.add_artist(Rectangle(((left + 0.05) / fw, 1 - (top + IMG_H + 0.93) / fh), (CELL_W - 0.1) / fw, 0.3 / fh,
                                 fc=BAND, ec="none", transform=fig.transFigure, zorder=0))
        # (варианты, отступ вниз от снимка, позиции клеток по горизонтали) — у «находок» слова длиннее
        rows_opts = [(MODE[:3], 0.18, (0.12, 0.89, 1.66)), (MODE[3:], 0.44, (0.12, 0.89, 1.66)),
                     (FINDING, 0.78, (0.12, 0.80, 1.63))]
        for opts, dy, xs in rows_opts:
            y = 1 - (top + IMG_H + dy) / fh
            for o, dx in zip(opts, xs):
                x = (left + dx) / fw
                fig.add_artist(Rectangle((x, y - sq / (2 * fh)), sq / fw, sq / fh, fc="white", ec=INK, lw=0.9,
                                         transform=fig.transFigure, zorder=2))
                fig.text(x + (sq + 0.05) / fw, y, o, fontsize=8.5, va="center", color=INK, zorder=3)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("mrt"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/doctor/no_tumor_review"))
    args = ap.parse_args()

    t = build_table(args.processed, args.out_dir.parent / "no_tumor_mri_type.csv")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old in args.out_dir.glob("page_*.png"):
        old.unlink()
    pages = int(t["page"].max())
    for page, chunk in t.groupby("page"):
        plot_page(chunk, page, pages, len(t), args.data_root, args.out_dir / f"page_{page:02d}.png")
    t.to_csv(args.out_dir / "answers.csv", index=False, encoding="utf-8-sig")
    print(f"снимков: {len(t)} ({t['split'].value_counts().to_dict()}); листов: {pages}; "
          f"из них были на листе-выборке: {int((t['sample48_num'] != '').sum())}")
    print(f"листы и таблица ответов: {args.out_dir}")


if __name__ == "__main__":
    main()
