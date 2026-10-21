"""Считывание отметок с листов проверки класса «нет опухоли» (папка doctorReview).

Сканы листов с отметками проверяющих в публичный репозиторий не входят; результат считывания —
reports/doctor/no_tumor_review/marks_raw.csv, сведенные ответы — answers.csv там же.

Размеченный лист сопоставляется с чистым листом reports/doctor/no_tumor_review/page_NN.png
(номер листа — по наибольшему сходству). Отметки ищутся по цвету: листы серые, пометки цветные.
Клетка считается отмеченной, если в окне вокруг нее достаточно цветных пикселей. Цветные пометки
поверх самого снимка считаются рисунком проверяющего (обведенный очаг, подпись). Текст подписей
программа не читает. Геометрия клеток — как в src/plot_no_tumor_review.py.

Результат (файлы проверяющих не меняются):
  reports/doctor/no_tumor_review/marks_raw.csv   — строка на снимок и на файл проверяющего;
  reports/doctor/no_tumor_review/marks_pages.csv — какой файл какому листу соответствует.

Пример:
    python -m src.read_doctor_marks --marks-dir doctorReview
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.io_utils import read_image
from src.plot_no_tumor_review import CELL_W, COLS, FINDING, HEAD, IMG_H, LAB_H, MODE

DPI = 120
SQ = 0.15  # сторона клетки, дюймы
# (варианты, отступ вниз от снимка, позиции клеток)
OPTION_ROWS = [(MODE[:3], 0.18, (0.12, 0.89, 1.66)), (MODE[3:], 0.44, (0.12, 0.89, 1.66)), (FINDING, 0.78, (0.12, 0.80, 1.63))]


def cell_origin(k: int) -> tuple[float, float]:
    row, col = divmod(k, COLS)
    return col * CELL_W, HEAD + row * (IMG_H + LAB_H)


def box_centers(k: int) -> dict[str, tuple[float, float]]:
    """Центры клеток (x, y в пикселях) для k-го снимка на листе."""
    left, top = cell_origin(k)
    return {o: ((left + dx + SQ / 2) * DPI, (top + IMG_H + dy) * DPI) for opts, dy, xs in OPTION_ROWS for o, dx in zip(opts, xs)}


def read_cell(color: np.ndarray, k: int, min_px: int) -> dict:
    """Отметки одного снимка: число цветных пикселей в окне каждой клетки и площадь рисунка на снимке.

    Окно клетки — 40×30 пикселей вокруг ее центра, смещенное влево (отметки ставят на клетку или
    левее, справа идет подпись варианта). Подписи проверяющих между рядами в окно не попадают."""
    left, top = cell_origin(k)
    x0, x1 = int(left * DPI), int((left + CELL_W) * DPI)
    y_img0, y_img1 = int(top * DPI), int((top + IMG_H) * DPI)
    px = {}
    for o, (cx, cy) in box_centers(k).items():
        wx0, wx1 = max(int(cx - 25), 0), int(cx + 15)
        px[o] = int(color[int(cy - 15) : int(cy + 15), wx0:wx1].sum())
    marked = {o: v for o, v in px.items() if v >= min_px}
    return {"mode": "|".join(o for o in MODE if o in marked), "finding": "|".join(o for o in FINDING if o in marked),
            "drawn_px": int(color[y_img0:y_img1, x0:x1].sum()), **{f"px_{o}": px[o] for o in MODE + FINDING}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--marks-dir", type=Path, default=Path("doctorReview"))
    ap.add_argument("--sheets-dir", type=Path, default=Path("reports/doctor/no_tumor_review"))
    ap.add_argument("--sat-thr", type=int, default=70, help="насыщенность цвета, с которой пиксель считается пометкой")
    ap.add_argument("--min-px", type=int, default=40, help="сколько цветных пикселей в окне клетки считается отметкой")
    args = ap.parse_args()

    answers = pd.read_csv(args.sheets_dir / "answers.csv")
    pages = {int(p.stem.split("_")[1]): read_image(p, gray=True) for p in sorted(args.sheets_dir.glob("page_*.png"))}
    shape = next(iter(pages.values())).shape
    small = {n: cv2.resize(g, (288, 282), interpolation=cv2.INTER_AREA).astype(np.float32) for n, g in pages.items()}

    page_rows, rows = [], []
    for f in sorted(args.marks_dir.iterdir()):
        if f.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        rgb = read_image(f, gray=False)
        if rgb.shape[:2] != shape:
            rgb = cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        s = cv2.resize(gray, (288, 282), interpolation=cv2.INTER_AREA).astype(np.float32)
        err = {n: float(np.abs(s - ref).mean()) for n, ref in small.items()}
        page = min(err, key=err.get)
        sat = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
        color = sat > args.sat_thr
        # пометки без цвета (серым или черным) цветовой поиск не увидит — считаем их отдельно
        gray_change = (cv2.absdiff(gray, pages[page]) > 90) & ~cv2.dilate(color.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        nums = answers[answers["page"] == page].sort_values("num")
        cells = []
        for k, a in enumerate(nums.itertuples(index=False)):
            cells.append({"file": f.name, "page": page, "num": a.num, "path": a.path, "split": a.split,
                          **read_cell(color, k, args.min_px)})
        rows += cells
        c = pd.DataFrame(cells)
        page_rows.append({
            "file": f.name, "page": page, "match_err": round(err[page], 2), "second_err": round(sorted(err.values())[1], 2),
            "resized": rgb.shape[:2] != read_image(f, gray=True).shape, "color_px": int(color.sum()), "gray_change_px": int(gray_change.sum()),
            "cells": len(c), "mode_one": int((c["mode"].str.count(r"\|") == 0).mul(c["mode"] != "").sum()),
            "mode_none": int((c["mode"] == "").sum()), "mode_multi": int(c["mode"].str.contains(r"\|").sum()),
            "finding_one": int(((c["finding"] != "") & ~c["finding"].str.contains(r"\|")).sum()),
            "finding_none": int((c["finding"] == "").sum()), "finding_multi": int(c["finding"].str.contains(r"\|").sum()),
            "with_drawing": int((c["drawn_px"] >= 40).sum()),
        })
    pd.DataFrame(page_rows).to_csv(args.sheets_dir / "marks_pages.csv", index=False, encoding="utf-8-sig")
    raw = pd.DataFrame(rows)
    raw.to_csv(args.sheets_dir / "marks_raw.csv", index=False, encoding="utf-8-sig")
    show = pd.DataFrame(page_rows).assign(file=lambda d: d["file"].str.slice(0, 14))
    print(show.drop(columns=["second_err"]).to_string(index=False))
    print(f"\nстрок: {len(raw)}; снимков с отметками хотя бы от одного проверяющего: "
          f"{raw[(raw['mode'] != '') | (raw['finding'] != '')]['num'].nunique()} из {answers['num'].nunique()}")


if __name__ == "__main__":
    main()
