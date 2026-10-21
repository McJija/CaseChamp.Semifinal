"""Лист для врача: снимки, которые лежат в датасете сразу в двух классах.

Каждая конфликтная группа (group_id с несколькими классами) — это один и тот же
снимок под разными именами и метками. На листе один случай = одна группа:
слева файл с одной меткой, справа — с другой (как есть в датасете), под ними
поле для ответа врача и список всех файлов случая. Все на одном PNG.

Дополнительно пишется data/processed/label_conflict_cases.csv — по строке на
файл, с пустой колонкой doctor_label: файлы с пустым ответом исключаются из
обучения и оценки.

Пример:
    python -m src.plot_label_conflicts
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from src.io_utils import read_image

CLS_RU = {"glioma": "глиома", "meningioma": "менингиома", "pituitary": "опухоль гипофиза", "no_tumor": "опухоли нет"}
# Буква для ответа. «Г» не используем: глиома и гипофиз начинаются одинаково.
CLS_LETTER = {"glioma": "Гл", "meningioma": "М", "pituitary": "Гп", "no_tumor": "Н"}
TF_RU = {"id": "", "flipH": "снимки зеркальны", "flipV": "снимки перевернуты",
         "rot180": "повернуты на 180°", "rot90": "повернуты на 90°", "rot270": "повернуты на 90°"}
INK, MUTED = "#0b0b0b", "#52514e"


def build_cases(index: pd.DataFrame, groups: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    d = index.merge(groups[["path", "group_id", "h", "w"]], on="path")
    n_cls = d.groupby("group_id")["cls"].nunique()
    c = d[d["group_id"].isin(n_cls.index[n_cls > 1])].copy()
    order = {g: k + 1 for k, g in enumerate(sorted(c["group_id"].unique()))}
    c["case"] = c["group_id"].map(order)
    # пара для показа: оригинал ↔ оригинал из разных классов с максимальной корреляцией
    orig = set(c.loc[c["is_original"], "path"])
    p = pairs[pairs["path_i"].isin(orig) & pairs["path_j"].isin(orig)].sort_values("corr", ascending=False)
    case_of = c.set_index("path")["case"]
    p = p.assign(case=p["path_i"].map(case_of)).drop_duplicates("case")
    shown = {r.case: (r.path_i, r.path_j, r.tf) for r in p.itertuples()}
    c["shown"] = [path in shown[case][:2] for path, case in zip(c["path"], c["case"])]
    c["tf"] = c["case"].map(lambda k: shown[k][2])
    c["doctor_label"] = ""
    return c.sort_values(["case", "cls", "path"]).reset_index(drop=True)


def plot_sheet(cases: pd.DataFrame, data_root: Path, out: Path) -> None:
    n = cases["case"].nunique()
    ncols, nrows = 2, (n + 1) // 2
    cell_w, cell_h, head = 8.6, 6.9, 1.15
    fig = plt.figure(figsize=(cell_w * ncols, cell_h * nrows + head), dpi=120, facecolor="white")
    H = fig.get_figheight()
    fig.text(0.012, 1 - 0.22 / H, "Проверка меток: один и тот же снимок лежит в датасете в двух классах",
             fontsize=17, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 1 - 0.62 / H,
             "В каждом случае слева и справа — ОДИН снимок (иногда зеркально отраженный), но с разными метками. "
             "Отметьте в рамке верный вариант или впишите свой.\n"
             "Обозначения:  М — менингиома,   Гп — опухоль гипофиза,   Гл — глиома,   "
             "? — по одному срезу определить нельзя.",
             fontsize=11, color=MUTED, va="top", linespacing=1.5)

    for case, x in cases.groupby("case"):
        k = case - 1
        row, col = divmod(k, ncols)
        x0 = col / ncols
        y_top = 1 - (head + row * cell_h) / H
        to_fig = lambda dx, dy: (x0 + dx / (cell_w * ncols), y_top - dy / H)  # дюймы от левого верхнего угла ячейки

        classes = sorted(x["cls"].unique())
        fx, fy = to_fig(0.12, 0.12)
        note = TF_RU.get(x["tf"].iloc[0], "")
        fig.text(fx, fy, f"Случай {case}", fontsize=15, fontweight="bold", color=INK, va="top")
        fig.text(to_fig(1.45, 0.17)[0], fy - 0.05 / H,
                 f"в датасете: {' и '.join(CLS_RU[c] for c in classes)}"
                 + (f"   ·   {note}" if note else ""),
                 fontsize=10, color=MUTED, va="top")

        img_w, img_h, gap = 4.05, 4.05, 0.16
        shown = x[x["shown"]].sort_values("cls")
        for c_i, r in enumerate(shown.itertuples()):
            left, top = to_fig(0.12 + c_i * (img_w + gap), 0.78)
            ax = fig.add_axes((left, top - img_h / H, img_w / (cell_w * ncols), img_h / H))
            ax.imshow(read_image(data_root / r.path), cmap="gray", vmin=0, vmax=255)
            ax.set_anchor("N")
            ax.set_axis_off()
            n_files = int((x["cls"] == r.cls).sum())
            ax.set_title(f"метка в датасете: {CLS_RU[r.cls].upper()}  ({n_files} файл.)\n{Path(r.path).name}",
                         fontsize=9.5, color=INK, pad=4, loc="left")

        # поле ответа
        bx, by = to_fig(0.12, 5.0)
        bw, bh = (2 * img_w + gap) / (cell_w * ncols), 0.95 / H
        fig.add_artist(Rectangle((bx, by - bh), bw, bh, fill=False, ec=INK, lw=1.4, transform=fig.transFigure))
        fig.text(bx + 0.006, by - 0.14 / H, "Ответ врача:", fontsize=11, fontweight="bold", color=INK, va="top")
        opts = [f"{CLS_LETTER[c]} — {CLS_RU[c]}" for c in classes] + ["? — нельзя определить", "другое: _______"]
        ox = bx + 0.006
        for o in opts:
            sq = 0.2
            sx, sy = ox, by - 0.52 / H
            fig.add_artist(Rectangle((sx, sy - sq / H), sq / (cell_w * ncols), sq / H, fill=False, ec=INK, lw=1.1,
                                     transform=fig.transFigure))
            fig.text(sx + (sq + 0.07) / (cell_w * ncols), sy - sq / (2 * H), o, fontsize=10.5, color=INK, va="center")
            ox += (sq + 0.07 + 0.085 * len(o) + 0.22) / (cell_w * ncols)

        # список всех файлов случая
        lines = []
        for c in classes:
            y = x[x["cls"] == c]
            names = [Path(p).name for p in y.loc[y["is_original"], "path"]]
            n_aug = int((~y["is_original"]).sum())
            lines.append(f"{CLS_RU[c]}: {', '.join(names)}" + (f"  + {n_aug} аугм. копий" if n_aug else ""))
        lx, ly = to_fig(0.12, 6.08)
        fig.text(lx, ly, "Файлы случая — " + ";   ".join(lines), fontsize=8, color=MUTED, va="top", wrap=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("mrt"))
    ap.add_argument("--index", type=Path, default=Path("data/processed/index.csv"))
    ap.add_argument("--groups", type=Path, default=Path("data/processed/groups.csv"))
    ap.add_argument("--pairs", type=Path, default=Path("data/processed/label_conflict_pairs.csv"))
    ap.add_argument("--out", type=Path, default=Path("reports/doctor/label_conflicts_for_doctor.png"))
    ap.add_argument("--cases-csv", type=Path, default=Path("data/processed/label_conflict_cases.csv"))
    args = ap.parse_args()

    cases = build_cases(pd.read_csv(args.index), pd.read_csv(args.groups), pd.read_csv(args.pairs))
    plot_sheet(cases, args.data_root, args.out)
    cols = ["case", "group_id", "path", "split", "cls", "aug", "is_original", "shown", "doctor_label"]
    cases[cols].to_csv(args.cases_csv, index=False, encoding="utf-8-sig")
    print(f"случаев: {cases['case'].nunique()}; файлов: {len(cases)} (оригиналов {int(cases['is_original'].sum())})")
    print(f"лист: {args.out}\nтаблица: {args.cases_csv}")


if __name__ == "__main__":
    main()
