"""Рисунки к слайдам презентации.

Числа берутся из готовых отчетов и таблиц проекта; модели не обучаются и не меняются (используются
замороженные models/final/). Снимки для иллюстраций: проверочная часть — рисунки 1, 4, 11; тестовая —
рисунок 9 (обводки), снимки выбираются правилом, без ручного отбора. Результат — reports/slides/figNN_*.png.
Рисунок 12 (причины ошибок) рисует src/import_doctor_answers.py.

Пример:
    python -m src.make_slide_figures              # все рисунки
    python -m src.make_slide_figures --only 3 5   # выборочно
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from src import slide_style as st  # noqa: E402
from src.dataset import PreprocCache, to_tensor  # noqa: E402
from src.inference import _load  # noqa: E402
from src.io_utils import read_image  # noqa: E402
from src.labels import CLASSES, CLS_RU, NO_TUMOR_IDX  # noqa: E402
from src.stress import occlude, tumor_region  # noqa: E402

TYPES = ["glioma", "meningioma", "pituitary"]
BOX_FC = st.BOX


# ---------------------------------------------------------------- общие помощники


class Models:
    """Замороженные модели системы и кэш снимков (координаты модели 256×256)."""

    def __init__(self, final: Path, cache_dir: Path):
        self.cls, _ = _load(final / "classifier.pt", torch.device("cpu"))
        self.seg, _ = _load(final / "segmenter.pt", torch.device("cpu"))
        self.cache = PreprocCache(cache_dir, "pool", 256)

    @torch.no_grad()
    def run(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = to_tensor(img).unsqueeze(0)
        probs = torch.softmax(self.cls(x).float(), 1)[0].numpy()
        mask = (torch.sigmoid(self.seg(x).float())[0, 0] > 0.5).numpy()
        return probs, mask


def contours(mask: np.ndarray) -> list[np.ndarray]:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c[:, 0, :] for c in cs if len(c) > 2]


def draw_contours(ax, mask: np.ndarray, color: str, lw: float = 1.6, extent=None, ls="-") -> None:
    h, w = mask.shape
    for c in contours(mask):
        x, y = c[:, 0].astype(float), c[:, 1].astype(float)
        x, y = np.append(x, x[0]), np.append(y, y[0])
        if extent is not None:  # из пикселей в координаты данных (imshow с extent, начало сверху)
            x0, x1, y0, y1 = extent
            x, y = x0 + (x + 0.5) / w * (x1 - x0), y1 - (y + 0.5) / h * (y1 - y0)
        ax.plot(x, y, color=color, lw=lw, ls=ls)


def box(ax, x0, y0, x1, y1, title, body="", fc=BOX_FC, ec=st.BOX_EDGE, ls="-", fs=11.5) -> None:
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.1, ls=ls))
    cx = (x0 + x1) / 2
    if body:
        ax.text(cx, y1 - 0.22, title, ha="center", va="top", fontsize=fs, fontweight="bold", color=st.INK)
        ax.text(cx, y1 - 0.62, body, ha="center", va="top", fontsize=fs - 1.5, color=st.INK2, linespacing=1.3)
    else:
        ax.text(cx, (y0 + y1) / 2, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=st.INK)


def arrow(ax, x0, y0, x1, y1, ls="-", color=st.INK2, label="", label_dy=0.14) -> None:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=13))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + label_dy, label, ha="center", va="bottom", fontsize=9.5, color=st.MUTED)


def canvas(w: float, h: float, xlim: float, ylim: float):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}".replace(".", ",") + "%"


def num(x: int) -> str:
    return f"{x:,}".replace(",", " ")


# ---------------------------------------------------------------- рисунки


def fig01_system(m: Models, out: Path) -> None:
    """Схема системы с настоящим примером с проверочной части (менингиома, обводка близка к эталонной)."""
    pv = pd.read_csv("reports/eval_val/predictions.csv", keep_default_na=False)
    pv = pv[(pv.root == "figshare") & (pv.cls == "meningioma") & (pv.answer == "meningioma")].copy()
    pv["dice"] = pd.to_numeric(pv.dice, errors="coerce")
    row = pv.iloc[int((pv.dice - 0.9).abs().argmin())]
    raw = read_image(Path("data/external/figshare/png") / row.path)
    img = m.cache.image(row.path)
    probs, mask = m.run(img)

    fig, ax = canvas(13, 4.6, 16.4, 5.8)
    ex_in, ex_out = (0.1, 2.5, 1.9, 4.3), (13.9, 16.3, 1.9, 4.3)
    ax.imshow(raw, cmap="gray", extent=ex_in, interpolation="antialiased")
    ax.text(1.3, 1.7, "Снимок МРТ\nT1 с контрастом", ha="center", va="top", fontsize=10, color=st.INK)
    arrow(ax, 2.6, 3.1, 3.2, 3.1)
    box(ax, 3.3, 2.1, 6.0, 4.1, "Подготовка", "обрезка черных полей\nединый размер\nвыравнивание яркости")
    arrow(ax, 6.0, 3.4, 6.7, 4.1)
    arrow(ax, 6.0, 2.8, 6.7, 2.1)
    box(ax, 6.8, 3.5, 10.0, 5.4, "Модель 1 — тип", "называет тип опухоли\nи уверенность;\nтепловая карта —\nкуда смотрела")
    box(ax, 6.8, 0.8, 10.0, 2.7, "Модель 2 — контур", "обводит опухоль\nпо точкам снимка")
    arrow(ax, 10.0, 4.4, 10.6, 3.9)
    arrow(ax, 10.0, 1.8, 10.6, 2.3)
    box(ax, 10.7, 1.2, 13.5, 5.0, "Ответ", "«опухоль есть», если\nмодель 2 нашла\nучасток > 50 точек\nили модель 1 уверена\nна 97,6% и больше\n\nтип — от модели 1\n\nпометка — если модели\nне согласны",
        ec=st.MARK, fs=11)
    arrow(ax, 13.5, 3.1, 13.85, 3.1)
    ax.imshow(img, cmap="gray", extent=ex_out, interpolation="antialiased")
    draw_contours(ax, mask, st.SYSTEM_RED, lw=1.4, extent=ex_out)
    k = int(probs[:NO_TUMOR_IDX].argmax())
    ax.text(15.1, 1.7, f"Экран врача\n{CLS_RU[CLASSES[k]]},\nуверенность {probs[k]:.0%}",
            ha="center", va="top", fontsize=10, color=st.INK)
    fig.savefig(out)
    plt.close(fig)


def fig02_data(out: Path) -> None:
    s = pd.read_csv("data/processed/splits.csv", keep_default_na=False)
    s["primary"] = s.primary.astype(str).str.lower() == "true"
    n_files = len(s)
    n_named_orig = int((s.aug == "orig").sum())
    n_unique = int(s.primary.sum())
    n_work = int((s.primary & s.split.isin(["train", "val", "test"])).sum())
    p = pd.read_csv("data/processed/pool.csv", keep_default_na=False)
    n_fig = int((p.source == "figshare").sum())
    n_case_only = len(p) - n_fig
    assert n_fig == 3064 and len(p) == n_fig + n_case_only

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.42})
    steps = [("Файлы организаторов", n_files, ""),
             ("Без искаженных копий (осветленных,\nотраженных, с шумом и т. п.)", n_named_orig, f"убрали {num(n_files - n_named_orig)}"),
             ("Уникальные снимки:\nубраны повторы одного снимка", n_unique, f"убрали {num(n_named_orig - n_unique)}"),
             ("Рабочие снимки: после ручной проверки\nкласса «нет опухоли»", n_work, f"убрали {num(n_unique - n_work)}")]
    y = np.arange(len(steps))[::-1]
    for yi, (label, v, minus) in zip(y, steps):
        a1.barh(yi, v, color=st.MARK if yi == 0 else st.MARK_LIGHT, height=0.62)
        a1.text(v + 150, yi, ("осталось " if minus else "") + num(v) + (f"  ({minus})" if minus else ""), va="center", fontsize=15, color=st.INK)
    a1.set_yticks(y, [s_[0] for s_ in steps], fontsize=13.5)
    a1.tick_params(axis="y", length=0)
    a1.set_xticks([])
    a1.spines[["left", "bottom"]].set_visible(False)
    a1.set_xlim(0, n_files * 1.6)
    a1.set_title("Из 12 064 файлов — 3 574 уникальных рабочих снимка", fontsize=16)

    parts = ["train", "val", "test"]
    names = {"train": "обучение", "val": "проверка", "test": "тест"}
    fig_n = [int(((p.split == s_) & (p.source == "figshare")).sum()) for s_ in parts]
    case_n = [int(((p.split == s_) & (p.source != "figshare")).sum()) for s_ in parts]
    yy = np.arange(3)[::-1]
    a2.barh(yy, fig_n, color=st.MARK, height=0.62, label="срезы Figshare с обводкой")
    a2.barh(yy, case_n, left=fig_n, color=st.MARK_LIGHT, height=0.62, label="снимки кейса без пары в Figshare")
    for yi, f_, c_ in zip(yy, fig_n, case_n):
        a2.text(f_ + c_ + 40, yi, num(f_ + c_), va="center", fontsize=15, color=st.INK)
    a2.set_yticks(yy, [names[s_] for s_ in parts], fontsize=15)
    a2.tick_params(axis="y", length=0)
    a2.set_xticks([])
    a2.spines[["left", "bottom"]].set_visible(False)
    a2.set_xlim(0, max(np.add(fig_n, case_n)) * 1.2)
    a2.set_title(f"Набор для обучения и оценки: {num(len(p))}", fontsize=16)
    a2.legend(loc="upper left", bbox_to_anchor=(0.0, -0.02), frameon=False, fontsize=13, ncol=1)
    fig.savefig(out)
    plt.close(fig)


def fig03_leakage(metrics: dict, out: Path) -> None:
    """Цена утечки: на тесте организаторов «как есть» — доля файлов с верным ответом (тип опухоли или «нет опухоли»)
    у пациентов, которых модель видела при обучении, и у новых пациентов. 100% — все файлы группы распознаны верно."""
    o = metrics["organizers_test"]
    seen, new, total = o["пациенты из нашего train и val"], o["наши тестовые пациенты"], o["весь тест организаторов"]
    rows = [(f"Пациенты, которых модель\nвидела при обучении\n({num(seen['n'])} файлов)", seen, st.MARK_LIGHT),
            (f"Новые пациенты\n({num(new['n'])} файлов)", new, st.MARK)]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    y = np.arange(len(rows))[::-1]
    for yi, (label, d, color) in zip(y, rows):
        ax.barh(yi, 1.0, color=st.GRID, height=0.62)            # фон до 100%
        ax.barh(yi, d["acc"], color=color, height=0.62)
        ax.text(d["acc"] - 0.02, yi, pct(d["acc"], 0), va="center", ha="right", fontsize=19, fontweight="bold",
                color=st.INK if color == st.MARK_LIGHT else "#ffffff")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=12.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"], fontsize=11)
    ax.spines[["left"]].set_visible(False)
    ax.set_title("Сколько файлов теста организаторов\nсистема распознала верно", fontsize=14)
    fig.text(0.01, 0.01, f"100% — все файлы группы с верным ответом.\nВесь тест организаторов ({num(total['n'])} файлов) — "
             f"{pct(total['acc'], 0)}:\nпочти все его файлы — знакомые пациенты.", fontsize=10.5, color=st.INK2)
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(out)
    plt.close(fig)


def fig04_occlusion(m: Models, out: Path) -> None:
    """Проверочная часть: по одному срезу каждого типа (площадь опухоли ближе всего к медиане своего типа)."""
    occ = yaml.safe_load(Path("configs/eval_definitions.yaml").read_text(encoding="utf-8"))["occlusion"]
    t = m.cache.table
    t = t[(t.split == "val") & (t.root == "figshare") & t.has_mask]
    fig, axes = plt.subplots(2, 3, figsize=(6.4, 5.4), gridspec_kw={"hspace": 0.5, "wspace": 0.06})
    for j, typ in enumerate(TYPES):
        sub = t[t.cls == typ]
        areas = np.array([(m.cache.mask(p_) > 0).sum() for p_ in sub.path])
        path = sub.path.iloc[int(np.abs(areas - np.median(areas)).argmin())]
        img, mask = m.cache.image(path), m.cache.mask(path) > 0
        filled = occlude(img, tumor_region(mask, occ), occ)
        for i, (im, title) in enumerate(((img, "исходный снимок"), (filled, "опухоль закрашена"))):
            probs, _ = m.run(im)
            p_t = 1 - probs[NO_TUMOR_IDX]
            ans = CLASSES[int(probs.argmax())]
            ax = axes[i, j]
            ax.imshow(im, cmap="gray", interpolation="antialiased")
            draw_contours(ax, mask, st.REFERENCE_GREEN if i == 0 else "#ffffff", lw=1.3, ls="-" if i == 0 else ":")
            ax.set_xticks([])
            ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
            ax.set_title(f"{CLS_RU[typ].replace('опухоль гипофиза', 'гипофиз')}\n{title}" if i == 0 else title, fontsize=9.5, loc="center")
            ax.text(0.5, -0.04, f"вероятность опухоли {pct(p_t)}\nответ: {CLS_RU[ans]}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=9, color=st.INK)
    fig.savefig(out)
    plt.close(fig)


def fig05_seg_attempts(out: Path) -> None:
    e = pd.read_csv("reports/experiments.csv")
    runs = [("run1_seg", "1. обычное обучение"), ("run1_seg_crop", "2. учили на кусочках снимка"),
            ("run1_seg_cf", "3. учили на снимках\n    с закрашенной опухолью"), ("run1_seg_neg", "4. добавили 246 здоровых\n    снимков с пустой обводкой")]
    rows = []
    for r, label in runs:
        x = e[e.run == r]
        assert len(x) == 3, r
        fm = x.val_false_mask_rate * x.val_n_no_tumor_seg
        rows.append((label, fm.mean(), fm.min(), fm.max(), x.val_dice.mean(), x.val_dice.min(), x.val_dice.max(), r == "run1_seg_neg"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.0), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        label, fm, fl, fh, d, dl, dh, chosen = r
        c = st.MARK if chosen else st.MARK_LIGHT
        a1.barh(yi, fm, color=c, height=0.6)
        a1.plot([fl, fh], [yi, yi], color=st.INK2, lw=1.2)
        a1.text(max(fh, fm) + 1.2, yi, f"{fl:.0f}–{fh:.0f} из 56", va="center", fontsize=13, color=st.INK)
        a2.barh(yi, d, color=c, height=0.6)
        a2.plot([dl, dh], [yi, yi], color=st.INK2, lw=1.2)
        a2.text(dh + 0.02, yi, pct(d, 0), va="center", fontsize=13, color=st.INK)
    a1.set_yticks(y, [r[0] for r in rows], fontsize=13)
    a1.tick_params(axis="y", length=0)
    a2.tick_params(axis="y", length=0)
    a1.set_xlim(0, 80)
    a2.set_xlim(0, 1.0)
    for a, title in ((a1, "Здоровые снимки, на которых модель\nошибочно обвела «опухоль» — меньше лучше"),
                     (a2, "Совпадение контура модели\nс обводкой врача — больше лучше")):
        a.set_xticks([])
        a.spines[["left", "bottom"]].set_visible(False)
        a.set_title(title, fontsize=14)
    fig.text(0.125, -0.06, "Проверочная часть: 56 здоровых снимков и 391 снимок с обводкой врача. Каждый вариант обучали трижды: "
             "полоса — среднее, отрезок — от худшего до лучшего повтора.", fontsize=10.5, color=st.MUTED)
    fig.savefig(out)
    plt.close(fig)


def fig06_subgroups(metrics: dict, out: Path) -> None:
    keys = [("опухоли: все", "Опухоли — все"), ("опухоли: figshare", "  снимки Figshare"),
            ("опухоли: figshare_crop", "  Figshare, другое кадрирование"), ("опухоли: other", "  снимки из других источников"),
            ("нет опухоли: все", "Нет опухоли — все"), ("нет опухоли: T1", "  режим МРТ T1"), ("нет опухоли: другой режим", "  другие режимы МРТ")]
    sysd = {r["группа"]: r for r in metrics["presence"]}
    clsd = {r["группа"]: r for r in metrics["presence_classifier_only"]}
    fig, ax = plt.subplots(figsize=(11, 4.6))
    y = np.arange(len(keys))[::-1].astype(float)
    y[4:] -= 0.5  # промежуток между блоками
    for yi, (k, _) in zip(y, keys):
        for d, dy, face, edge in ((sysd[k], 0.14, st.MARK, st.MARK), (clsd[k], -0.14, st.SURFACE, st.MUTED)):
            ax.plot([d["lo"], d["hi"]], [yi + dy] * 2, color=edge, lw=1.4)
            ax.plot(d["rate"], yi + dy, "o", ms=7.5, mfc=face, mec=edge, mew=1.5)
        ax.text(1.012, yi + 0.14, f"{sysd[k]['k']} из {sysd[k]['n']}", va="center", fontsize=9.5, color=st.INK)
        ax.text(1.012, yi - 0.14, f"{clsd[k]['k']} из {clsd[k]['n']}", va="center", fontsize=9.5, color=st.MUTED)
    ax.set_yticks(y, [lbl for _, lbl in keys], fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.7, 1.0)
    ax.set_xticks([0.7, 0.8, 0.9, 1.0], ["70%", "80%", "90%", "100%"])
    ax.grid(axis="x", color=st.GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[["left"]].set_visible(False)
    ax.plot([], [], "o", ms=7.5, mfc=st.MARK, mec=st.MARK, label="система (две модели)")
    ax.plot([], [], "o", ms=7.5, mfc=st.SURFACE, mec=st.MUTED, mew=1.5, label="одна модель типа")
    ax.legend(loc="lower left", frameon=False, fontsize=9.5, ncol=2, bbox_to_anchor=(0, 1.0))
    ax.set_title("Итоговый тест: доля найденных опухолей и верных ответов «опухоли нет»", pad=26)
    fig.text(0.125, -0.03, "Точка — доля верных ответов; отрезок — где с вероятностью 95% лежит настоящее значение. "
             "Снимков без опухоли в тесте 49, поэтому отрезки широкие.", fontsize=8.5, color=st.MUTED)
    fig.savefig(out)
    plt.close(fig)


def fig07_confusion(metrics: dict, out: Path) -> None:
    cm = np.array(metrics["cls"]["confusion"])
    share = cm / cm.sum(1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("mark", ["#fff6e8", "#2a6251"])
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            dark = share[i, j] > 0.5
            ax.text(j, i - 0.1, str(cm[i, j]), ha="center", va="center", fontsize=13, fontweight="bold",
                    color="#ffffff" if dark else st.INK)
            ax.text(j, i + 0.22, pct(share[i, j], 0), ha="center", va="center", fontsize=8.5, color="#ffffff" if dark else st.MUTED)
    names = [CLS_RU[c].replace("опухоль гипофиза", "гипофиз").replace("опухоли нет", "нет опухоли") for c in CLASSES]
    ax.set_xticks(range(4), names, fontsize=9.5)
    ax.set_yticks(range(4), names, fontsize=9.5)
    ax.xaxis.tick_top()
    ax.set_xlabel("ответ системы", labelpad=8)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("правильный ответ")
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    k, n = metrics["cls"]["type_correct_among_found"]["k"], metrics["cls"]["type_correct_among_found"]["n"]
    fig.text(0.02, 0.01, f"Итоговый тест, 582 снимка. Тип верен у {k} из {n} найденных опухолей ({pct(k / n)}); "
             "чаще всего путаются глиома и менингиома.", fontsize=8.5, color=st.MUTED)
    fig.savefig(out)
    plt.close(fig)


def fig08_copies(metrics: dict, out: Path) -> None:
    names = {"BR": "ярче в 1,9 раза", "DA": "темнее в 2 раза", "HF": "отражен слева направо", "RO": "немного повернут",
             "VF": "перевернут вверх ногами", "SP": "шум: белые и черные точки"}
    c = metrics["copies"]
    rows = sorted(((names[k], v["present_ok"], v["n"]) for k, v in c.items()), key=lambda r: -r[1])
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    y = np.arange(len(rows))[::-1]
    for yi, (label, v, n) in zip(y, rows):
        ax.barh(yi, v, color=st.MARK if v >= 0.95 else st.MARK_LIGHT, height=0.6)
        ax.text(v + 0.012, yi, f"{pct(v)} из {n} копий", va="center", fontsize=11, color=st.INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.32)
    ax.set_xticks([])
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.set_title("Искаженные копии снимков: доля копий, где система\nверно ответила, есть ли опухоль", fontsize=12)
    fig.savefig(out)
    plt.close(fig)


def fig09_contours(m: Models, out: Path) -> None:
    """Тест, срезы Figshare с обводкой: по каждому типу — срез с Dice ближе всего к медиане типа (типичный)
    и к 10-му перцентилю (трудный)."""
    pt = pd.read_csv("reports/eval_test/predictions.csv", keep_default_na=False)
    pt = pt[(pt.root == "figshare") & (pt.has_mask.astype(str).str.lower() == "true")].copy()
    pt["dice"] = pd.to_numeric(pt.dice)
    fig, axes = plt.subplots(2, 3, figsize=(5.6, 5.0), gridspec_kw={"hspace": 0.35, "wspace": 0.06})
    for j, typ in enumerate(TYPES):
        sub = pt[pt.cls == typ]
        for i, q in enumerate((0.5, 0.1)):
            target = sub.dice.quantile(q)
            row = sub.iloc[int((sub.dice - target).abs().argmin())]
            img, ref = m.cache.image(row.path), m.cache.mask(row.path) > 0
            _, pred = m.run(img)
            both = ref | pred
            ys, xs = np.nonzero(both)
            cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
            half = int(max(40, 1.1 * max(ys.max() - ys.min(), xs.max() - xs.min())))
            y0, y1 = int(np.clip(cy - half, 0, 256 - 2 * half)), int(np.clip(cy - half, 0, 256 - 2 * half)) + 2 * half
            x0, x1 = int(np.clip(cx - half, 0, 256 - 2 * half)), int(np.clip(cx - half, 0, 256 - 2 * half)) + 2 * half
            ax = axes[i, j]
            ax.imshow(img[y0:y1, x0:x1], cmap="gray", interpolation="antialiased")
            draw_contours(ax, ref[y0:y1, x0:x1], st.REFERENCE_GREEN, lw=1.6)
            draw_contours(ax, pred[y0:y1, x0:x1], st.SYSTEM_RED, lw=1.6)
            ax.set_xticks([])
            ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
            kind = "типичный случай" if q == 0.5 else "трудный случай"
            ax.set_title(f"{CLS_RU[typ].replace('опухоль гипофиза', 'гипофиз')}\nсовпадение {pct(row.dice, 0)}", fontsize=9.5, loc="center")
            if j == 0:
                ax.set_ylabel(kind, fontsize=10.5, color=st.INK2)
    fig.text(0.5, 0.02, "красная линия — контур системы, зеленая — обводка врача", ha="center", fontsize=9.5, color=st.INK2)
    fig.savefig(out)
    plt.close(fig)


def fig10_deployment(out: Path) -> None:
    fig, ax = canvas(13.5, 5.0, 18.6, 6.9)
    ax.add_patch(FancyBboxPatch((0.15, 0.35), 15.0, 6.2, boxstyle="round,pad=0,rounding_size=0.2", fc="none", ec=st.MUTED, lw=1.1, ls=(0, (5, 4))))
    ax.text(0.4, 6.3, "Больница: снимки и результаты не выходят наружу", fontsize=10, color=st.INK2, va="top", fontweight="bold")
    box(ax, 0.5, 3.9, 2.9, 5.4, "Томограф МРТ")
    box(ax, 3.7, 3.9, 6.1, 5.4, "Архив (PACS)", "хранит снимки\nи результаты")
    box(ax, 6.9, 3.9, 11.7, 5.4, "Отбор снимков", "пропускает только МРТ головы\nв режиме T1 с контрастом")
    box(ax, 12.35, 3.9, 14.95, 5.4, "Журнал", "кто, когда и какой\nснимок смотрел")
    box(ax, 0.5, 0.8, 5.3, 3.0, "Рабочее место врача", "снимок, контур, уверенность,\nпометка «требует внимания»\nв привычной программе")
    box(ax, 6.9, 0.8, 11.7, 3.0, "Сервер с системой", "две модели, ответ, пометка\nбез видеокарты\n0,6 с на снимок", ec=st.MARK)
    box(ax, 15.9, 0.8, 18.4, 3.0, "Доучивание", "контуры,\nисправленные\nврачом, без\nданных пациента", ls=(0, (4, 3)))
    arrow(ax, 2.9, 4.65, 3.7, 4.65)
    arrow(ax, 6.1, 4.65, 6.9, 4.65)
    arrow(ax, 9.3, 3.9, 9.3, 3.0)
    ax.text(9.45, 3.45, "нужные снимки", fontsize=9.5, color=st.MUTED, va="center")
    arrow(ax, 7.4, 3.0, 5.9, 3.9)
    ax.text(7.05, 3.62, "результат\nотдельной серией", fontsize=9.5, color=st.MUTED, ha="left", va="center")
    arrow(ax, 4.3, 3.9, 4.3, 3.0)
    ax.text(4.15, 3.45, "просмотр", fontsize=9.5, color=st.MUTED, ha="right", va="center")
    arrow(ax, 11.7, 2.7, 12.9, 3.9)
    arrow(ax, 11.7, 1.4, 15.9, 1.4, ls=(0, (4, 3)), label="без имен пациентов")
    fig.savefig(out)
    plt.close(fig)


def fig11_preannotation(m: Models, out: Path) -> None:
    """Макет экрана предразметки: пример с проверочной части (менингиома с неидеальной обводкой)."""
    pv = pd.read_csv("reports/eval_val/predictions.csv", keep_default_na=False)
    pv = pv[(pv.root == "figshare") & (pv.cls == "meningioma")].copy()
    pv["dice"] = pd.to_numeric(pv.dice, errors="coerce")
    pv["mask_area"] = pd.to_numeric(pv.mask_area)
    cand = pv[pv.dice.between(0.6, 0.8)]
    row = cand.loc[cand.mask_area.idxmax()]  # обводка заметно неидеальна, опухоль крупная
    img = m.cache.image(row.path)
    _, pred = m.run(img)

    fig = plt.figure(figsize=(7.7, 3.4))
    ax = fig.add_axes((0.0, 0.02, 0.44, 0.86))
    ax.imshow(img, cmap="gray", interpolation="antialiased")
    draw_contours(ax, pred, st.SYSTEM_RED, lw=1.8)
    for c in sorted(contours(pred), key=len)[-1:]:  # маркеры — на самом крупном контуре
        for pnt in c[:: max(1, len(c) // 8)]:
            ax.add_patch(Rectangle((pnt[0] - 1.8, pnt[1] - 1.8), 3.6, 3.6, fc="#ffffff", ec=st.SYSTEM_RED, lw=0.9))
    ys, xs = np.nonzero(pred)
    ax.add_patch(plt.Circle((xs.max() + 9, ys.mean()), 6, fc="none", ec="#ffd400", lw=1.6))
    ax.text(xs.max() + 18, ys.mean(), "кисть", color="#ffd400", fontsize=9, va="center")
    ax.axis("off")
    ax.set_title("Контур предложила модель", fontsize=10.5, loc="left")

    p = fig.add_axes((0.48, 0.02, 0.52, 0.9))
    p.set_xlim(0, 10)
    p.set_ylim(0, 10)
    p.axis("off")
    p.add_patch(FancyBboxPatch((7.6, 9.1), 2.3, 0.85, boxstyle="round,pad=0,rounding_size=0.15", fc="#fff4cc", ec="#d4a800"))
    p.text(8.75, 9.52, "МАКЕТ", ha="center", va="center", fontsize=10, fontweight="bold", color="#7a5c00")
    p.text(0.1, 9.55, "Предварительная разметка", fontsize=12.5, fontweight="bold", va="center")
    p.text(0.1, 8.5, "Снимок 12 из 300 · сначала те,\nгде модель сомневается", fontsize=9.5, color=st.INK2, va="center", linespacing=1.3)
    buttons = [("Принять обводку", st.MARK, "#ffffff"), ("Дорисовать кистью", BOX_FC, st.INK), ("Стереть лишнее", BOX_FC, st.INK),
               ("Сохранить и следующий снимок", BOX_FC, st.INK)]
    for i, (label, fc, tc) in enumerate(buttons):
        y0 = 6.2 - i * 1.6
        p.add_patch(FancyBboxPatch((0.1, y0), 9.0, 1.25, boxstyle="round,pad=0,rounding_size=0.2", fc=fc, ec=st.LIGHT))
        p.text(4.6, y0 + 0.62, label, ha="center", va="center", fontsize=10.5, color=tc, fontweight="bold" if i == 0 else "normal")
    fig.savefig(out)
    plt.close(fig)


FIGS = {1: "fig01_system", 2: "fig02_data", 3: "fig03_leakage", 4: "fig04_occlusion", 5: "fig05_seg_attempts",
        6: "fig06_subgroups", 7: "fig07_confusion", 8: "fig08_copies", 9: "fig09_contours", 10: "fig10_deployment",
        11: "fig11_preannotation"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", type=int, nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=Path("reports/slides"))
    ap.add_argument("--final", type=Path, default=Path("models/final"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/processed/cache"))
    args = ap.parse_args()
    st.apply()
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(Path("reports/eval_test/metrics.json").read_text(encoding="utf-8"))
    todo = args.only or list(FIGS)
    models = Models(args.final, args.cache_dir) if set(todo) & {1, 4, 9, 11} else None
    calls = {1: lambda p: fig01_system(models, p), 2: fig02_data, 3: lambda p: fig03_leakage(metrics, p),
             4: lambda p: fig04_occlusion(models, p), 5: fig05_seg_attempts, 6: lambda p: fig06_subgroups(metrics, p),
             7: lambda p: fig07_confusion(metrics, p), 8: lambda p: fig08_copies(metrics, p), 9: lambda p: fig09_contours(models, p),
             10: fig10_deployment, 11: lambda p: fig11_preannotation(models, p)}
    for k in todo:
        path = args.out / f"{FIGS[k]}.png"
        calls[k](path)
        print(f"рисунок {k}: {path}", flush=True)


if __name__ == "__main__":
    main()
