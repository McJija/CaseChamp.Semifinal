"""Общий стиль рисунков к слайдам: шрифт, цвета, оформление осей.

Цвета типов опухоли — первые три слота проверенной палитры (синий, оранжевый, бирюзовый), одни и те же
во всех рисунках; «нет опухоли» — нейтральный серый. Данные без типа — оттенки фиолетового, как акцент колоды
(выделенное — насыщенное, для сравнения — светлое); подписи — оттенки «чернил».
"""
from __future__ import annotations

import matplotlib
from matplotlib import font_manager

TYPE_COLORS = {"glioma": "#2a78d6", "meningioma": "#eb6834", "pituitary": "#1baf7a", "no_tumor": "#898781"}
INK = "#0b0b0b"         # основной текст, выделенные данные
INK2 = "#52514e"        # второстепенный текст
MUTED = "#898781"       # подписи осей
LIGHT = "#c3c2b7"       # данные для сравнения, линии осей
GRID = "#e1e0d9"        # сетка
SURFACE = "#ffffff"     # фон (слайд белый)
SYSTEM_RED = "#e34948"  # обводка системы — как в прототипе и листах врача
REFERENCE_GREEN = "#00b050"  # эталонная обводка Figshare
MARK = "#5b3fe0"        # выделенные данные
MARK_MID = "#9a88f3"    # промежуточное значение
MARK_LIGHT = "#d3ccf8"  # данные для сравнения
BOX = "#f3f0ff"         # заливка блоков на схемах
BOX_EDGE = "#c9c0f2"    # контур блоков на схемах


def apply() -> None:
    family = "Segoe UI"
    try:
        font_manager.findfont(family, fallback_to_default=False)
    except ValueError:
        family = "DejaVu Sans"
    matplotlib.rcParams.update({
        "font.family": family, "font.size": 10, "text.color": INK,
        "axes.edgecolor": LIGHT, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
    })
