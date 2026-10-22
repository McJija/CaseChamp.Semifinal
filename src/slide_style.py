"""Общий стиль рисунков к слайдам: шрифт, цвета, оформление осей.

Цвета типов опухоли — первые три слота проверенной палитры (синий, оранжевый, бирюзовый), одни и те же
во всех рисунках; «нет опухоли» — нейтральный серый. Данные без типа — оттенки зеленого, как в оформлении
презентации (выделенное — темно-зеленое, для сравнения — светло-мятное); фон кремовый, как у слайдов.
"""
from __future__ import annotations

import matplotlib
from matplotlib import font_manager

TYPE_COLORS = {"glioma": "#2a78d6", "meningioma": "#eb6834", "pituitary": "#1baf7a", "no_tumor": "#898781"}
INK = "#1e1a1b"         # основной текст
INK2 = "#4a4744"        # второстепенный текст
MUTED = "#8a847a"       # подписи осей
LIGHT = "#cfc4b2"       # линии осей
GRID = "#ebe0cc"        # сетка
SURFACE = "#fff6e8"     # фон — кремовый, как у слайдов
SYSTEM_RED = "#e34948"  # обводка системы — как в прототипе и листах врача
REFERENCE_GREEN = "#00b050"  # эталонная обводка Figshare
MARK = "#2a6251"        # выделенные данные
MARK_MID = "#2dcd9d"    # промежуточное значение
MARK_LIGHT = "#a8e8d2"  # данные для сравнения
BOX = "#e6f7ef"         # заливка блоков на схемах
BOX_EDGE = "#8fdcc2"    # контур блоков на схемах


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
