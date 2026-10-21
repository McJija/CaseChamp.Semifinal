"""Классы задачи: порядок фиксирован, индекс класса = позиция в CLASSES."""
from __future__ import annotations

CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NO_TUMOR_IDX = CLASS_TO_IDX["no_tumor"]
TUMOR_CLASSES = [c for c in CLASSES if c != "no_tumor"]
CLS_RU = {"glioma": "глиома", "meningioma": "менингиома", "pituitary": "опухоль гипофиза", "no_tumor": "опухоли нет"}
