"""Чтение изображений.

cv2.imread на Windows не открывает пути с кириллицей, поэтому читаем байты
через numpy и декодируем через cv2.imdecode.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path, gray: bool = True) -> np.ndarray:
    """Читает изображение: uint8, HxW (gray) или HxWx3 RGB."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imdecode(buf, flag)
    if img is None:
        raise ValueError(f"Не удалось декодировать изображение: {path}")
    if not gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
