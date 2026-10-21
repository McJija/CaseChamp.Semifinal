"""Единый препроцессинг снимка: один и тот же код для обучения и инференса.

Шаги:
  1. серый снимок в диапазоне 0–255 (цветные файлы переводятся в серый, 16 бит — min-max);
  2. обрезка: сначала от краев срезаются однородные строки и столбцы (поля любого цвета, в том
     числе белые, и рамки-линии), затем ищется рамка головы — порог берется относительно уровня
     фона, поэтому серый фон BR-копий, шум «соль-перец», подписи и линейки по краям не мешают;
  3. нормализация яркости в дробных числах: фон → 0, 99,5-й перцентиль тканей → 1;
  4. приведение к size×size с сохранением пропорций (длинная сторона = size, поля — нули);
  5. квантование в uint8.
Нормализация идет до масштабирования, чтобы интерполяция заполнила промежуточные уровни
серого (иначе число уровней выдает источник снимка).

Геометрия возвращается в PreprocMeta: по ней маска переносится в координаты модели
(`mask_to_model`), а маска или тепловая карта модели — обратно на исходный снимок
(`map_to_original`).

Пример:
    from src.preprocess import preprocess
    image, meta = preprocess(read_image(path))      # uint8, 256×256
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_SIZE = 256


@dataclass(frozen=True)
class PreprocMeta:
    orig_h: int
    orig_w: int
    y0: int  # рамка содержимого в исходном снимке (правые границы не включаются)
    y1: int
    x0: int
    x1: int
    new_h: int  # размер содержимого после масштабирования
    new_w: int
    top: int  # положение содержимого внутри size×size
    left: int
    size: int
    scale: float  # во сколько раз изменено содержимое
    bg: float  # уровень фона (0–255)
    hi: float  # верхняя граница яркости тканей (0–255)


def to_gray(image: np.ndarray) -> np.ndarray:
    """Серый снимок float32 в диапазоне 0–255. Принимает HxW, HxWx3 (RGB), HxWx4; uint8 или другие типы."""
    x = np.asarray(image)
    if x.ndim == 3:  # те же веса, что у OpenCV при чтении в сером: результат не зависит от способа чтения файла
        x = x[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32) if x.shape[2] >= 3 else x[..., 0]
    x = x.astype(np.float32)
    if np.asarray(image).dtype != np.uint8:  # например, 16-битный DICOM
        lo, hi = float(x.min()), float(x.max())
        x = (x - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(x)
    return x


def trim_uniform_borders(gray: np.ndarray, rel_tol: float = 0.03, frac: float = 0.98) -> tuple[int, int, int, int]:
    """Срезает от краев строки и столбцы однородной яркости: поля любого цвета (черные, серые,
    белые) и рамки-линии по краю кадра. Строка однородна, если не менее frac ее пикселей
    отличаются от медианы строки не больше чем на допуск. Допуск — доля rel_tol от разброса
    яркости снимка: на затемненных снимках тусклые края головы не принимаются за поле.
    Возвращает (y0, y1, x0, x1)."""
    h, w = gray.shape
    y0, y1, x0, x1 = 0, h, 0, w
    lo, hi = np.percentile(gray, [0.5, 99.5])
    tol = float(np.clip(rel_tol * (hi - lo), 2.0, 8.0))

    def edges(block: np.ndarray, axis: int) -> tuple[int, int]:
        med = np.median(block, axis=axis, keepdims=True)
        uniform = (np.abs(block - med) <= tol).mean(axis=axis) >= frac
        if uniform.all():
            return 0, 0
        return int(np.argmin(uniform)), int(np.argmin(uniform[::-1]))  # сколько однородных с начала и с конца

    for _ in range(4):  # строки и столбцы влияют друг на друга — несколько проходов
        top, bottom = edges(gray[y0:y1, x0:x1], axis=1)
        y0, y1 = y0 + top, y1 - bottom
        left, right = edges(gray[y0:y1, x0:x1], axis=0)
        x0, x1 = x0 + left, x1 - right
        if top + bottom + left + right == 0:
            break
    if (y1 - y0) < 0.15 * h or (x1 - x0) < 0.15 * w:  # осталось неправдоподобно мало — не срезаем
        return 0, h, 0, w
    return y0, y1, x0, x1


def estimate_background(gray: np.ndarray) -> float:
    """Уровень фона: меньшее из медианы рамки по краю кадра и 10-го перцентиля всего снимка.
    Рамка дает серый фон BR-копий, перцентиль страхует случай, когда голова касается краев."""
    h, w = gray.shape
    b = max(2, round(0.04 * min(h, w)))
    ring = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(), gray[b:-b, :b].ravel(), gray[b:-b, -b:].ravel()])
    return float(min(np.median(ring), np.percentile(gray, 10)))


def content_bbox(gray: np.ndarray, bg: float) -> tuple[tuple[int, int, int, int], float]:
    """Рамка головы (y0, y1, x0, x1) без отступа и использованный порог яркости."""
    h, w = gray.shape
    sm = cv2.medianBlur(np.clip(gray, 0, 255).astype(np.uint8), 5)  # убирает «соль-перец»
    thr = bg + max(6.0, 0.08 * (float(np.percentile(sm, 99.5)) - bg))
    m = (sm > thr).astype(np.uint8)
    k = max(5, int(round(0.015 * min(h, w))) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))  # тонкие линии, подписи
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    full = (0, h, 0, w)
    if n <= 1:
        return full, thr
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = stats[1:][areas >= 0.05 * areas.max()]  # голова и сравнимые с ней части; мелкие метки отбрасываются
    x0, y0 = int(keep[:, cv2.CC_STAT_LEFT].min()), int(keep[:, cv2.CC_STAT_TOP].min())
    x1 = int((keep[:, cv2.CC_STAT_LEFT] + keep[:, cv2.CC_STAT_WIDTH]).max())
    y1 = int((keep[:, cv2.CC_STAT_TOP] + keep[:, cv2.CC_STAT_HEIGHT]).max())
    if (y1 - y0) < 0.15 * h or (x1 - x0) < 0.15 * w:  # рамка неправдоподобно мала — не обрезаем
        return full, thr
    return (y0, y1, x0, x1), thr


def preprocess(image: np.ndarray, size: int = DEFAULT_SIZE) -> tuple[np.ndarray, PreprocMeta]:
    """Снимок → (uint8 size×size, геометрия)."""
    gray = to_gray(image)
    h, w = gray.shape
    ty0, ty1, tx0, tx1 = trim_uniform_borders(gray)
    inner = gray[ty0:ty1, tx0:tx1]
    bg = estimate_background(inner)
    (y0, y1, x0, x1), thr = content_bbox(inner, bg)
    y0, y1, x0, x1 = y0 + ty0, y1 + ty0, x0 + tx0, x1 + tx0

    # Отступ 2% вокруг рамки. В срезанную полосу он заходит, только если полоса темная (обычное
    # поле): так рамка одинакова у чистого снимка и у его зашумленной копии, где поля не срезаются.
    # Если в полосе белое поле или линия — отступ ограничен краем срезанной области.
    pad = int(round(0.02 * max(y1 - y0, x1 - x0)))
    dark = lambda strip: strip.size == 0 or float(np.percentile(strip, 99)) <= thr
    lim_y0 = 0 if dark(gray[:ty0]) else ty0
    lim_y1 = h if dark(gray[ty1:]) else ty1
    lim_x0 = 0 if dark(gray[:, :tx0]) else tx0
    lim_x1 = w if dark(gray[:, tx1:]) else tx1
    y0, y1 = max(lim_y0, y0 - pad), min(lim_y1, y1 + pad)
    x0, x1 = max(lim_x0, x0 - pad), min(lim_x1, x1 + pad)
    crop = gray[y0:y1, x0:x1]

    tissue = crop[crop > thr]
    hi = float(np.percentile(tissue, 99.5)) if tissue.size else float(crop.max())
    norm = np.clip((crop - bg) / max(hi - bg, 1e-6), 0.0, 1.0)

    ch, cw = crop.shape
    scale = size / max(ch, cw)
    new_h, new_w = max(1, round(ch * scale)), max(1, round(cw * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = np.clip(cv2.resize(norm, (new_w, new_h), interpolation=interp), 0.0, 1.0)

    top, left = (size - new_h) // 2, (size - new_w) // 2
    out = np.zeros((size, size), np.float32)
    out[top : top + new_h, left : left + new_w] = resized
    meta = PreprocMeta(h, w, y0, y1, x0, x1, new_h, new_w, top, left, size, float(scale), bg, hi)
    return np.round(out * 255).astype(np.uint8), meta


def mask_to_model(mask: np.ndarray, meta: PreprocMeta) -> np.ndarray:
    """Маска в координатах исходного снимка → бинарная маска size×size (uint8, 0/1)."""
    if mask.shape[:2] != (meta.orig_h, meta.orig_w):
        raise ValueError(f"размер маски {mask.shape[:2]} не совпадает со снимком {(meta.orig_h, meta.orig_w)}")
    crop = (mask[meta.y0 : meta.y1, meta.x0 : meta.x1] > 0).astype(np.float32)
    interp = cv2.INTER_AREA if meta.scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (meta.new_w, meta.new_h), interpolation=interp) >= 0.5
    out = np.zeros((meta.size, meta.size), np.uint8)
    out[meta.top : meta.top + meta.new_h, meta.left : meta.left + meta.new_w] = resized
    return out


def map_to_original(arr: np.ndarray, meta: PreprocMeta, binary: bool = True) -> np.ndarray:
    """Маска (binary=True) или тепловая карта модели size×size → координаты исходного снимка."""
    inner = arr[meta.top : meta.top + meta.new_h, meta.left : meta.left + meta.new_w].astype(np.float32)
    back = cv2.resize(inner, (meta.x1 - meta.x0, meta.y1 - meta.y0), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((meta.orig_h, meta.orig_w), np.float32)
    out[meta.y0 : meta.y1, meta.x0 : meta.x1] = back
    return (out >= 0.5).astype(np.uint8) if binary else out
