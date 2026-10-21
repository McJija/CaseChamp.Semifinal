"""Общий код сравнения изображений по содержимому.

Признаки снимка: md5 файла, обрезка черных полей, pHash для 8 преобразований
группы D4 (повороты на 90° и отражения), миниатюра 64×64. Сравнение пары:
минимальное по преобразованиям расстояние Хэмминга pHash + корреляция Пирсона
миниатюр с учетом найденного преобразования.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.io_utils import read_image

# Преобразования группы D4. Индекс сохраняется в паре как tf.
TF_NAMES = ["id", "rot90", "rot180", "rot270", "flipH", "flipV", "transpose", "antitranspose"]
TFS = [
    lambda a: a,
    lambda a: np.rot90(a, 1),
    lambda a: np.rot90(a, 2),
    lambda a: np.rot90(a, 3),
    lambda a: a[:, ::-1],
    lambda a: a[::-1, :],
    lambda a: a.T,
    lambda a: np.rot90(a, 2).T,
]
# Те же преобразования для батча (B, H, W), порядок совпадает с TFS
TFS_BATCH = [
    lambda a: a,
    lambda a: np.rot90(a, 1, axes=(1, 2)),
    lambda a: np.rot90(a, 2, axes=(1, 2)),
    lambda a: np.rot90(a, 3, axes=(1, 2)),
    lambda a: a[:, :, ::-1],
    lambda a: a[:, ::-1, :],
    lambda a: a.transpose(0, 2, 1),
    lambda a: np.rot90(a, 2, axes=(1, 2)).transpose(0, 2, 1),
]


def black_bbox(gray: np.ndarray, thr: int = 15) -> tuple[int, int, int, int]:
    """Рамка содержимого без черных полей: (y0, y1, x0, x1), правые границы не включаются.
    Маска строится по медианно-сглаженному снимку, чтобы шум «соль-перец» на полях не мешал."""
    m = cv2.medianBlur(gray, 5) > thr
    rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    if rows.size < 8 or cols.size < 8:
        return 0, gray.shape[0], 0, gray.shape[1]
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def crop_black(gray: np.ndarray, thr: int = 15) -> np.ndarray:
    y0, y1, x0, x1 = black_bbox(gray, thr)
    return gray[y0:y1, x0:x1]


def phash_u64(arr: np.ndarray) -> np.uint64:
    bits = imagehash.phash(Image.fromarray(np.ascontiguousarray(arr))).hash.flatten()
    return np.packbits(bits).view(">u8")[0].astype(np.uint64)


def compute_features(keys: list[str], paths: list[Path], cache: Path, desc: str = "хеши") -> dict:
    """Признаки для списка файлов. keys — идентификаторы строк (сохраняются в кэш как paths)."""
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["paths"]) == list(keys):
            print(f"кэш: {cache}")
            return {k: z[k] for k in z.files}
    n = len(paths)
    md5 = np.empty(n, dtype="U32")
    shape = np.zeros((n, 2), np.int32)
    hashes = np.zeros((n, 8), np.uint64)
    thumbs = np.zeros((n, 64, 64), np.uint8)
    for i, p in enumerate(tqdm(paths, desc=desc)):
        md5[i] = hashlib.md5(Path(p).read_bytes()).hexdigest()
        g = read_image(p, gray=True)
        shape[i] = g.shape
        t128 = cv2.resize(crop_black(g), (128, 128), interpolation=cv2.INTER_AREA)
        hashes[i] = [phash_u64(tf(t128)) for tf in TFS]
        thumbs[i] = cv2.resize(t128, (64, 64), interpolation=cv2.INTER_AREA)
    feats = {"paths": np.array(keys, dtype=str), "md5": md5, "shape": shape, "hashes": hashes, "thumbs": thumbs}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **feats)
    return feats


def candidate_pairs(base: np.ndarray, hashes: np.ndarray, cand_max: int, symmetric: bool,
                    chunk: int = 128) -> tuple[pd.DataFrame, np.ndarray]:
    """Пары (i, j) с min_tf hamming(base[i], hashes[j, tf]) <= cand_max.

    base — хеши набора A без преобразования (n,), hashes — хеши набора B для 8 преобразований (m, 8);
    tf в результате — преобразование, приводящее снимок j к снимку i.
    symmetric=True — A и B один и тот же набор: пары i<j, сам с собой не сравнивается.
    Возвращает пары и расстояние до ближайшего соседа для каждого i.
    """
    n, m = len(base), len(hashes)
    nn = np.full(n, 64, np.int16)
    out = []
    for s in tqdm(range(0, n, chunk), desc="пары"):
        e = min(s + chunk, n)
        d_all = np.bitwise_count(base[s:e, None, None] ^ hashes[None, :, :])  # (c, m, 8)
        tf = d_all.argmin(2)
        d = np.take_along_axis(d_all, tf[..., None], 2)[..., 0].astype(np.int16)
        keep = d <= cand_max
        if symmetric:
            d[np.arange(e - s), np.arange(s, e)] = 64  # себя не считаем
            keep = (d <= cand_max) & (np.arange(m)[None, :] > np.arange(s, e)[:, None])
        nn[s:e] = d.min(1)
        ii, jj = np.nonzero(keep)
        out.append(pd.DataFrame({"i": (ii + s).astype(np.int32), "j": jj.astype(np.int32),
                                 "phash_dist": d[ii, jj].astype(np.int8), "tf": tf[ii, jj].astype(np.int8)}))
    return pd.concat(out, ignore_index=True), nn


def _znorm(thumbs: np.ndarray) -> np.ndarray:
    z = thumbs.reshape(len(thumbs), -1).astype(np.float32)
    z -= z.mean(1, keepdims=True)
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    z /= np.where(norm > 0, norm, 1)
    return z.reshape(thumbs.shape)  # нормировка инвариантна к перестановке пикселей -> TF можно применять после


def pair_corr(thumbs_a: np.ndarray, thumbs_b: np.ndarray, pairs: pd.DataFrame, batch: int = 10000) -> np.ndarray:
    """Корреляция Пирсона thumbs_a[i] и TF(thumbs_b[j]) для всех пар, векторно."""
    za = _znorm(thumbs_a)
    zb = za if thumbs_b is thumbs_a else _znorm(thumbs_b)
    corr = np.empty(len(pairs), np.float32)
    ii, jj, tt = (pairs[c].to_numpy() for c in ("i", "j", "tf"))
    for t in range(len(TFS)):
        sel = np.flatnonzero(tt == t)
        for s in range(0, len(sel), batch):
            k = sel[s : s + batch]
            corr[k] = (za[ii[k]] * TFS_BATCH[t](zb[jj[k]])).sum(axis=(1, 2))
    return corr


class DSU:
    """Система непересекающихся множеств; корень группы — минимальный индекс."""

    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)

    def labels(self, prefix: str) -> np.ndarray:
        """Стабильные id групп: по порядку первого вхождения."""
        roots = np.array([self.find(k) for k in range(len(self.p))])
        _, first = np.unique(roots, return_index=True)
        order = {r: g for g, r in enumerate(roots[np.sort(first)])}
        return np.array([f"{prefix}{order[r]:05d}" for r in roots])
