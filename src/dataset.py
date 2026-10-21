"""Данные для обучения и оценки.

- `DataConfig` — настройки данных (раздел `data` в configs/*.yaml).
- Кэш препроцессинга: препроцессинг детерминирован и занимает ≈ 32 мс на снимок, поэтому
  считается один раз (`build_cache`) и хранится массивом uint8 (N, size, size) в .npy;
  `PreprocCache` читает его через memmap.
- `build_transforms` — аугментации на лету: базовые (отражение по горизонтали, поворот до 15°,
  небольшой сдвиг и масштаб, яркость/контраст, шум, размытие) и по флагу «рандомизация домена»
  (уменьшение-увеличение, сжатие JPEG, размытие/резкость, гамма, постеризация, шум).
  Вертикального отражения и поворотов на 90° нет.
- `MRIDataset` — снимок (и маска, если есть) → тензор 3×S×S с нормализацией ImageNet.
- `load_split`, `class_weights`, `make_loader`, `seed_everything`.

Источник правды — data/processed/splits.csv: в работу идут только строки primary == True.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, fields
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.io_utils import read_image
from src import counterfactual
from src.labels import CLASS_TO_IDX, CLASSES
from src.preprocess import mask_to_model, preprocess

CACHE_VERSION = "preprocess-1.5"  # менять при изменении src/preprocess.py — старый кэш станет недействительным
_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)  # ImageNet: энкодеры предобучены на нем
_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
GEOMETRY = ["orig_h", "orig_w", "y0", "y1", "x0", "x1", "new_h", "new_w", "top", "left", "scale"]


@dataclass
class DataConfig:
    data_root: str = "mrt"
    processed_dir: str = "data/processed"
    cache_dir: str = "data/processed/cache"
    size: int = 256
    batch_size: int = 32
    num_workers: int = 2
    domain_randomization: bool = True
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "DataConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")).get("data", {})
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"неизвестные ключи в разделе data: {sorted(unknown)}")
        raw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**raw)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------- кэш препроцессинга


def _cache_files(cache_dir: str | Path, name: str, size: int) -> dict[str, Path]:
    base = Path(cache_dir) / f"{name}_{size}"
    return {"images": Path(f"{base}.npy"), "masks": Path(f"{base}_mask.npy"), "table": Path(f"{base}.csv"),
            "info": Path(f"{base}.json")}


def build_cache(table: pd.DataFrame, image_root: str | Path, cache_dir: str | Path, name: str, size: int = 256,
                mask_root: str | Path | None = None, roots: dict[str, str | Path] | None = None) -> dict[str, Path]:
    """Считает препроцессинг для строк table (колонка `path`, необязательная `mask` — маска в
    координатах исходного снимка; пустое значение = пустая маска) и сохраняет кэш `name`.

    Если снимки лежат в разных папках, в table есть колонка `root`, а roots задает папку для
    каждого ее значения (снимок и его маска берутся из одной папки); image_root и mask_root
    тогда не используются. Значения `path` должны быть уникальны по всей таблице."""
    files = _cache_files(cache_dir, name, size)
    files["images"].parent.mkdir(parents=True, exist_ok=True)
    table = table.reset_index(drop=True)
    if table["path"].duplicated().any():
        raise ValueError("в таблице кэша повторяются значения path")
    has_masks = "mask" in table.columns
    by_root = roots is not None and "root" in table.columns
    images = np.lib.format.open_memmap(files["images"], mode="w+", dtype=np.uint8, shape=(len(table), size, size))
    masks = (np.lib.format.open_memmap(files["masks"], mode="w+", dtype=np.uint8, shape=(len(table), size, size))
             if has_masks else None)
    geometry = []
    for i, row in enumerate(tqdm(table.itertuples(index=False), total=len(table), desc=f"кэш {name}")):
        img_dir = Path(roots[row.root]) if by_root else Path(image_root)
        img = read_image(img_dir / row.path)
        images[i], meta = preprocess(img, size)
        if has_masks:
            if isinstance(row.mask, str) and row.mask:
                mask_dir = img_dir if by_root else Path(mask_root)
                masks[i] = mask_to_model(read_image(mask_dir / row.mask), meta)
            else:
                masks[i] = 0
        geometry.append([getattr(meta, k) for k in GEOMETRY])
    images.flush()
    if has_masks:
        masks.flush()
    pd.concat([table, pd.DataFrame(geometry, columns=GEOMETRY)], axis=1).to_csv(files["table"], index=False, encoding="utf-8")
    files["info"].write_text(json.dumps({"version": CACHE_VERSION, "size": size, "n": len(table), "has_masks": has_masks}),
                             encoding="utf-8")
    return files


class PreprocCache:
    """Чтение кэша препроцессинга. Массивы открываются лениво — у каждого процесса-загрузчика свой memmap."""

    def __init__(self, cache_dir: str | Path, name: str, size: int = 256):
        self.files = _cache_files(cache_dir, name, size)
        if not self.files["info"].exists():
            raise FileNotFoundError(f"кэш не найден: {self.files['info']} — запустите python -m src.build_cache")
        info = json.loads(self.files["info"].read_text(encoding="utf-8"))
        if info["version"] != CACHE_VERSION or info["size"] != size:
            raise ValueError(f"кэш устарел ({info}), нужен {CACHE_VERSION}, size={size} — пересоберите его")
        self.has_masks = info["has_masks"]
        self.table = pd.read_csv(self.files["table"])
        self._pos = {p: i for i, p in enumerate(self.table["path"])}
        self._images = self._masks = None

    def __getstate__(self):  # memmap не передаем в процессы-загрузчики
        state = self.__dict__.copy()
        state["_images"] = state["_masks"] = None
        return state

    def __contains__(self, path: str) -> bool:
        return path in self._pos

    def image(self, path: str) -> np.ndarray:
        if self._images is None:
            self._images = np.load(self.files["images"], mmap_mode="r")
        return np.array(self._images[self._pos[path]])

    def mask(self, path: str) -> np.ndarray:
        if self._masks is None:
            self._masks = np.load(self.files["masks"], mmap_mode="r")
        return np.array(self._masks[self._pos[path]])


# ---------------------------------------------------------------- аугментации


def build_transforms(train: bool, domain_randomization: bool, seed: int) -> A.Compose | None:
    """Аугментации для обучения; для оценки — None (снимок подается как есть после препроцессинга)."""
    if not train:
        return None
    geometric = [
        A.HorizontalFlip(p=0.5),  # сторона в данных ненадежна; вертикального отражения нет
        A.Affine(rotate=(-15, 15), scale=(0.92, 1.08), translate_percent=(-0.04, 0.04), interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.8),
    ]
    # рандомизация домена (умеренная): стирает технические различия источников — разрешение, сжатие,
    # резкость, тональную кривую, число уровней серого
    domain = [
        A.Downscale(scale_range=(0.45, 0.9), interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR}, p=0.3),
        A.ImageCompression(quality_range=(35, 90), p=0.3),
        A.OneOf([A.GaussianBlur(sigma_limit=(0.5, 1.5)), A.Sharpen(alpha=(0.1, 0.4), lightness=(0.8, 1.2))], p=0.3),
        A.RandomGamma(gamma_limit=(70, 140), p=0.3),
        A.Posterize(num_bits=(5, 7), p=0.15),
    ] if domain_randomization else []
    photometric = [
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussNoise(std_range=(0.01, 0.04), p=0.2),
        A.GaussianBlur(sigma_limit=(0.3, 1.0), p=0.2),
    ]
    return A.Compose(geometric + domain + photometric, seed=seed)


# ---------------------------------------------------------------- датасет


def tumor_aware_crop(image: np.ndarray, mask: np.ndarray, size: int, empty_fraction: float,
                     rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Случайный фрагмент size×size без изменения масштаба (вариант обучения сегментатора на фрагментах).

    С вероятностью empty_fraction берется фрагмент без опухоли, не менее чем на 60% заполненный тканью;
    иначе — фрагмент, в который попала заметная часть опухоли. Если за 30 попыток подходящий фрагмент
    не найден, фрагмент с опухолью центрируется на ней, а «пустой» берется случайным."""
    rng = rng or random
    h, w = image.shape
    total = int(mask.sum())
    want_empty = total == 0 or rng.random() < empty_fraction
    need = min(0.5 * total, 50)
    y = x = 0
    for _ in range(30):
        y, x = rng.randrange(0, h - size + 1), rng.randrange(0, w - size + 1)
        inside = int(mask[y : y + size, x : x + size].sum())
        if want_empty and inside == 0 and (image[y : y + size, x : x + size] > 20).mean() >= 0.6:
            break
        if not want_empty and inside >= need:
            break
    else:
        if not want_empty:
            ys, xs = np.nonzero(mask)
            y = int(np.clip(ys.mean() - size / 2, 0, h - size))
            x = int(np.clip(xs.mean() - size / 2, 0, w - size))
    return image[y : y + size, x : x + size], mask[y : y + size, x : x + size]


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """uint8 S×S → float32 3×S×S (серый канал повторен трижды, нормализация ImageNet)."""
    x = torch.from_numpy(np.ascontiguousarray(image)).float().div_(255.0)
    return (x.unsqueeze(0).expand(3, -1, -1) - _MEAN) / _STD


class MRIDataset(Dataset):
    """Элемент — словарь: image (float32 3×S×S), label (int), path (str) и, если with_mask, mask
    (float32 1×S×S) и has_mask (bool: маска известна; у снимков без маски mask нулевая).

    table: колонки `path`, `cls`, для сегментации `mask` и необязательная `has_mask`. Снимки берутся
    из кэша, а если кэша нет или в нем нет снимка — препроцессинг считается на лету из исходного файла.
    """

    def __init__(self, table: pd.DataFrame, cfg: DataConfig, transform: A.Compose | None = None,
                 cache: PreprocCache | None = None, with_mask: bool = False, image_root: str | Path | None = None,
                 mask_root: str | Path | None = None, crop: dict | None = None, counterfactual: dict | None = None):
        self.table = table.reset_index(drop=True)
        self.paths = self.table["path"].tolist()
        self.labels = [CLASS_TO_IDX[c] for c in self.table["cls"]]
        self.masks = self.table["mask"].tolist() if with_mask and "mask" in self.table else None
        # has_mask: маска известна (иначе нулевая маска — заглушка, и ошибку сегментации по ней считать нельзя)
        self.has_mask = self.table["has_mask"].astype(bool).tolist() if "has_mask" in self.table else None
        self.cfg, self.transform, self.cache, self.with_mask = cfg, transform, cache, with_mask
        self.image_root = Path(image_root if image_root is not None else cfg.data_root)
        self.mask_root = Path(mask_root) if mask_root is not None else None
        # crop: {"size": 128, "empty_fraction": 0.3} — обучение на фрагментах (только с масками)
        self.crop = crop if crop and with_mask else None
        # counterfactual: {"p_remove", "p_control", "dilate_px"} — контрфактические примеры (src/counterfactual.py), только с кэшем
        self.counterfactual = counterfactual
        if counterfactual and cache is None:
            raise ValueError("контрфактические примеры требуют кэша препроцессинга с масками")

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, i: int) -> tuple[np.ndarray, np.ndarray | None]:
        path = self.paths[i]
        if self.cache is not None and path in self.cache:
            return self.cache.image(path), (self.cache.mask(path) if self.with_mask or self.counterfactual else None)
        image, meta = preprocess(read_image(self.image_root / path), self.cfg.size)
        mask = None
        if self.with_mask:
            m = self.masks[i] if self.masks is not None else None
            mask = (mask_to_model(read_image(self.mask_root / m), meta) if isinstance(m, str) and m
                    else np.zeros((self.cfg.size, self.cfg.size), np.uint8))
        return image, mask

    def __getitem__(self, i: int) -> dict:
        image, mask = self._load(i)
        label = self.labels[i]
        if self.counterfactual:
            image, mask, label, _ = counterfactual.apply(image, mask, label, self.counterfactual)
            if not self.with_mask:
                mask = None
        if self.transform is not None:
            out = self.transform(image=image, mask=mask) if mask is not None else self.transform(image=image)
            image, mask = out["image"], out.get("mask")
        if self.crop is not None:
            image, mask = tumor_aware_crop(image, mask, self.crop["size"], self.crop["empty_fraction"])
        item = {"image": to_tensor(image), "label": label, "path": self.paths[i]}
        if self.with_mask:
            item["mask"] = torch.from_numpy(np.ascontiguousarray(mask > 0)).float().unsqueeze(0)
            item["has_mask"] = True if self.has_mask is None else self.has_mask[i]
        return item


def load_split(processed_dir: str | Path, split: str) -> pd.DataFrame:
    """Рабочие снимки части split (train / val / test) из splits.csv."""
    s = pd.read_csv(Path(processed_dir) / "splits.csv")
    d = s[s["primary"] & (s["split"] == split)].reset_index(drop=True)
    if d.empty:
        raise ValueError(f"в splits.csv нет рабочих снимков для split={split!r}")
    return d


def class_weights(table: pd.DataFrame, p_remove: float = 0.0) -> torch.Tensor:
    """Веса классов для функции потерь: N / (K · n_c), порядок как в CLASSES.

    p_remove — доля снимков с маской, которые при обучении превращаются в «нет опухоли» (контрфактические
    примеры): веса считаются по ожидаемому составу эпохи."""
    counts = table["cls"].value_counts().reindex(CLASSES).fillna(0).to_numpy(dtype=np.float64)
    if p_remove > 0:
        moved = table.loc[table["has_mask"].astype(bool), "cls"].value_counts().reindex(CLASSES).fillna(0).to_numpy(dtype=np.float64) * p_remove
        counts = counts - moved
        counts[CLASSES.index("no_tumor")] += moved.sum()
    if (counts == 0).any():
        raise ValueError(f"в таблице нет снимков класса: {[c for c, n in zip(CLASSES, counts) if n == 0]}")
    return torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32)


def _worker_init(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    seed = info.seed % 2**32  # зависит от seed генератора загрузчика, номера эпохи и номера процесса
    random.seed(seed)
    np.random.seed(seed)
    if info.dataset.transform is not None:
        info.dataset.transform.set_random_seed(seed)


def make_loader(table: pd.DataFrame, cfg: DataConfig, train: bool, cache: PreprocCache | None = None,
                with_mask: bool = False, image_root: str | Path | None = None,
                mask_root: str | Path | None = None, crop: dict | None = None, counterfactual: dict | None = None) -> DataLoader:
    """cfg.seed задает порядок батчей и случайность аугментаций (seed обучения); деление данных от него не зависит."""
    dataset = MRIDataset(table, cfg, build_transforms(train, cfg.domain_randomization, cfg.seed), cache, with_mask,
                         image_root, mask_root, crop if train else None, counterfactual if train else None)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=train, drop_last=train, num_workers=cfg.num_workers,
                      worker_init_fn=_worker_init, generator=generator, pin_memory=torch.cuda.is_available(),
                      persistent_workers=cfg.num_workers > 0)
