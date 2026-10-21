"""Чтение конфигов экспериментов.

Конфиг — YAML. Ключ `_base_` задает родительский файл (путь относительно текущего): его
значения берутся по умолчанию, значения текущего файла их переопределяют (вложенные
словари объединяются).
"""
from __future__ import annotations

from pathlib import Path

import yaml


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("_base_", None)
    if base is not None:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    return cfg
