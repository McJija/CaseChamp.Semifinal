"""Замер кандидатов в модели: число параметров и время прохода на CPU.

Модели строятся без загрузки предобученных весов (на скорость и размер веса не влияют).
Для каждого энкодера: классификатор (timm, 4 класса), сегментатор (smp.Unet, 1 класс)
и многозадачная сеть (smp.Unet + aux_params: маска и 4 класса). Вход 1×3×256×256.

Результат: reports/model_bench.csv.

Пример:
    python -m src.bench_models
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import timm
import torch

SEED = 42
# имя энкодера: (в timm, в smp). Префикс tu- — энкодер timm внутри smp: U-Net и классификатор
# получают одну и ту же сеть и одни и те же предобученные веса.
ENCODERS = {
    "resnet18": ("resnet18", "tu-resnet18"),
    "resnet34": ("resnet34", "tu-resnet34"),
    "efficientnet_b0": ("efficientnet_b0", "tu-efficientnet_b0"),
    "efficientnet_b3": ("efficientnet_b3", "tu-efficientnet_b3"),
    "convnext_tiny": ("convnext_tiny", "tu-convnext_tiny"),
}


def n_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def cpu_ms(model: torch.nn.Module, size: int, runs: int = 40, warmup: int = 5) -> float:
    model.eval()
    x = torch.randn(1, 3, size, size)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path("reports/model_bench.csv"))
    args = ap.parse_args()
    torch.manual_seed(SEED)

    rows = []
    for name, (timm_name, smp_name) in ENCODERS.items():
        builders = {
            "классификатор (timm)": lambda: timm.create_model(timm_name, pretrained=False, num_classes=4),
            "сегментатор (smp.Unet)": lambda: smp.Unet(smp_name, encoder_weights=None, in_channels=3, classes=1),
            "многозадачная (smp.Unet + aux)": lambda: smp.Unet(smp_name, encoder_weights=None, in_channels=3, classes=1,
                                                               aux_params={"classes": 4, "pooling": "avg", "dropout": 0.2}),
        }
        for kind, build in builders.items():
            try:
                model = build()
                out = model.eval()(torch.randn(1, 3, args.size, args.size))
                shapes = [tuple(o.shape) for o in out] if isinstance(out, tuple) else [tuple(out.shape)]
                rows.append({"энкодер": name, "модель": kind, "параметров, млн": round(n_params(model), 1),
                             "CPU, мс": round(cpu_ms(model, args.size), 1), "выход": str(shapes), "ошибка": ""})
            except Exception as e:  # энкодер может не поддерживаться в smp.Unet
                rows.append({"энкодер": name, "модель": kind, "параметров, млн": np.nan, "CPU, мс": np.nan, "выход": "",
                             "ошибка": f"{type(e).__name__}: {str(e)[:90]}"})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"\nпотоков CPU: {torch.get_num_threads()}; вход {args.size}×{args.size}")
    print("тег весов timm по умолчанию:", {t: timm.create_model(t, pretrained=False).pretrained_cfg.get("tag")
                                           for t, _ in ENCODERS.values()})


if __name__ == "__main__":
    main()
