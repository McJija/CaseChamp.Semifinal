"""Модели: классификатор, сегментатор и многозадачная сеть на одном и том же энкодере.

kind:
  classifier — timm, 4 класса;
  segmenter  — smp.Unet, бинарная маска опухоли;
  multitask  — smp.Unet + классификационная голова на признаках энкодера (aux_params).
В smp энкодер берется с префиксом `tu-` — это та же сеть timm с теми же предобученными
весами, что и у классификатора.

Двухфазное дообучение: сначала энкодер заморожен (учатся только головы), затем учится вся
сеть, у энкодера скорость обучения ниже.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import timm
import torch
from torch import nn

from src.labels import CLASSES

KINDS = ("classifier", "segmenter", "multitask")


def build_model(kind: str, encoder: str, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    if kind not in KINDS:
        raise ValueError(f"неизвестный вид модели: {kind!r}")
    if kind == "classifier":
        return timm.create_model(encoder, pretrained=pretrained, num_classes=len(CLASSES))
    aux = {"classes": len(CLASSES), "pooling": "avg", "dropout": dropout} if kind == "multitask" else None
    return smp.Unet(f"tu-{encoder}", encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1, aux_params=aux)


def split_params(model: nn.Module, kind: str) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Параметры энкодера и параметры голов (у сегментатора голова — декодер и выходные слои)."""
    if kind == "classifier":
        head_ids = {id(p) for p in model.get_classifier().parameters()}
        enc = [p for p in model.parameters() if id(p) not in head_ids]
        head = [p for p in model.parameters() if id(p) in head_ids]
    else:
        enc = list(model.encoder.parameters())
        enc_ids = {id(p) for p in enc}
        head = [p for p in model.parameters() if id(p) not in enc_ids]
    return enc, head


def set_encoder_frozen(model: nn.Module, kind: str, frozen: bool) -> None:
    enc, _ = split_params(model, kind)
    for p in enc:
        p.requires_grad = not frozen


def set_train_mode(model: nn.Module, kind: str, encoder_frozen: bool) -> None:
    """Режим обучения; у замороженного энкодера слои нормализации остаются в режиме оценки,
    чтобы их накопленные статистики не сдвигались."""
    model.train()
    if not encoder_frozen:
        return
    if kind == "classifier":
        head_modules = set(model.get_classifier().modules())
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and m not in head_modules:
                m.eval()
    else:
        model.encoder.eval()


def forward(model: nn.Module, kind: str, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Единый выход: logits (B×4) и/или mask_logits (B×1×H×W)."""
    if kind == "classifier":
        return {"logits": model(x)}
    out = model(x)
    if kind == "multitask":
        return {"mask_logits": out[0], "logits": out[1]}
    return {"mask_logits": out}
