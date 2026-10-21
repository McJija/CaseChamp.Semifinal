"""Единая точка инференса: снимок → класс, уверенность, маска, тепловая карта, флаги.

Используется и в прототипе, и в оценке. Вход проходит тот же препроцессинг, что при обучении
(src/preprocess.py); маска и тепловая карта возвращаются в координатах исходного снимка.

    from src.inference import load_predictor
    predictor = load_predictor()            # models/final/, а если там пусто — основные запуски (seed 42)
    result = predictor.predict(image)       # image: HxW или HxWx3, uint8 или 16 бит

Пример из командной строки (печатает ответ и сохраняет картинку «снимок | маска | тепловая карта»):
    python -m src.inference path/to/image.jpg --out reports/prototype/example.png
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.dataset import to_tensor
from src.io_utils import read_image
from src.labels import CLASSES, CLS_RU, NO_TUMOR_IDX
from src.models import build_model
from src.preprocess import map_to_original, preprocess

ROOT = Path(__file__).resolve().parents[1]  # пути к весам — от корня проекта, откуда бы ни запускали
MODELS = ROOT / "models"
FINAL = MODELS / "final"
DEFAULT_RUNS = {"classifier": "run1_cls_s42_*", "segmenter": "run1_seg_neg_s42_*"}  # основной seed — 42 (правило 6)
FLAG_ATTENTION = "требует внимания врача"


def cam_layer(model: torch.nn.Module, encoder: str) -> torch.nn.Module:
    """Последний блок энкодера — слой для Grad-CAM (как в configs/eval_definitions.yaml)."""
    return model.layer4[-1] if encoder.startswith("resnet") else model.blocks[-1]


def _load(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ck = torch.load(path, weights_only=False, map_location="cpu")
    model = build_model(ck["kind"], ck["encoder"], pretrained=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


class Predictor:
    """Классификатор + сегментатор основной схемы.

    temperature — температура калибровки. min_mask_area_px — площадь маски в координатах модели,
    начиная с которой область считается найденной. Решение «опухоль есть»:
      presence_threshold задан (и есть сегментатор) — по двум моделям (src/presence.py):
          область найдена ИЛИ p_tumor ≥ presence_threshold;
      иначе tumor_threshold задан — по порогу p_tumor = 1 − p(«нет опухоли»); иначе — по argmax классификатора.
    Тип опухоли — самый вероятный из трех у классификатора. Флаг «требует внимания врача» — модели расходятся:
    классификатор (argmax) и сегментатор (найдена ли область) отвечают по-разному.
    """

    def __init__(self, classifier: Path, segmenter: Path | None = None, device: str = "cpu", temperature: float = 1.0,
                 tumor_threshold: float | None = None, mask_threshold: float = 0.5, min_mask_area_px: int = 50,
                 presence_threshold: float | None = None):
        self.device = torch.device(device)
        self.cls, self.cls_info = _load(classifier, self.device)
        self.seg, self.seg_info = _load(segmenter, self.device) if segmenter else (None, None)
        self.temperature, self.tumor_threshold, self.presence_threshold = temperature, tumor_threshold, presence_threshold
        self.mask_threshold, self.min_mask_area_px = mask_threshold, min_mask_area_px
        self.cam = GradCAM(model=self.cls, target_layers=[cam_layer(self.cls, self.cls_info["encoder"])])

    def predict(self, image: np.ndarray, heatmap: bool = True) -> dict:
        x256, meta = preprocess(image)
        x = to_tensor(x256).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.cls(x).float() / self.temperature, 1)[0].cpu().numpy()
        p_tumor = float(1 - probs[NO_TUMOR_IDX])
        cls_says_tumor = int(probs.argmax()) != NO_TUMOR_IDX
        result = {"p_tumor": p_tumor, "probs": {c: float(p) for c, p in zip(CLASSES, probs)}, "mask": None, "mask_area_px": 0,
                  "mask_area_fraction": 0.0, "heatmap": None, "flags": [], "flag_reason": ""}

        found = None
        if self.seg is not None:
            with torch.no_grad():
                m256 = (torch.sigmoid(self.seg(x).float())[0, 0] > self.mask_threshold).cpu().numpy()
            area = int(m256.sum())
            found = area > self.min_mask_area_px
            result["mask"] = map_to_original(m256.astype(np.uint8), meta, binary=True).astype(bool) if found else np.zeros((meta.orig_h, meta.orig_w), bool)
            result["mask_area_px"] = area if found else 0
            result["mask_area_fraction"] = float(result["mask"].mean())

        if self.presence_threshold is not None and found is not None:
            present = bool(found or p_tumor >= self.presence_threshold)
        elif self.tumor_threshold is not None:
            present = p_tumor >= self.tumor_threshold
        else:
            present = cls_says_tumor
        k = int(probs[:NO_TUMOR_IDX].argmax()) if present else NO_TUMOR_IDX
        result.update({"class": CLASSES[k], "class_ru": CLS_RU[CLASSES[k]], "confidence": float(probs[k])})

        if found is not None and cls_says_tumor != found:  # модели расходятся
            result["flags"].append(FLAG_ATTENTION)
            result["flag_reason"] = ("классификатор не видит опухоли, но сегментатор выделил область" if found
                                     else "классификатор подозревает опухоль, но область не выделена")

        if heatmap:
            target = k if k != NO_TUMOR_IDX else int(probs[:NO_TUMOR_IDX].argmax())  # для «нет опухоли» — карта самого вероятного типа
            cam = self.cam(input_tensor=x, targets=[ClassifierOutputTarget(target)])[0]
            result["heatmap"] = np.clip(map_to_original(cam, meta, binary=False), 0, 1)
            result["heatmap_class"] = CLASSES[target]
        return result


def load_predictor(classifier: Path | None = None, segmenter: Path | None = None, models_dir: Path = MODELS, **kw) -> Predictor:
    """Веса по умолчанию: models/final/{classifier,segmenter}.pt, иначе основные запуски (seed 42) из models/."""
    def pick(kind: str, given: Path | None) -> Path:
        if given:
            return Path(given)
        if (FINAL / f"{kind}.pt").exists():
            return FINAL / f"{kind}.pt"
        runs = sorted(models_dir.glob(DEFAULT_RUNS[kind]))
        if not runs:
            raise FileNotFoundError(f"нет весов: ни {FINAL / (kind + '.pt')}, ни {models_dir / DEFAULT_RUNS[kind]}")
        return runs[-1] / "best.pt"

    # порог решения по двум моделям (src/presence.py) — только для моделей по умолчанию, для которых он подобран
    models_dir = Path(models_dir)
    frozen = (FINAL / "presence.yaml") if models_dir.resolve() == MODELS else (models_dir / "final" / "presence.yaml")
    rule = frozen if frozen.exists() else models_dir.parent / "configs" / "presence.yaml"
    if classifier is None and segmenter is None and "presence_threshold" not in kw and rule.exists():
        cfg = yaml.safe_load(rule.read_text(encoding="utf-8"))
        kw["presence_threshold"] = float(cfg["cls_threshold"])
        kw.setdefault("min_mask_area_px", int(cfg["min_area_px"]))
    return Predictor(pick("classifier", classifier), pick("segmenter", segmenter), **kw)


_default: Predictor | None = None


def predict(image: np.ndarray) -> dict:
    """Ответ модели по умолчанию (загружается при первом вызове)."""
    global _default
    if _default is None:
        _default = load_predictor()
    return _default.predict(image)


def overlay(image: np.ndarray, result: dict, alpha: float = 0.45) -> np.ndarray:
    """Картинка для отчета: снимок | маска поверх снимка | тепловая карта (BGR)."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    with_mask = base.copy()
    if result["mask"] is not None and result["mask"].any():
        with_mask[result["mask"]] = ((1 - alpha) * with_mask[result["mask"]] + alpha * np.array([0, 0, 255])).astype(np.uint8)
        contours, _ = cv2.findContours(result["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(with_mask, contours, -1, (0, 0, 255), 1)
    heat = base.copy()
    if result["heatmap"] is not None:
        color = cv2.applyColorMap((result["heatmap"] * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.addWeighted(base, 1 - alpha, color, alpha, 0)
    return np.hstack([base, with_mask, heat])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("--classifier", type=Path, default=None)
    ap.add_argument("--segmenter", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="куда сохранить картинку «снимок | маска | тепловая карта»")
    args = ap.parse_args()

    predictor = load_predictor(args.classifier, args.segmenter)
    image = read_image(args.image)
    t0 = time.perf_counter()
    r = predictor.predict(image)
    print(f"{args.image.name}: {r['class_ru']} (уверенность {r['confidence']:.3f}; вероятность опухоли {r['p_tumor']:.3f}); "
          f"область {r['mask_area_px']} px ({r['mask_area_fraction']:.2%} кадра); флаги: {r['flags'] or 'нет'}; {time.perf_counter() - t0:.2f} с")
    print("не является медицинским заключением")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".png", overlay(image, r))[1].tofile(str(args.out))
        print("картинка:", args.out)


if __name__ == "__main__":
    main()
