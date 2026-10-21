"""Преобразования снимков для проверок против shortcut (определения — configs/eval_definitions.yaml).

  to_no_tumor_profile / to_tumor_profile — стресс-тест: привести файл к техническому профилю другого класса;
  occlude / control_region               — закрашивание опухоли и контрольного участка вне опухоли.
Здесь только операции над снимками; прогон моделей и подсчет метрик — на этапе 6.

Подбор параметров стресс-теста и проверка осуществимости контроля — только по обучающей части:
    python -m src.stress --calibrate
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.io_utils import read_image
from src.preprocess import preprocess

SEED = 42
DEFS = Path("configs/eval_definitions.yaml")


def _long_side(img: np.ndarray, target: int, interp: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = target / max(h, w)
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)


def _jpeg(img: np.ndarray, quality: int) -> tuple[np.ndarray, int]:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE), len(buf)


def to_no_tumor_profile(gray: np.ndarray, p: dict) -> np.ndarray:
    """Файл опухоли → «маленький резкий сжатый файл»: уменьшение, нерезкое маскирование, JPEG."""
    x = _long_side(gray, p["long_side"], cv2.INTER_AREA).astype(np.float32)
    if p["unsharp_amount"] > 0:
        x = x + p["unsharp_amount"] * (x - cv2.GaussianBlur(x, (0, 0), p["unsharp_sigma"]))
    return _jpeg(np.clip(np.round(x), 0, 255).astype(np.uint8), p["jpeg_quality"])[0]


def to_tumor_profile(gray: np.ndarray, p: dict) -> np.ndarray:
    """Файл «нет опухоли» → «большой мягкий файл»: увеличение и размытие, без повторного сжатия."""
    x = _long_side(gray, p["long_side"], cv2.INTER_CUBIC).astype(np.float32)
    if p["blur_sigma"] > 0:
        x = cv2.GaussianBlur(x, (0, 0), p["blur_sigma"])
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def occlude(img: np.ndarray, region: np.ndarray, p: dict) -> np.ndarray:
    """Закрашивает область region (bool) по окружающим тканям."""
    return cv2.inpaint(img, region.astype(np.uint8), p["inpaint_radius"], cv2.INPAINT_TELEA)


def tumor_region(mask: np.ndarray, p: dict) -> np.ndarray:
    return _dilate(mask > 0, p["dilate_px"])


def control_region(img: np.ndarray, mask: np.ndarray, p: dict) -> tuple[np.ndarray | None, str]:
    """Участок той же формы вне опухоли: отражение или поворот области опухоли относительно центра рамки головы.
    Возвращает (область, способ) или (None, "нет")."""
    c = p["control"]
    region = tumor_region(mask, p)
    tissue = img > c["tissue_threshold"]
    ys, xs = np.nonzero(tissue)
    if ys.size == 0:
        return None, "нет"
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    forbidden = _dilate(mask > 0, c["no_overlap_dilate_px"])
    ry, rx = np.nonzero(region)
    h, w = img.shape
    for name in c["candidates"]:
        ny = ry if name == "mirror_horizontal" else np.round(2 * cy - ry).astype(int)
        nx = rx if name == "mirror_vertical" else np.round(2 * cx - rx).astype(int)
        ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        if ok.mean() < 1.0:
            continue
        cand = np.zeros_like(region)
        cand[ny, nx] = True
        if (cand & forbidden).any() or tissue[cand].mean() < c["min_tissue_fraction"]:
            continue
        return cand, name
    return None, "нет"


# ---------------------------------------------------------------- подбор параметров (только train)


def _sharp(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calibrate(processed: Path, roots: dict, cache_dir: Path, report_dir: Path, n_tumor: int = 200) -> None:
    d = load_config(DEFS)
    pool = pd.read_csv(processed / "pool.csv", keep_default_na=False)
    train = pool[pool["split"] == "train"]
    nt = train[train["cls"] == "no_tumor"]
    tu = train[train["cls"] != "no_tumor"].sample(n_tumor, random_state=SEED)
    load = lambda r: read_image(Path(roots[r.root]) / r.path)

    nt_imgs = [load(r) for r in nt.itertuples()]
    tu_imgs = [load(r) for r in tu.itertuples()]
    nt_files = [Path(roots[r.root]) / r.path for r in nt.itertuples()]
    target = {
        "long_nt": float(np.median([max(i.shape) for i in nt_imgs])), "sharp_nt": float(np.median([_sharp(i) for i in nt_imgs])),
        "bpp_nt": float(np.median([f.stat().st_size / (i.shape[0] * i.shape[1]) for f, i in zip(nt_files, nt_imgs)])),
        "long_t": float(np.median([max(i.shape) for i in tu_imgs])), "sharp_t": float(np.median([_sharp(i) for i in tu_imgs])),
    }
    print(f"профили train: «нет опухоли» ({len(nt_imgs)} файлов) — длинная сторона {target['long_nt']:.0f}, резкость {target['sharp_nt']:.0f}, "
          f"байт на пиксель {target['bpp_nt']:.3f}; опухоли ({len(tu_imgs)} файлов) — длинная сторона {target['long_t']:.0f}, резкость {target['sharp_t']:.0f}")

    pa = dict(d["stress_test"]["tumor_to_no_tumor_profile"])
    print("\nопухоли → профиль «нет опухоли»: резкость после преобразования (цель", round(target["sharp_nt"]), ")")
    for amount in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        q = dict(pa, unsharp_amount=amount, jpeg_quality=100)
        print(f"  unsharp_amount {amount}: резкость {np.median([_sharp(to_no_tumor_profile(i, q)) for i in tu_imgs]):.0f}")
    print("  байт на пиксель после JPEG (цель", round(target["bpp_nt"], 3), "), при unsharp_amount из конфига:")
    for quality in (60, 75, 85, 90, 95):
        x = [_long_side(i, pa["long_side"], cv2.INTER_AREA).astype(np.float32) for i in tu_imgs]
        x = [np.clip(np.round(v + pa["unsharp_amount"] * (v - cv2.GaussianBlur(v, (0, 0), pa["unsharp_sigma"]))), 0, 255).astype(np.uint8) for v in x]
        print(f"  jpeg_quality {quality}: {np.median([_jpeg(v, quality)[1] / v.size for v in x]):.3f}")
    final = [to_no_tumor_profile(i, pa) for i in tu_imgs]
    print(f"  с параметрами из конфига: длинная сторона {np.median([max(i.shape) for i in final]):.0f}, резкость {np.median([_sharp(i) for i in final]):.0f}")

    pb = dict(d["stress_test"]["no_tumor_to_tumor_profile"])
    print("\n«нет опухоли» → профиль опухолей: резкость после преобразования (цель", round(target["sharp_t"]), ")")
    for sigma in (0.0, 0.5, 1.0, 1.5, 2.0):
        print(f"  blur_sigma {sigma}: резкость {np.median([_sharp(to_tumor_profile(i, dict(pb, blur_sigma=sigma))) for i in nt_imgs]):.0f}")

    # контрольный участок: для скольких снимков train с маской он находится
    oc = d["occlusion"]
    tab = pd.read_csv(cache_dir / "pool_256.csv", keep_default_na=False)
    imgs = np.load(cache_dir / "pool_256.npy", mmap_mode="r")
    masks = np.load(cache_dir / "pool_256_mask.npy", mmap_mode="r")
    idx = np.flatnonzero((tab["split"] == "train") & (tab["has_mask"].astype(str).str.lower() == "true"))
    how = pd.Series([control_region(np.asarray(imgs[i]), np.asarray(masks[i]), oc)[1] for i in idx])
    by_cls = pd.crosstab(tab.loc[idx, "cls"].to_numpy(), how.to_numpy())
    print(f"\nконтрольный участок (train, {len(idx)} снимков с маской): найден у {(how != 'нет').mean():.1%}")
    print(by_cls.to_string())

    # примеры для просмотра
    report_dir.mkdir(parents=True, exist_ok=True)
    pick_t = tu.head(3)
    pick_n = nt.sample(3, random_state=SEED)
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.4), dpi=100, facecolor="white")
    for k, r in enumerate(pick_t.itertuples()):
        a, b = preprocess(load(r))[0], preprocess(to_no_tumor_profile(load(r), pa))[0]
        for ax, im, t in ((axes[0, 2 * k], a, f"опухоль, как есть\n{Path(r.path).name}"), (axes[0, 2 * k + 1], b, "→ профиль «нет опухоли»")):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255); ax.set_axis_off(); ax.set_title(t, fontsize=8)
    for k, r in enumerate(pick_n.itertuples()):
        a, b = preprocess(load(r))[0], preprocess(to_tumor_profile(load(r), pb))[0]
        for ax, im, t in ((axes[1, 2 * k], a, f"нет опухоли, как есть\n{Path(r.path).name}"), (axes[1, 2 * k + 1], b, "→ профиль опухолей")):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255); ax.set_axis_off(); ax.set_title(t, fontsize=8)
    fig.suptitle("Стресс-тест: снимок после препроцессинга до и после приведения к профилю другого класса (train)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(report_dir / "stress_examples.png", facecolor="white")
    plt.close(fig)

    rng = np.random.default_rng(SEED)
    sel = rng.choice(idx[how.to_numpy() != "нет"], 6, replace=False)
    fig, axes = plt.subplots(3, 6, figsize=(15, 7.9), dpi=100, facecolor="white")
    for k, i in enumerate(sel):
        img, mask = np.asarray(imgs[i]), np.asarray(masks[i])
        reg = tumor_region(mask, oc)
        ctrl, name = control_region(img, mask, oc)
        for ax, im, cont, t in ((axes[0, k], img, [(reg, "#eb6834"), (ctrl, "#2a78d6")], f"{tab.loc[i, 'cls']}: опухоль и контроль\n({name})"),
                                (axes[1, k], occlude(img, reg, oc), [], "опухоль закрашена"), (axes[2, k], occlude(img, ctrl, oc), [], "закрашен контрольный участок")):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            for m, color in cont:
                ax.contour(m, levels=[0.5], colors=[color], linewidths=1.1)
            ax.set_axis_off(); ax.set_title(t, fontsize=8)
    fig.suptitle("Закрашивание: оранжевый контур — область опухоли, синий — контрольный участок той же формы (train)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(report_dir / "occlusion_examples.png", facecolor="white")
    plt.close(fig)
    print(f"\nпримеры: {report_dir / 'stress_examples.png'}, {report_dir / 'occlusion_examples.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/processed/cache"))
    ap.add_argument("--data-root", type=Path, default=Path("mrt"))
    ap.add_argument("--fig-root", type=Path, default=Path("data/external/figshare/png"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports/stress"))
    args = ap.parse_args()
    if args.calibrate:
        calibrate(args.processed, {"figshare": args.fig_root, "case": args.data_root}, args.cache_dir, args.report_dir)


if __name__ == "__main__":
    main()
