"""Конвертация Figshare (.mat v7.3) в PNG + metadata.csv.

Для каждого <id>.mat:
  images/<id>.png — снимок, 8 бит, min-max нормализация (как в README датасета);
  masks/<id>.png  — маска опухоли, 0/255.
Размер исходный (512×512, у 15 файлов 256×256). h5py отдает массивы MATLAB
транспонированными, поэтому снимок и маска разворачиваются (.T).

Пример:
    python -m src.convert_figshare --src data/external/figshare/mat --out data/external/figshare/png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

SEED = 42
LABEL_TO_CLS = {1: "meningioma", 2: "glioma", 3: "pituitary"}
CLS_RU = {"meningioma": "менингиома", "glioma": "глиома", "pituitary": "гипофиз"}
C_ORANGE = "#eb6834"


def read_mat(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        cj = f["cjdata"]
        return {
            "image": np.array(cj["image"]).T,
            "mask": np.array(cj["tumorMask"]).T,
            "label": int(cj["label"][0, 0]),
            "pid": "".join(chr(int(c)) for c in np.array(cj["PID"]).ravel()),
        }


def to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float64)
    lo, hi = img.min(), img.max()
    if hi <= lo:
        return np.zeros(img.shape, np.uint8)
    return np.round(255.0 * (img - lo) / (hi - lo)).astype(np.uint8)


def write_png(path: Path, arr: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError(f"Не удалось закодировать PNG: {path}")
    buf.tofile(str(path))  # tofile вместо imwrite: работает с любыми путями на Windows


def convert(src: Path, out: Path) -> pd.DataFrame:
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    files = sorted(src.glob("*.mat"), key=lambda p: int(p.stem))
    for p in tqdm(files, desc="figshare → png"):
        d = read_mat(p)
        img, mask = d["image"], (d["mask"] > 0)
        if img.shape != mask.shape:
            raise ValueError(f"{p.name}: размеры снимка и маски не совпадают")
        write_png(out / "images" / f"{p.stem}.png", to_uint8(img))
        write_png(out / "masks" / f"{p.stem}.png", mask.astype(np.uint8) * 255)
        ys, xs = np.nonzero(mask)
        rows.append({
            "id": int(p.stem),
            "image": f"images/{p.stem}.png",
            "mask": f"masks/{p.stem}.png",
            "label": d["label"],
            "cls": LABEL_TO_CLS[d["label"]],
            "pid": d["pid"],
            # MR040240, MR040240B, MR040240C — вероятно, повторные исследования одного пациента;
            # для группировки при сплите считаем их одним пациентом (осторожный вариант)
            "pid_base": re.sub(r"(?<=\d)[A-Z]$", "", d["pid"]),
            "h": img.shape[0],
            "w": img.shape[1],
            "raw_min": int(img.min()),
            "raw_max": int(img.max()),
            "mask_px": int(mask.sum()),
            "mask_frac": round(float(mask.mean()), 5),
            "bbox_x0": int(xs.min()) if xs.size else -1,
            "bbox_y0": int(ys.min()) if ys.size else -1,
            "bbox_x1": int(xs.max()) if xs.size else -1,
            "bbox_y1": int(ys.max()) if ys.size else -1,
        })
    return pd.DataFrame(rows)


def plot_samples(meta: pd.DataFrame, out: Path, path: Path, per_cls: int = 4) -> None:
    """Контроль ориентации и совмещения: случайные снимки каждого класса с контуром маски."""
    classes = list(LABEL_TO_CLS.values())
    fig, axes = plt.subplots(len(classes), per_cls, figsize=(3.3 * per_cls, 3.75 * len(classes)), dpi=100)
    for r, cls in enumerate(classes):
        sub = meta[meta["cls"] == cls].sample(per_cls, random_state=SEED)
        for c, row in enumerate(sub.itertuples(index=False)):
            img = cv2.imdecode(np.fromfile(str(out / row.image), np.uint8), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imdecode(np.fromfile(str(out / row.mask), np.uint8), cv2.IMREAD_GRAYSCALE)
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.contour(mask > 0, levels=[0.5], colors=[C_ORANGE], linewidths=1.3)
            ax.set_axis_off()
            ax.set_title(f"{CLS_RU[cls]} · {row.id}.mat · PID {row.pid}\n{row.w}×{row.h}, опухоль {100 * row.mask_frac:.1f}% кадра",
                         fontsize=8)
    fig.suptitle("Figshare после конвертации: контроль ориентации и маски (контур — tumorMask)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("data/external/figshare/mat"))
    ap.add_argument("--out", type=Path, default=Path("data/external/figshare/png"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports/masks"))
    ap.add_argument("--plot-only", action="store_true", help="не конвертировать, только перерисовать сетку")
    args = ap.parse_args()

    if args.plot_only:
        meta = pd.read_csv(args.out / "metadata.csv", dtype={"pid": str, "pid_base": str})
    else:
        meta = convert(args.src, args.out)
        meta.to_csv(args.out / "metadata.csv", index=False, encoding="utf-8")
    plot_samples(meta, args.out, args.report_dir / "figshare_samples.png")

    print(f"файлов: {len(meta)}; PID: {meta['pid'].nunique()}; пациентов по базовому номеру: {meta['pid_base'].nunique()}")
    print(meta.groupby("cls").agg(срезов=("id", "size"), пациентов=("pid", "nunique"),
                                  маска_медиана_проц=("mask_frac", lambda s: round(100 * s.median(), 2)),
                                  маска_мин_проц=("mask_frac", lambda s: round(100 * s.min(), 3)),
                                  маска_макс_проц=("mask_frac", lambda s: round(100 * s.max(), 2))).to_string())
    print("размеры:", meta.groupby(["h", "w"]).size().to_dict())
    print("пустых масок:", int((meta["mask_px"] == 0).sum()), "| масок < 100 px:", int((meta["mask_px"] < 100).sum()))
    print(f"metadata: {args.out / 'metadata.csv'}; сетка: {args.report_dir / 'figshare_samples.png'}")


if __name__ == "__main__":
    main()
