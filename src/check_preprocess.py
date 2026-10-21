"""Проверка препроцессинга. Test не используется.

1. Статистика обрезки и время работы — по рабочим снимкам train и val.
2. Устойчивость к аугментациям организаторов — рамка и результат для копий BR/DA/SP/HF/VF/RO
   сравниваются с оригиналом (полные группы из train).
3. Маски — на снимках Figshare, не связанных с test: не срезает ли обрезка опухоль, точность
   переноса маски в координаты модели и обратно, размер опухоли после препроцессинга.
4. Сетки «до и после»: reports/preprocess/*.png.

Результат: data/processed/preprocess_stats.csv, reports/preprocess_report.md.

Пример:
    python -m src.check_preprocess
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from tqdm import tqdm

from src.io_utils import read_image
from src.preprocess import map_to_original, mask_to_model, preprocess

SEED = 42
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
CLS_RU = {"glioma": "глиома", "meningioma": "менингиома", "pituitary": "гипофиз", "no_tumor": "нет опухоли"}
AUGS = ["BR", "DA", "SP", "HF", "VF", "RO"]
C_ORANGE = "#eb6834"


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float32).ravel(), b.astype(np.float32).ravel()
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


def _rel_box(m) -> np.ndarray:
    return np.array([m.y0 / m.orig_h, m.y1 / m.orig_h, m.x0 / m.orig_w, m.x1 / m.orig_w])


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ih = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    iw = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = ih * iw
    union = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / union if union > 0 else 0.0


def crop_stats(work: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    rows = []
    for r in tqdm(work.itertuples(index=False), total=len(work), desc="рабочие снимки"):
        img = read_image(data_root / r.path)
        t = time.perf_counter()
        out, m = preprocess(img)
        ms = (time.perf_counter() - t) * 1000
        content = out[m.top : m.top + m.new_h, m.left : m.left + m.new_w]
        rows.append({
            "path": r.path, "cls": r.cls, "split": r.split, "orig_h": m.orig_h, "orig_w": m.orig_w,
            "box_frac": (m.y1 - m.y0) * (m.x1 - m.x0) / (m.orig_h * m.orig_w),
            "full_frame": (m.y0, m.x0, m.y1, m.x1) == (0, 0, m.orig_h, m.orig_w),
            "touch_edges": int(m.y0 == 0) + int(m.x0 == 0) + int(m.y1 == m.orig_h) + int(m.x1 == m.orig_w),
            "aspect": (m.x1 - m.x0) / (m.y1 - m.y0), "scale": m.scale, "bg": m.bg, "hi": m.hi,
            "pad_frac": 1 - m.new_h * m.new_w / m.size**2, "out_mean": float(content.mean()),
            "out_levels": int(len(np.unique(content))), "ms": ms,
        })
    return pd.DataFrame(rows)


def aug_robustness(s: pd.DataFrame, data_root: Path, n_groups: int = 100) -> pd.DataFrame:
    g = s[(s["source"] == "aug_GMPN") & (s["split"] == "train")]
    full = g.groupby("base_id")["aug"].nunique()
    bases = pd.Series(sorted(full[full == 7].index)).sample(n_groups, random_state=SEED)
    rows = []
    for base in tqdm(bases, desc="аугментации"):
        x = g[g["base_id"] == base].set_index("aug")["path"]
        o_img, o_meta = preprocess(read_image(data_root / x["orig"]))
        ob = _rel_box(o_meta)
        for aug in AUGS:
            a_img, a_meta = preprocess(read_image(data_root / x[aug]))
            ab = _rel_box(a_meta)
            ref_box, ref_img = ob, o_img
            if aug == "HF":
                ref_box, ref_img = np.array([ob[0], ob[1], 1 - ob[3], 1 - ob[2]]), o_img[:, ::-1]
            if aug == "VF":
                ref_box, ref_img = np.array([1 - ob[1], 1 - ob[0], ob[2], ob[3]]), o_img[::-1, :]
            rows.append({"aug": aug, "box_iou": np.nan if aug == "RO" else _box_iou(ab, ref_box),
                         "corr": _corr(a_img, ref_img)})
    t = pd.DataFrame(rows).groupby("aug").agg(
        групп=("corr", "size"), рамка_IoU_медиана=("box_iou", "median"),
        рамка_IoU_5й_перцентиль=("box_iou", lambda v: v.quantile(0.05)),
        доля_IoU_от_0_9=("box_iou", lambda v: (v >= 0.9).mean() if v.notna().any() else np.nan),
        corr_результата_медиана=("corr", "median"), corr_5й_перцентиль=("corr", lambda v: v.quantile(0.05)))
    return t.reindex(AUGS).round(3)


def mask_checks(processed: Path, fig_root: Path) -> pd.DataFrame:
    meta = pd.read_csv(fig_root / "metadata.csv", dtype={"pid": str, "pid_base": str})
    fm = pd.read_csv(processed / "figshare_matches.csv", usecols=["path", "level", "fig_id"], dtype={"fig_id": str})
    sp = pd.read_csv(processed / "splits.csv", usecols=["path", "split"])
    fm = fm[fm["level"] != "none"].merge(sp, on="path")
    in_test = set(fm.loc[fm["split"] == "test", "fig_id"].astype(int))
    meta = meta[~meta["id"].isin(in_test)]
    rows = []
    for r in tqdm(meta.itertuples(index=False), total=len(meta), desc="маски Figshare"):
        img = read_image(fig_root / r.image)
        mask = read_image(fig_root / r.mask) > 0
        _, m = preprocess(img)
        inside = mask[m.y0 : m.y1, m.x0 : m.x1].sum()
        mm = mask_to_model(mask.astype(np.uint8), m)
        back = map_to_original(mm, m) > 0
        inter = (back & mask).sum()
        rows.append({
            "id": r.id, "cls": r.cls, "mask_px": int(mask.sum()), "lost_by_crop": 1 - inside / mask.sum(),
            "model_px": int(mm.sum()), "model_frac": float(mm.sum() / (m.new_h * m.new_w)),
            "orig_frac": float(mask.mean()), "dice_roundtrip": 2 * inter / (back.sum() + mask.sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- сетки


def plot_pairs(items: list[tuple[str, Path]], title: str, out: Path, ncols: int = 4) -> None:
    """Пары «исходный снимок с рамкой → результат»."""
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(2.3 * ncols * 2, 2.62 * nrows + 0.55), dpi=100, squeeze=False,
                             facecolor="white")
    for ax in axes.ravel():
        ax.set_axis_off()
    for k, (label, path) in enumerate(items):
        img = read_image(path)
        out_img, m = preprocess(img)
        a, b = axes[k // ncols, (k % ncols) * 2], axes[k // ncols, (k % ncols) * 2 + 1]
        a.imshow(img, cmap="gray", vmin=0, vmax=255)
        a.add_patch(Rectangle((m.x0 - 0.5, m.y0 - 0.5), m.x1 - m.x0, m.y1 - m.y0, fill=False, ec=C_ORANGE, lw=1.3))
        a.set_title(f"{label}\n{m.orig_w}×{m.orig_h}, фон {m.bg:.0f}", fontsize=7.5, pad=3)
        b.imshow(out_img, cmap="gray", vmin=0, vmax=255)
        b.set_title(f"→ 256×256, масштаб ×{m.scale:.2f}", fontsize=7.5, pad=3)
    fig.suptitle(title, fontsize=11, fontweight="bold", x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=0.2, h_pad=0.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def plot_masks(checks: pd.DataFrame, fig_root: Path, out: Path, n: int = 8) -> None:
    """Снимок Figshare с маской → результат с маской в координатах модели."""
    pick = pd.concat([checks.nsmallest(n // 2, "model_px"), checks.sample(n - n // 2, random_state=SEED)])
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5.9), dpi=100, facecolor="white")
    for k, r in enumerate(pick.itertuples(index=False)):
        img, mask = read_image(fig_root / "images" / f"{r.id}.png"), read_image(fig_root / "masks" / f"{r.id}.png") > 0
        out_img, m = preprocess(img)
        mm = mask_to_model(mask.astype(np.uint8), m)
        for ax, im, mk, ttl in ((axes[0, k], img, mask, f"{r.id}.mat · {CLS_RU[r.cls]}\nмаска {r.mask_px} px"),
                                (axes[1, k], out_img, mm, f"256×256: маска {r.model_px} px\n({100 * r.model_frac:.2f}% содержимого)")):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.contour(mk > 0, levels=[0.5], colors=[C_ORANGE], linewidths=1.1)
            ax.set_axis_off()
            ax.set_title(ttl, fontsize=7.5, pad=3)
    fig.suptitle(f"Маски после препроцессинга: слева {n // 2} самых мелких опухолей, справа случайные", fontsize=11,
                 fontweight="bold", x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=0.2, h_pad=1.6)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- отчет


def _md_table(df: pd.DataFrame) -> str:
    df = df.reset_index()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("mrt"))
    ap.add_argument("--fig-root", type=Path, default=Path("data/external/figshare/png"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    s = pd.read_csv(args.processed / "splits.csv")
    work = s[s["primary"] & s["split"].isin(["train", "val"])].sort_values("path").reset_index(drop=True)
    st = crop_stats(work, args.data_root)
    st.to_csv(args.processed / "preprocess_stats.csv", index=False, encoding="utf-8")
    aug = aug_robustness(s, args.data_root)
    mk = mask_checks(args.processed, args.fig_root)

    rd = args.report_dir / "preprocess"
    root = args.data_root
    rnd = [(f"{CLS_RU[c]} · {Path(p).name}", root / p) for c in CLASSES
           for p in work[work["cls"] == c].sample(4, random_state=SEED)["path"]]
    plot_pairs(rnd, "Препроцессинг: случайные рабочие снимки (рамка — найденная область головы)", rd / "examples_random.png")
    hard = pd.concat([
        st.nsmallest(4, "box_frac").assign(why="самая маленькая рамка"),
        st[st["full_frame"]].sample(min(4, int(st["full_frame"].sum())), random_state=SEED).assign(why="без обрезки"),
        st.nlargest(4, "bg").assign(why="самый светлый фон"),
        pd.concat([st.nsmallest(2, "aspect"), st.nlargest(2, "aspect")]).assign(why="крайние пропорции"),
    ])
    plot_pairs([(f"{r.why} · {Path(r.path).name}", root / r.path) for r in hard.itertuples()],
               "Препроцессинг: трудные случаи", rd / "examples_hard.png")
    g = s[(s["source"] == "aug_GMPN") & (s["split"] == "train")]
    full = g.groupby("base_id")["aug"].nunique()
    firsts = g[g["base_id"].isin(full[full == 7].index)].drop_duplicates("base_id").sort_values("base_id")
    aug_items = []
    for c in ("no_tumor", "glioma"):
        base = firsts[firsts["cls"] == c].sample(1, random_state=SEED)["base_id"].iloc[0]
        x = g[g["base_id"] == base].set_index("aug")["path"]
        aug_items += [(f"{a} · {Path(x[a]).name}", root / x[a]) for a in ["orig", "BR", "DA", "SP", "RO", "HF", "VF"]]
        aug_items.append(aug_items[-7])  # восьмая ячейка ряда — повтор оригинала для сравнения
    plot_pairs(aug_items, "Препроцессинг аугментированных копий организаторов (две группы)", rd / "examples_augmentations.png")
    plot_masks(mk, args.fig_root, rd / "examples_masks.png")

    by_cls = st.groupby("cls").agg(
        снимков=("path", "size"), рамка_доля_кадра_медиана=("box_frac", "median"), без_обрезки=("full_frame", "sum"),
        масштаб_медиана=("scale", "median"), поля_после_приведения_медиана=("pad_frac", "median"),
        яркость_результата_медиана=("out_mean", "median"), уровней_серого_медиана=("out_levels", "median")).reindex(CLASSES)
    by_cls.index = [CLS_RU[c] for c in by_cls.index]
    mk_cls = mk.groupby("cls").agg(
        масок=("id", "size"), срезано_обрезкой_макс=("lost_by_crop", "max"),
        площадь_до_медиана_проц=("orig_frac", lambda v: 100 * v.median()),
        площадь_после_медиана_проц=("model_frac", lambda v: 100 * v.median()),
        пикселей_после_медиана=("model_px", "median"), пикселей_после_минимум=("model_px", "min"),
        dice_туда_обратно_медиана=("dice_roundtrip", "median"), dice_5й_перцентиль=("dice_roundtrip", lambda v: v.quantile(0.05)))
    mk_cls.index = [CLS_RU[c] for c in mk_cls.index]

    out = ["# Препроцессинг", "",
           "`src/preprocess.py`: серый → срезание однородных полей и рамок-линий → рамка головы относительно уровня фона → "
           "нормализация яркости (фон → 0, 99,5-й перцентиль тканей → 1) → 256×256 с сохранением пропорций → uint8. "
           "Test в проверках не участвует.", "",
           f"## Обрезка и приведение размера (рабочие снимки train и val, {len(st)})", "", _md_table(by_cls.round(3)), "",
           f"- без обрезки (рамка = весь кадр): {int(st['full_frame'].sum())}; рамка касается ≥ 3 краев кадра: "
           f"{int((st['touch_edges'] >= 3).sum())}",
           f"- рамка меньше 25% кадра: {int((st['box_frac'] < 0.25).sum())}; минимум {st['box_frac'].min():.3f}",
           f"- время препроцессинга на CPU (без чтения файла): медиана {st['ms'].median():.1f} мс, "
           f"95-й перцентиль {st['ms'].quantile(0.95):.1f} мс", "",
           "![](preprocess/examples_random.png)", "", "![](preprocess/examples_hard.png)", "",
           "## Устойчивость к аугментациям организаторов (100 полных групп из train)", "",
           "Рамка копии сравнивается с рамкой оригинала (в долях кадра), результат копии — с результатом оригинала. "
           "Для HF и VF оригинал предварительно отражается; для RO рамка не сравнивается (снимок повернут).", "",
           _md_table(aug), "", "![](preprocess/examples_augmentations.png)", "",
           f"## Маски (Figshare, не связанные с test: {len(mk)} снимков)", "", _md_table(mk_cls.round(3)), "",
           f"- масок, задетых обрезкой: {int((mk['lost_by_crop'] > 0).sum())} (потеряно более 1% площади: "
           f"{int((mk['lost_by_crop'] > 0.01).sum())})",
           f"- масок меньше 50 пикселей после препроцессинга: {int((mk['model_px'] < 50).sum())}; меньше 100: "
           f"{int((mk['model_px'] < 100).sum())}",
           f"- Dice «в модель и обратно» ниже 0,9: {int((mk['dice_roundtrip'] < 0.9).sum())}", "",
           "![](preprocess/examples_masks.png)", ""]
    (args.report_dir / "preprocess_report.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(l for l in out if not l.startswith("![")))


if __name__ == "__main__":
    main()
