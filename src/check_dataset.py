"""Проверка Dataset и DataLoader. Test не используется.

1. Тензоры: форма, тип, диапазон; веса классов; состав эпохи.
2. Воспроизводимость: одинаковый seed → одинаковые батчи (0 и 2 процесса-загрузчика),
   другой seed и следующая эпоха → другие.
3. Скорость: с кэшем препроцессинга и без него.
4. Маски: снимок и маска преобразуются согласованно; чтение через кэш совпадает с расчетом на лету.
5. Рандомизация домена: насколько технические признаки выдают класс «нет опухоли»
   без аугментаций, с базовыми и с рандомизацией домена (только train).
6. Сетки примеров: reports/dataset/*.png.

Результат: reports/dataset_report.md.

Пример:
    python -m src.check_dataset --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import time
from dataclasses import replace
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from tqdm import tqdm

from src.dataset import (DataConfig, MRIDataset, PreprocCache, build_cache, build_transforms, class_weights, load_split,
                         make_loader, seed_everything)
from src.labels import CLASSES, CLS_RU

SEED = 42
C_ORANGE = "#eb6834"
TECH = {"std": "контраст (СКО яркости)", "sat": "доля пересвеченных пикселей", "sharp": "резкость (лапласиан)",
        "mean": "средняя яркость", "levels": "число уровней серого"}


def batches_digest(cfg: DataConfig, table: pd.DataFrame, cache: PreprocCache, n_batches: int = 3, epochs: int = 1) -> list[str]:
    """Контрольные суммы первых n_batches батчей каждой из epochs эпох."""
    seed_everything(cfg.seed)
    loader = make_loader(table, cfg, train=True, cache=cache)
    out = []
    for _ in range(epochs):
        h = hashlib.sha1()
        for k, b in enumerate(loader):
            h.update(b["image"].numpy().tobytes())
            h.update(b["label"].numpy().tobytes())
            if k + 1 == n_batches:
                break
        out.append(h.hexdigest()[:12])
    del loader
    return out


def tech_features(img: np.ndarray) -> dict:
    x = img.astype(np.float32) / 255
    return {"std": float(x.std()), "sat": float((x > 0.95).mean()), "sharp": float(cv2.Laplacian(img, cv2.CV_64F).var()),
            "mean": float(x.mean()), "levels": int(len(np.unique(img)))}


def shortcut_under_aug(train: pd.DataFrame, cache: PreprocCache) -> pd.DataFrame:
    """AUC «опухоль / нет опухоли» по техническим признакам снимков train при разных аугментациях."""
    y = (train["cls"] != "no_tumor").to_numpy().astype(int)
    groups = train["patient_group_id"].to_numpy()
    variants = {"без аугментаций": None, "базовые": build_transforms(True, False, SEED),
                "базовые + рандомизация домена": build_transforms(True, True, SEED)}
    rows = {}
    for name, tf in variants.items():
        feats = []
        for p in tqdm(train["path"], desc=name):
            img = cache.image(p)
            feats.append(tech_features(tf(image=img)["image"] if tf is not None else img))
        f = pd.DataFrame(feats)
        row = {TECH[c]: max(a, 1 - a) for c in TECH for a in [roc_auc_score(y, f[c])]}
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
        clf = HistGradientBoostingClassifier(max_iter=150, class_weight="balanced", random_state=SEED)
        proba = cross_val_predict(clf, f, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
        row["все признаки вместе (3 фолда по пациентам)"] = roc_auc_score(y, proba)
        rows[name] = row
    return pd.DataFrame(rows).round(3)


def mask_checks(cfg: DataConfig, fig_root: Path, processed: Path, tmp_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Таблица снимков Figshare с масками (не связанных с test) и результаты проверок согласованности."""
    meta = pd.read_csv(fig_root / "metadata.csv")
    fm = pd.read_csv(processed / "figshare_matches.csv", usecols=["path", "level", "fig_id"])
    sp = pd.read_csv(processed / "splits.csv", usecols=["path", "split"])
    fm = fm[fm["level"] != "none"].merge(sp, on="path")
    in_test = set(fm.loc[fm["split"] == "test", "fig_id"].astype(int))
    pool = meta[~meta["id"].isin(in_test)]
    t = pd.concat([pool[pool["cls"] == c].sample(6, random_state=SEED) for c in CLASSES if c != "no_tumor"])
    t = t.rename(columns={"image": "path"})[["path", "mask", "cls"]].reset_index(drop=True)
    res = {}

    # 1) отражение по горизонтали: маска обязана отразиться вместе со снимком
    import albumentations as A
    plain = MRIDataset(t, cfg, None, None, with_mask=True, image_root=fig_root, mask_root=fig_root)
    flip = MRIDataset(t, cfg, A.Compose([A.HorizontalFlip(p=1.0)]), None, with_mask=True, image_root=fig_root, mask_root=fig_root)
    res["flip_ok"] = all(torch.equal(plain[i]["mask"].flip(-1), flip[i]["mask"]) and
                         torch.equal(plain[i]["image"].flip(-1), flip[i]["image"]) for i in range(len(t)))
    # 2) кэш с масками дает то же, что расчет на лету
    build_cache(t, fig_root, tmp_dir, "checkseg", cfg.size, mask_root=fig_root)
    cached = MRIDataset(t, cfg, None, PreprocCache(tmp_dir, "checkseg", cfg.size), with_mask=True)
    res["cache_ok"] = all(torch.equal(plain[i]["mask"], cached[i]["mask"]) and
                          torch.equal(plain[i]["image"], cached[i]["image"]) for i in range(len(t)))
    del cached
    shutil.rmtree(tmp_dir, ignore_errors=True)
    # 3) полные аугментации: маска остается бинарной, площадь меняется в разумных пределах
    aug = MRIDataset(t, cfg, build_transforms(True, True, SEED), None, with_mask=True, image_root=fig_root, mask_root=fig_root)
    ratios, binary = [], True
    for i in range(len(t)):
        base = float(plain[i]["mask"].sum())
        for _ in range(5):
            m = aug[i]["mask"]
            binary &= bool(((m == 0) | (m == 1)).all())
            ratios.append(float(m.sum()) / base)
    res.update(binary_ok=binary, ratio_min=min(ratios), ratio_med=float(np.median(ratios)), ratio_max=max(ratios), n=len(t))
    return t, res


# ---------------------------------------------------------------- сетки


def plot_aug_grid(train: pd.DataFrame, cache: PreprocCache, domain: bool, title: str, out: Path, n_aug: int = 7) -> None:
    tf = build_transforms(True, domain, SEED)
    picks = [train[train["cls"] == c].sample(1, random_state=SEED).iloc[0] for c in CLASSES]
    fig, axes = plt.subplots(len(picks), n_aug + 1, figsize=(2.05 * (n_aug + 1), 2.2 * len(picks) + 0.6), dpi=100, facecolor="white")
    for r, row in enumerate(picks):
        img = cache.image(row["path"])
        for c in range(n_aug + 1):
            ax = axes[r, c]
            ax.imshow(img if c == 0 else tf(image=img)["image"], cmap="gray", vmin=0, vmax=255)
            ax.set_axis_off()
            if c == 0:
                ax.set_title(f"{CLS_RU[row['cls']]} · без аугментаций", fontsize=8, pad=3)
    fig.suptitle(title, fontsize=11, fontweight="bold", x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=0.2, h_pad=0.6)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def plot_mask_grid(t: pd.DataFrame, cfg: DataConfig, fig_root: Path, out: Path, n_aug: int = 6) -> None:
    rows = pd.concat([t[t["cls"] == c].head(2) for c in CLASSES if c != "no_tumor"]).reset_index(drop=True)
    plain = MRIDataset(rows, cfg, None, None, with_mask=True, image_root=fig_root, mask_root=fig_root)
    aug = MRIDataset(rows, cfg, build_transforms(True, True, SEED), None, with_mask=True, image_root=fig_root, mask_root=fig_root)
    fig, axes = plt.subplots(len(rows), n_aug + 1, figsize=(2.05 * (n_aug + 1), 2.2 * len(rows) + 0.6), dpi=100, facecolor="white")
    for r in range(len(rows)):
        for c in range(n_aug + 1):
            item = plain[r] if c == 0 else aug[r]
            img = item["image"][0].numpy()
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=img.min(), vmax=img.max())
            if item["mask"].any():
                ax.contour(item["mask"][0].numpy(), levels=[0.5], colors=[C_ORANGE], linewidths=1.1)
            ax.set_axis_off()
            if c == 0:
                ax.set_title(f"{CLS_RU[rows.loc[r, 'cls']]} · без аугментаций", fontsize=8, pad=3)
    fig.suptitle("Сегментация: снимок и маска преобразуются вместе (контур — маска опухоли)", fontsize=11, fontweight="bold",
                 x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975), w_pad=0.2, h_pad=0.6)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def _md_table(df: pd.DataFrame) -> str:
    df = df.reset_index()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--fig-root", type=Path, default=Path("data/external/figshare/png"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()
    cfg = DataConfig.from_yaml(args.config)
    processed = Path(cfg.processed_dir)
    cache = PreprocCache(cfg.cache_dir, "cls", cfg.size)
    train, val = load_split(processed, "train"), load_split(processed, "val")
    rd = args.report_dir / "dataset"

    # 1. тензоры, веса, состав эпохи
    cfg0 = replace(cfg, num_workers=0)
    seed_everything(cfg.seed)
    loader = make_loader(train, cfg0, train=True, cache=cache)
    t0 = time.perf_counter()
    counts, n_seen, first = np.zeros(len(CLASSES), int), 0, None
    for b in tqdm(loader, desc="эпоха train"):
        first = b if first is None else first
        counts += np.bincount(b["label"].numpy(), minlength=len(CLASSES))
        n_seen += len(b["label"])
    epoch_s = time.perf_counter() - t0
    vb = next(iter(make_loader(val, cfg0, train=False, cache=cache)))
    w = class_weights(train)
    cls_tab = pd.DataFrame({"снимков в train": train["cls"].value_counts().reindex(CLASSES).to_numpy(),
                            "в эпохе (drop_last)": counts, "вес класса": [round(float(v), 3) for v in w]},
                           index=[CLS_RU[c] for c in CLASSES])

    # 2. воспроизводимость
    a0, b0 = batches_digest(cfg0, train, cache, epochs=2), batches_digest(cfg0, train, cache, epochs=2)
    cfg2 = replace(cfg, num_workers=2)
    a2, b2 = batches_digest(cfg2, train, cache, epochs=2), batches_digest(cfg2, train, cache, epochs=2)
    other = batches_digest(replace(cfg0, seed=cfg.seed + 1), train, cache)
    repro = pd.DataFrame({
        "проверка": ["тот же seed, 0 процессов: два запуска совпали", "тот же seed, 2 процесса: два запуска совпали",
                     "вторая эпоха отличается от первой (0 процессов)", "вторая эпоха отличается от первой (2 процесса)",
                     "другой seed дает другие батчи"],
        "результат": [a0 == b0, a2 == b2, a0[0] != a0[1], a2[0] != a2[1], other[0] != a0[0]],
    }).set_index("проверка")
    repro["результат"] = repro["результат"].map({True: "да", False: "НЕТ"})

    # 3. скорость (один процесс, с аугментациями)
    ds_c = MRIDataset(train, cfg, build_transforms(True, cfg.domain_randomization, SEED), cache)
    ds_n = MRIDataset(train, cfg, build_transforms(True, cfg.domain_randomization, SEED), None)
    t0 = time.perf_counter(); [ds_c[i] for i in range(300)]; ms_cache = (time.perf_counter() - t0) / 300 * 1000
    t0 = time.perf_counter(); [ds_n[i] for i in range(100)]; ms_raw = (time.perf_counter() - t0) / 100 * 1000

    # 4. маски
    mt, mres = mask_checks(cfg, args.fig_root, processed, Path(cfg.cache_dir) / "_tmp_check")

    # 5. рандомизация домена и технические признаки
    sc = shortcut_under_aug(train, cache)

    # 6. сетки
    plot_aug_grid(train, cache, False, "Базовые аугментации: отражение, поворот до 15°, сдвиг и масштаб, яркость/контраст, шум, размытие",
                  rd / "aug_base.png")
    plot_aug_grid(train, cache, True, "Базовые + рандомизация домена: уменьшение-увеличение, JPEG, размытие/резкость, гамма, постеризация",
                  rd / "aug_domain.png")
    plot_mask_grid(mt, cfg, args.fig_root, rd / "aug_masks.png")

    yes = lambda v: "да" if v else "НЕТ"
    out = ["# Dataset и DataLoader", "",
           f"`src/dataset.py`, настройки — `{args.config.as_posix()}`. Источник — `splits.csv`, только рабочие снимки (primary): "
           f"train {len(train)}, val {len(val)}. Test в проверках не участвует.", "",
           "## Тензоры", "",
           f"- батч train: image {tuple(first['image'].shape)} {first['image'].dtype}, диапазон "
           f"{first['image'].min():.2f}…{first['image'].max():.2f}; label {tuple(first['label'].shape)} {first['label'].dtype}",
           f"- батч val (без аугментаций): image {tuple(vb['image'].shape)}, среднее {vb['image'].mean():.3f}, СКО {vb['image'].std():.3f}",
           f"- эпоха train: {n_seen} из {len(train)} снимков (batch_size {cfg.batch_size}, неполный последний батч отбрасывается)", "",
           "## Классы и веса", "", _md_table(cls_tab), "",
           "Вес класса = N / (K · n_c): чем меньше снимков, тем больше вес в функции потерь.", "",
           "## Воспроизводимость", "", _md_table(repro), "",
           "## Скорость (CPU, один процесс, с аугментациями)", "",
           f"- с кэшем препроцессинга: {ms_cache:.1f} мс на снимок; полная эпоха train — {epoch_s:.0f} с ({n_seen / epoch_s:.0f} снимков/с)",
           f"- без кэша (чтение файла + препроцессинг на лету): {ms_raw:.1f} мс на снимок — в {ms_raw / ms_cache:.1f} раза медленнее",
           f"- кэш: {len(cache.table)} снимков, {cache.files['images'].stat().st_size / 2**20:.0f} МБ", "",
           "## Аугментации", "", "![](dataset/aug_base.png)", "", "![](dataset/aug_domain.png)", "",
           f"## Маски (сегментация, {mres['n']} снимков Figshare, не связанных с test)", "",
           f"- отражение по горизонтали: снимок и маска отражаются одинаково — {yes(mres['flip_ok'])}",
           f"- чтение через кэш с масками совпадает с расчетом на лету — {yes(mres['cache_ok'])}",
           f"- после полных аугментаций маска остается бинарной — {yes(mres['binary_ok'])}; площадь маски относительно исходной: "
           f"минимум {mres['ratio_min']:.2f}, медиана {mres['ratio_med']:.2f}, максимум {mres['ratio_max']:.2f}", "",
           "![](dataset/aug_masks.png)", "",
           "## Рандомизация домена против технических признаков (train)", "",
           "AUC «опухоль / нет опухоли» по техническим признакам готового снимка 256×256 (0,5 — признак бесполезен). "
           "Чем ближе к 0,5 после аугментаций, тем труднее сети опереться на этот признак.", "",
           _md_table(sc), ""]
    (args.report_dir / "dataset_report.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(l for l in out if not l.startswith("![")))


if __name__ == "__main__":
    main()
