"""Поиск дубликатов и утечки train/test по содержимому изображений.

1. Для каждого файла: md5 байтов, обрезка черных полей, миниатюры 128 и 64 px,
   pHash для 8 преобразований D4 (повороты на 90° и отражения).
   Кэш: data/processed/hash_cache.npz.
2. Кандидаты: пары с минимальным (по преобразованиям) расстоянием Хэмминга
   pHash <= --cand-max.
3. Подтверждение: корреляция Пирсона миниатюр 64 px с учетом найденного
   преобразования. Два уровня связи:
   - exact: md5 совпадает или (phash <= --exact-phash и corr >= --exact-corr) —
     тот же снимок (пересохранение, отражение, поворот, яркость);
   - near:  phash <= --phash-max и corr >= --corr-min — тот же снимок
     с сильной обработкой или соседний срез того же пациента.
4. Группы: union-find по общему base_id и по связям near (включает exact).
   Результат — data/processed/groups.csv (path, group_id, group_exact_id).
5. Отчет reports/leakage_report.md, графики и сетки пар в reports/leakage/.

Пример:
    python -m src.find_duplicates --data-root mrt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.imhash import DSU, TF_NAMES, TFS, candidate_pairs, compute_features, pair_corr  # noqa: F401 (TFS — реэкспорт)
from src.io_utils import read_image

SEED = 42

# Цвета (палитра dataviz, светлая тема)
C_BLUE, C_ORANGE, C_MUTED = "#2a78d6", "#eb6834", "#898781"


def build_groups(index: pd.DataFrame, links: pd.DataFrame, prefix: str) -> np.ndarray:
    dsu = DSU(len(index))
    for _, idx in index.groupby("base_id").indices.items():
        for k in idx[1:]:
            dsu.union(idx[0], k)
    for i, j in links[["i", "j"]].itertuples(index=False):
        dsu.union(i, j)
    return dsu.labels(prefix)


# ---------------------------------------------------------------- графики


def _clean(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_scatter(pairs: pd.DataFrame, args, path: Path, max_points: int = 300_000) -> None:
    sub = pairs.sample(min(max_points, len(pairs)), random_state=SEED)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    jitter = np.random.default_rng(SEED).uniform(-0.3, 0.3, len(sub))
    for same, color, label in ((False, C_BLUE, "разные base_id (по имени не связаны)"),
                               (True, C_ORANGE, "общий base_id (аугментации одной группы)")):
        m = (sub["same_base"] == same).to_numpy()
        ax.scatter(sub["phash_dist"].to_numpy()[m] + jitter[m], sub["corr"].to_numpy()[m], s=5, c=color,
                   alpha=0.35, lw=0, label=label)
    ax.plot([-0.6, args.phash_max + 0.5, args.phash_max + 0.5], [args.corr_min] * 2 + [1.02], c=C_MUTED, lw=1, ls="--")
    ax.plot([-0.6, args.exact_phash + 0.5, args.exact_phash + 0.5], [args.exact_corr] * 2 + [1.02], c=C_MUTED, lw=1)
    ax.text(args.phash_max + 0.7, args.corr_min + 0.01, "near", fontsize=8, color=C_MUTED)
    ax.text(args.exact_phash + 0.7, args.exact_corr + 0.01, "exact", fontsize=8, color=C_MUTED)
    ax.set_xlabel("расстояние pHash (min по поворотам/отражениям), бит из 64")
    ax.set_ylabel("корреляция миниатюр 64×64")
    ax.set_title(f"Пары-кандидаты (случайные {len(sub):,} из {len(pairs):,}); связь — левый верхний угол",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower left", markerscale=2)
    ax.grid(alpha=0.25, lw=0.5)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_nn_hist(nn: np.ndarray, phash_max: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=120)
    ax.hist(nn, bins=np.arange(0, 34) - 0.5, color=C_BLUE, rwidth=0.85)
    ax.axvline(phash_max + 0.5, c=C_MUTED, lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("расстояние pHash до ближайшего другого снимка, бит (всегда четное: у pHash ровно 32 единицы)")
    ax.set_ylabel("файлов (лог. шкала)")
    ax.set_title("Ближайший сосед каждого снимка", fontsize=10, loc="left")
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pair_grid(pairs: pd.DataFrame, index: pd.DataFrame, data_root: Path, title: str, path: Path,
                   n: int = 12, sample: bool = True) -> None:
    if pairs.empty:
        return
    sub = pairs.sample(min(n, len(pairs)), random_state=SEED) if sample else pairs.head(n)
    sub = sub.sort_values(["phash_dist", "corr"], ascending=[True, False])
    rows = len(sub)
    fig, axes = plt.subplots(rows, 2, figsize=(6, 2.6 * rows), dpi=90)
    axes = np.atleast_2d(axes)
    for r, p in enumerate(sub.itertuples(index=False)):
        for c, k in enumerate((p.i, p.j)):
            row = index.iloc[k]
            ax = axes[r, c]
            ax.imshow(read_image(data_root / row["path"]), cmap="gray")
            ax.set_axis_off()
            ax.set_title(f"{row['split']}/{row['cls']}\n{Path(row['path']).name}", fontsize=7)
        axes[r, 0].text(-0.05, 0.5, f"d={p.phash_dist}\nr={p.corr:.3f}\n{TF_NAMES[p.tf]}", fontsize=8,
                        transform=axes[r, 0].transAxes, ha="right", va="center")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- отчет


def _md_table(df: pd.DataFrame) -> str:
    df = df.reset_index()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _leak_section(index: pd.DataFrame, links: pd.DataFrame, level: str) -> list[str]:
    """Test-файлы, у которых есть прямая связь (данного уровня) с train-файлом."""
    cross = links[links["split_i"] != links["split_j"]]
    test_idx = np.unique(np.where(index["split"].to_numpy()[cross["i"]] == "test", cross["i"], cross["j"]))
    train_idx = np.unique(np.where(index["split"].to_numpy()[cross["i"]] == "train", cross["i"], cross["j"]))
    t = index.iloc[test_idx]
    n_test = int((index["split"] == "test").sum())
    out = [f"### Уровень {level}: прямая связь test ↔ train", "",
           f"Test-файлов с копией в train: **{len(t)} из {n_test} ({len(t) / n_test:.1%})**, "
           f"из них оригиналов: {int(t['is_original'].sum())}. Train-файлов с копией в test: {len(train_idx)}.", ""]
    tab = pd.crosstab(index.loc[index["split"] == "test", "cls"],
                      index.loc[index["split"] == "test"].index.isin(test_idx)).rename(
        columns={True: "есть копия в train", False: "нет копии"})
    out += [_md_table(tab), ""]
    return out


def make_report(index: pd.DataFrame, pairs: pd.DataFrame, exact: pd.DataFrame, near: pd.DataFrame,
                nn: np.ndarray, args) -> str:
    out = ["# Утечка и дубликаты", ""]
    out += [
        "Уровни связи двух снимков:",
        f"- **exact** — тот же снимок: md5 совпадает или pHash ≤ {args.exact_phash} и корреляция ≥ {args.exact_corr} "
        "(пересохранение, отражение, поворот на 90°, яркость);",
        f"- **near** — exact или (оба файла — оригиналы, pHash ≤ {args.phash_max} и корреляция ≥ {args.corr_min}): "
        "тот же снимок с сильной обработкой или соседний срез того же пациента. "
        "Аугментированные копии в near-связях не участвуют (у BR/DA/SP корреляция завышается), "
        "в группы они попадают через base_id.",
        "",
        f"- файлов: {len(index)}; пар-кандидатов (pHash ≤ {args.cand_max}): {len(pairs):,}",
        f"- пар exact: {len(exact):,}; пар near (включая exact): {len(near):,}",
        f"- точных копий по md5: {int(index.duplicated('md5', keep=False).sum())} файлов в "
        f"{index.loc[index.duplicated('md5', keep=False), 'md5'].nunique()} группах",
        f"- файлов с хотя бы одной копией: exact {len(np.union1d(exact['i'], exact['j']))}, "
        f"near {len(np.union1d(near['i'], near['j']))}",
        "",
        "![](leakage/pairs_scatter.png)",
        "",
        "![](leakage/nn_hist.png)",
        "",
    ]

    d = near.copy()
    d["уровень"] = np.where(d["is_exact"], "exact", "near (не exact)")
    d["splits"] = np.where(d["split_i"] == d["split_j"], d["split_i"] + "↔" + d["split_j"], "train↔test")
    d["классы"] = np.where(d["cls_i"] == d["cls_j"], "один класс", "РАЗНЫЕ классы")
    out += ["## Пары по типу", "", _md_table(pd.crosstab([d["уровень"], d["splits"]], d["классы"])), ""]

    out += ["## Утечка train → test", ""]
    out += _leak_section(index, exact, "exact")
    out += _leak_section(index, near, "near")

    # Группы
    for col, level in (("group_exact_id", "exact"), ("group_id", "near")):
        g = index.groupby(col)
        gs = pd.DataFrame({"n": g.size(), "n_splits": g["split"].nunique(), "n_cls": g["cls"].nunique()})
        out += [
            f"## Группы уровня {level} (base_id ∪ связи {level}) — `{col}`",
            "",
            f"- всего групп: {len(gs)}; одиночных: {int((gs['n'] == 1).sum())}; максимальный размер: {gs['n'].max()}",
            f"- групп и в train, и в test: **{int((gs['n_splits'] > 1).sum())}**",
            f"- групп с разными классами: **{int((gs['n_cls'] > 1).sum())}**",
            "",
        ]

    cc = near[near["cls_i"] != near["cls_j"]]
    out += [f"## Конфликт меток: прямые пары снимков из разных классов — {len(cc)}", "",
            "Сетки для врача: `leakage/label_conflicts_exact.png`, `leakage/label_conflicts_near.png`; "
            "список — `data/processed/label_conflict_pairs.csv`.", ""]
    if len(cc):
        cc = cc.assign(классы=[" + ".join(sorted(x)) for x in zip(cc["cls_i"], cc["cls_j"])],
                       уровень=np.where(cc["is_exact"], "exact", "near"))
        out += [_md_table(pd.crosstab(cc["классы"], cc["уровень"])), ""]

    # группы aug_GMPN без оригинала в своем split
    a = index[index["source"] == "aug_GMPN"]
    has_orig = a.groupby(["base_id", "split"])["is_original"].any().unstack()  # NaN — группы нет в split
    rows = []
    for _, r in has_orig.iterrows():
        for split, other in (("train", "test"), ("test", "train")):
            v, v_other = r.get(split), r.get(other)
            if pd.notna(v) and not bool(v):
                rows.append({"split без оригинала": split,
                             "оригинал в другом split": bool(pd.notna(v_other) and bool(v_other))})
    if rows:
        out += ["## Группы aug_GMPN без оригинала в своем split", "",
                _md_table(pd.DataFrame(rows).value_counts().rename("групп").to_frame()), ""]

    # файлы "- Copy"
    cp = index[index["aug"] == "COPY"]
    same_md5 = sum(
        (index.loc[(index["base_id"] == r.base_id) & (index["aug"] == "orig"), "md5"] == r.md5).any() for r in cp.itertuples()
    )
    out += ["## Файлы `image(N) - Copy`", "",
            f"Совпадают по md5 с оригиналом `image(N)`: {same_md5} из {len(cp)}.", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("mrt"))
    ap.add_argument("--index", type=Path, default=Path("data/processed/index.csv"))
    ap.add_argument("--cache", type=Path, default=Path("data/processed/hash_cache.npz"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports"))
    ap.add_argument("--cand-max", type=int, default=16, help="макс. pHash-расстояние для кандидатов")
    ap.add_argument("--exact-phash", type=int, default=2)
    ap.add_argument("--exact-corr", type=float, default=0.98)
    ap.add_argument("--phash-max", type=int, default=10, help="порог pHash для near")
    ap.add_argument("--corr-min", type=float, default=0.90, help="порог корреляции для near")
    ap.add_argument("--save-corr-min", type=float, default=0.80, help="в CSV сохраняются пары с corr >= порога")
    args = ap.parse_args()

    index = pd.read_csv(args.index)
    feats = compute_features(index["path"].tolist(), [args.data_root / p for p in index["path"]], args.cache)
    index["md5"] = feats["md5"]
    index["h"], index["w"] = feats["shape"][:, 0], feats["shape"][:, 1]

    pairs, nn = candidate_pairs(feats["hashes"][:, 0], feats["hashes"], args.cand_max, symmetric=True)
    pairs["corr"] = pair_corr(feats["thumbs"], feats["thumbs"], pairs)
    for side in ("i", "j"):
        for col in ("split", "cls", "base_id"):
            pairs[f"{col}_{side}"] = index[col].to_numpy()[pairs[side]]
    pairs["same_base"] = pairs["base_id_i"] == pairs["base_id_j"]
    md5_eq = feats["md5"][pairs["i"]] == feats["md5"][pairs["j"]]
    pairs["is_exact"] = md5_eq | ((pairs["phash_dist"] <= args.exact_phash) & (pairs["corr"] >= args.exact_corr))
    # near — только между оригиналами: у BR/DA/SP-копий корреляция завышается (пересвет, шум), это давало
    # ложные связи между разными снимками; аугментированные копии входят в группы через base_id.
    is_orig = index["is_original"].to_numpy()
    both_orig = is_orig[pairs["i"]] & is_orig[pairs["j"]]
    pairs["is_near"] = pairs["is_exact"] | (
        both_orig & (pairs["phash_dist"] <= args.phash_max) & (pairs["corr"] >= args.corr_min))
    exact = pairs[pairs["is_exact"]].reset_index(drop=True)
    near = pairs[pairs["is_near"]].reset_index(drop=True)

    index["group_exact_id"] = build_groups(index, exact, "e")
    index["group_id"] = build_groups(index, near, "g")
    index["nn_phash"] = nn

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = index["path"].to_numpy()
    saved = pairs[(pairs["corr"] >= args.save_corr_min) | pairs["is_near"]]
    saved.assign(path_i=paths[saved["i"]], path_j=paths[saved["j"]], tf=np.array(TF_NAMES)[saved["tf"]]).drop(
        columns=["base_id_i", "base_id_j"]).to_csv(args.out_dir / "dup_pairs.csv", index=False)
    index[["path", "split", "cls", "base_id", "group_id", "group_exact_id", "md5", "h", "w", "nn_phash"]].to_csv(
        args.out_dir / "groups.csv", index=False)
    cc = near[near["cls_i"] != near["cls_j"]]
    cc.assign(path_i=paths[cc["i"]], path_j=paths[cc["j"]], tf=np.array(TF_NAMES)[cc["tf"]])[
        ["path_i", "cls_i", "split_i", "path_j", "cls_j", "split_j", "phash_dist", "corr", "tf", "is_exact"]].to_csv(
        args.out_dir / "label_conflict_pairs.csv", index=False)

    rd = args.report_dir / "leakage"
    rd.mkdir(parents=True, exist_ok=True)
    for old in rd.glob("pairs_*.png"):
        old.unlink()
    plot_scatter(pairs, args, rd / "pairs_scatter.png")
    plot_nn_hist(nn, args.phash_max, rd / "nn_hist.png")
    # Сетки для визуальной проверки порогов: только пары, не связанные по имени
    nb = pairs[~pairs["same_base"]]
    grids = {
        "pairs_exact.png": (nb["is_exact"], "exact"),
        "pairs_near_only.png": (nb["is_near"] & ~nb["is_exact"], "near, но не exact"),
        "pairs_rejected_high_corr.png": (~nb["is_near"] & (nb["corr"] >= 0.85), "отвергнуты, corr ≥ 0.85"),
    }
    for name, (mask, title) in grids.items():
        sel = nb[mask]
        plot_pair_grid(sel, index, args.data_root, f"{title}, разные base_id ({len(sel):,} пар)", rd / name)
    for lvl, mask in (("exact", cc["is_exact"]), ("near", ~cc["is_exact"])):
        sel = cc[mask]
        plot_pair_grid(sel, index, args.data_root, f"Разные классы, уровень {lvl} ({len(sel)} пар)",
                       rd / f"label_conflicts_{lvl}.png", n=40, sample=False)

    report = make_report(index, pairs, exact, near, nn, args)
    (args.report_dir / "leakage_report.md").write_text(report, encoding="utf-8")
    print(f"кандидатов: {len(pairs):,}; exact: {len(exact):,}; near: {len(near):,}; "
          f"групп near: {index['group_id'].nunique()}, exact: {index['group_exact_id'].nunique()}")
    print(f"отчет: {args.report_dir / 'leakage_report.md'}")


if __name__ == "__main__":
    main()
