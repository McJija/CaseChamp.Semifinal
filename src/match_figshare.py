"""Сопоставление снимков кейса с Figshare.

Для каждого файла кейса ищется тот же снимок в Figshare (pHash по 8 преобразованиям
D4 + корреляция миниатюр, как на шаге 1.2). Уровни:
  exact — pHash <= --exact-phash и corr >= --exact-corr: тот же снимок;
  near  — только для оригиналов: pHash <= --phash-max и corr >= --corr-min.
По exact-совпадениям файл кейса получает id Figshare, маску и номер пациента (pid_base).

Группы пациентов для сплита: union-find по base_id, near-связям шага 1.2 и общему pid_base.

Результаты:
  data/processed/figshare_matches.csv — файл кейса → Figshare (id, класс, PID, tf, corr, уровень)
  data/processed/groups_pid.csv       — path, group_id, group_exact_id, patient_group_id, pid_base
  reports/figshare_match_report.md, reports/masks/match_check_*.png

Пример:
    python -m src.match_figshare
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
from tqdm import tqdm

from src.imhash import DSU, TF_NAMES, TFS, black_bbox, candidate_pairs, compute_features, pair_corr
from src.io_utils import read_image

SEED = 42
C_ORANGE = "#eb6834"
TF_RU = {"id": "без преобразования", "rot90": "поворот 90°", "rot180": "поворот 180°", "rot270": "поворот 270°",
         "flipH": "отражение по горизонтали", "flipV": "отражение по вертикали",
         "transpose": "транспонирование", "antitranspose": "антитранспонирование"}


def transfer_mask(fig_img: np.ndarray, fig_mask: np.ndarray, tf: int, case_img: np.ndarray) -> tuple[np.ndarray, float]:
    """Переносит маску Figshare на снимок кейса: преобразование tf, затем совмещение рамок
    содержимого (без черных полей). Возвращает маску в координатах кейса и корреляцию
    совмещенных снимков 256×256 (оценка качества совмещения)."""
    fi, fm = TFS[tf](fig_img), TFS[tf](fig_mask)
    fy0, fy1, fx0, fx1 = black_bbox(np.ascontiguousarray(fi))
    cy0, cy1, cx0, cx1 = black_bbox(case_img)
    out = np.zeros(case_img.shape, np.uint8)
    out[cy0:cy1, cx0:cx1] = cv2.resize(np.ascontiguousarray(fm[fy0:fy1, fx0:fx1]), (cx1 - cx0, cy1 - cy0),
                                       interpolation=cv2.INTER_NEAREST)
    a = cv2.resize(case_img[cy0:cy1, cx0:cx1], (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    b = cv2.resize(np.ascontiguousarray(fi[fy0:fy1, fx0:fx1]), (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return out, float((a * b).sum() / den) if den > 0 else 0.0


def best_matches(index: pd.DataFrame, meta: pd.DataFrame, pairs: pd.DataFrame, args) -> pd.DataFrame:
    is_orig = index["is_original"].to_numpy()
    p = pairs.copy()
    p["is_exact"] = (p["phash_dist"] <= args.exact_phash) & (p["corr"] >= args.exact_corr)
    p["is_near"] = p["is_exact"] | (is_orig[p["i"]] & (p["phash_dist"] <= args.phash_max) & (p["corr"] >= args.corr_min))
    n_exact = p[p["is_exact"]].groupby("i").size()
    best = p[p["is_near"]].sort_values(["corr", "phash_dist"], ascending=[False, True]).drop_duplicates("i")
    m = index[["path", "split", "cls", "aug", "is_original", "source"]].copy()
    m["level"] = "none"
    for col in ("fig_id", "fig_cls", "pid", "pid_base", "tf"):
        m[col] = ""
    m["phash_dist"], m["corr"], m["n_exact"] = -1, np.nan, 0
    bi, bj = best["i"].to_numpy(), best["j"].to_numpy()
    m.loc[bi, "level"] = np.where(best["is_exact"], "exact", "near")
    m.loc[bi, "fig_id"] = meta["id"].to_numpy()[bj].astype(str)
    for col in ("cls", "pid", "pid_base"):
        m.loc[bi, "fig_cls" if col == "cls" else col] = meta[col].to_numpy()[bj]
    m.loc[bi, "tf"] = np.array(TF_NAMES)[best["tf"].to_numpy()]
    m.loc[bi, "phash_dist"] = best["phash_dist"].to_numpy()
    m.loc[bi, "corr"] = best["corr"].to_numpy().round(4)
    m.loc[n_exact.index, "n_exact"] = n_exact.to_numpy()
    return m


def patient_groups(index: pd.DataFrame, matches: pd.DataFrame, near_links: pd.DataFrame, pid_levels: tuple) -> np.ndarray:
    dsu = DSU(len(index))
    for _, idx in index.groupby("base_id").indices.items():
        for k in idx[1:]:
            dsu.union(idx[0], k)
    for i, j in near_links[["i", "j"]].itertuples(index=False):
        dsu.union(i, j)
    with_pid = matches[matches["level"].isin(pid_levels)]
    for _, idx in with_pid.groupby("pid_base").indices.items():
        rows = with_pid.index[idx]
        for k in rows[1:]:
            dsu.union(rows[0], k)
    return dsu.labels("p")


# ---------------------------------------------------------------- графики


def plot_check(sel: pd.DataFrame, data_root: Path, fig_root: Path, title: str, out_prefix: Path, per_page: int = 10) -> None:
    pages = int(np.ceil(len(sel) / per_page))
    for pg in range(pages):
        chunk = sel.iloc[pg * per_page : (pg + 1) * per_page]
        rows = int(np.ceil(len(chunk) / 2))
        fig = plt.figure(figsize=(17.5, 3.5 * rows + 0.5), dpi=100)
        fh = fig.get_figheight()
        gs = fig.add_gridspec(rows, 7, width_ratios=[1, 1, 1, 0.12, 1, 1, 1], left=0.008, right=0.992,
                              top=1 - 1.0 / fh, bottom=0.005, wspace=0.03, hspace=0.26)
        for k, r in enumerate(chunk.itertuples(index=False)):
            row, col = divmod(k, 2)
            case = read_image(data_root / r.path)
            fimg = read_image(fig_root / "images" / f"{r.fig_id}.png")
            fmask = read_image(fig_root / "masks" / f"{r.fig_id}.png")
            tf = TF_NAMES.index(r.tf)
            moved, align = transfer_mask(fimg, fmask, tf, case)
            panels = [
                (case, None, f"КЕЙС · {r.split}/{r.cls} · {Path(r.path).name} · {case.shape[1]}×{case.shape[0]}"),
                (fimg, fmask, f"FIGSHARE · {r.fig_cls} · {r.fig_id}.mat · PID {r.pid}"),
                (case, moved, f"КЕЙС + перенесенная маска (совмещение r={align:.3f})"),
            ]
            axes = []
            for c, (img, mask, label) in enumerate(panels):
                ax = fig.add_subplot(gs[row, col * 4 + c])
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                if mask is not None and mask.any():
                    ax.contour(mask > 0, levels=[0.5], colors=[C_ORANGE], linewidths=1.2)
                ax.set_anchor("N")
                ax.set_axis_off()
                ax.set_title(label, fontsize=7.5, pad=3)
                axes.append(ax)
            mark = "  ⚠ МЕТКИ РАЗНЫЕ" if r.cls != r.fig_cls else ""
            axes[0].text(0.0, 1.085,
                         f"№{pg * per_page + k + 1}  pHash={r.phash_dist}, corr={r.corr:.3f}, Figshare → кейс: {TF_RU[r.tf]}{mark}",
                         transform=axes[0].transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom")
        fig.text(0.01, 1 - 0.2 / fh, f"{title} — стр. {pg + 1}/{pages}", fontsize=11, ha="left", va="top")
        fig.savefig(out_prefix.with_name(f"{out_prefix.name}_p{pg + 1}.png"))
        plt.close(fig)


# ---------------------------------------------------------------- отчет


def _md_table(df: pd.DataFrame) -> str:
    df = df.reset_index()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _level_table(m: pd.DataFrame, by: str) -> pd.DataFrame:
    t = pd.crosstab(m[by], m["level"]).reindex(columns=["exact", "near", "none"], fill_value=0)
    t["всего"] = t.sum(1)
    t.loc["всего"] = t.sum()
    t["% exact"] = (100 * t["exact"] / t["всего"]).round(1)
    return t


def make_report(index: pd.DataFrame, meta: pd.DataFrame, m: pd.DataFrame, g: pd.DataFrame, conflict_groups: set, args) -> str:
    out = ["# Сопоставление кейса с Figshare", ""]
    out += [
        f"Figshare: {len(meta)} срезов, {meta['pid_base'].nunique()} пациентов (`pid_base`). Уровни совпадения: "
        f"**exact** — pHash ≤ {args.exact_phash} и corr ≥ {args.exact_corr} (тот же снимок); "
        f"**near** — только оригиналы, pHash ≤ {args.phash_max} и corr ≥ {args.corr_min}.",
        "",
    ]
    orig = m[m["is_original"]]
    out += ["## Оригиналы кейса: найдено в Figshare, по классам", "", _md_table(_level_table(orig, "cls")), ""]
    out += ["## Оригиналы кейса: по схемам имен", "", _md_table(_level_table(orig, "source")), ""]
    out += ["## Все файлы кейса (с аугментированными копиями), по типу файла", "", _md_table(_level_table(m, "aug")), ""]

    # уникальные снимки
    u = orig.assign(group_exact_id=g.loc[orig.index, "group_exact_id"], hit=orig["level"] == "exact")
    ug = u.groupby(["cls", "group_exact_id"])["hit"].any().groupby("cls").agg(уникальных_снимков="size", найдено_exact="sum")
    ug["%"] = (100 * ug["найдено_exact"] / ug["уникальных_снимков"]).round(1)
    out += ["## Уникальные снимки кейса (exact-группы оригиналов) с маской из Figshare", "", _md_table(ug), ""]

    # покрытие Figshare
    ex = m[m["level"] == "exact"]
    used = set(ex["fig_id"].astype(int))
    cov = meta.assign(used=meta["id"].isin(used)).groupby("cls").agg(
        срезов=("id", "size"), есть_в_кейсе=("used", "sum"), пациентов=("pid_base", "nunique"))
    cov["пациентов_в_кейсе"] = meta[meta["id"].isin(used)].groupby("cls")["pid_base"].nunique()
    cov["% срезов"] = (100 * cov["есть_в_кейсе"] / cov["срезов"]).round(1)
    out += ["## Покрытие Figshare кейсом", "", _md_table(cov), "",
            f"Срезов Figshare, которых нет в кейсе: {len(meta) - len(used)} из {len(meta)}.", ""]

    out += ["## Преобразование Figshare → кейс (exact, оригиналы)", "",
            _md_table(orig.loc[orig["level"] == "exact", "tf"].map(TF_RU).value_counts().rename("файлов").to_frame()), ""]
    amb = ex[ex["n_exact"] > 1]
    out += [f"Файлов кейса с несколькими exact-кандидатами в Figshare: {len(amb)} "
            f"(в Figshare есть одинаковые срезы; берется кандидат с максимальной корреляцией).", ""]

    # метки
    out += ["## Согласованность меток (exact, все файлы): класс в кейсе × класс в Figshare", "",
            _md_table(pd.crosstab(ex["cls"], ex["fig_cls"], margins=True, margins_name="всего")), ""]
    bad = ex[ex["cls"] != ex["fig_cls"]]
    out += [f"Файлов кейса, чья метка не совпадает с Figshare: **{len(bad)}** "
            f"(оригиналов {int(bad['is_original'].sum())}). Список — `data/processed/figshare_label_mismatch.csv`.", ""]
    if len(bad):
        b = bad.assign(conflict=g.loc[bad.index, "group_id"].isin(conflict_groups))
        out += [f"Из них в 6 конфликтных группах шага 1.2: {int(b['conflict'].sum())}; вне их: {int((~b['conflict']).sum())}.", ""]

    # качество совмещения
    if "align_corr" in m:
        a = m.loc[m["align_corr"].notna(), "align_corr"]
        q = a.quantile([0.01, 0.05, 0.5]).round(3).tolist()
        out += ["## Качество совмещения для переноса масок (exact, оригиналы)", "",
                f"Корреляция совмещенных снимков 256×256: медиана {q[2]}, 5-й перцентиль {q[1]}, 1-й перцентиль {q[0]}. "
                f"Доля с r ≥ 0,95: {100 * (a >= 0.95).mean():.1f}%; с r ≥ 0,90: {100 * (a >= 0.90).mean():.1f}% (из {len(a)}).", "",
                "Сетки для визуальной проверки: `masks/match_check_exact_p*.png`, `masks/match_check_near_p*.png`.", ""]

    # группы пациентов
    d = index.assign(group_id=g["group_id"], group_exact_id=g["group_exact_id"], patient_group_id=g["patient_group_id"],
                     pid_base=g["pid_base"])
    d = d[~d["group_id"].isin(conflict_groups)]
    o = d[d["is_original"]]
    t = o.groupby("cls").agg(оригиналов=("path", "size"), уникальных_снимков=("group_exact_id", "nunique"),
                             групп_near=("group_id", "nunique"), групп_пациентов=("patient_group_id", "nunique"))
    t["оригиналов_с_PID"] = o[o["pid_base"] != ""].groupby("cls").size()
    t = t.fillna(0).astype(int)
    t.loc["всего"] = t.sum()
    out += ["## Группы для сплита (без 6 конфликтных групп)", "", _md_table(t), ""]
    ps = d.groupby("patient_group_id").agg(files=("path", "size"), cls_n=("cls", "nunique"),
                                           uniq=("group_exact_id", "nunique"), pids=("pid_base", lambda s: s[s != ""].nunique()))
    out += [
        f"- групп пациентов: {len(ps)}; с известным PID: {int((ps['pids'] > 0).sum())}; "
        f"групп, где слились несколько PID: {int((ps['pids'] > 1).sum())}",
        f"- уникальных снимков на группу: медиана {int(ps['uniq'].median())}, максимум {ps['uniq'].max()}; "
        f"файлов на группу: максимум {ps['files'].max()}",
        f"- групп пациентов с разными классами: **{int((ps['cls_n'] > 1).sum())}**",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("mrt"))
    ap.add_argument("--fig-root", type=Path, default=Path("data/external/figshare/png"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports"))
    ap.add_argument("--cand-max", type=int, default=12)
    ap.add_argument("--exact-phash", type=int, default=2)
    ap.add_argument("--exact-corr", type=float, default=0.98)
    ap.add_argument("--phash-max", type=int, default=10)
    ap.add_argument("--corr-min", type=float, default=0.90)
    ap.add_argument("--pid-levels", nargs="+", default=["exact", "near"], choices=["exact", "near"],
                    help="по каким совпадениям файлу кейса присваивается PID")
    ap.add_argument("--n-check", type=int, default=20)
    args = ap.parse_args()

    index = pd.read_csv(args.processed / "index.csv")
    groups = pd.read_csv(args.processed / "groups.csv")
    assert (groups["path"] == index["path"]).all()
    meta = pd.read_csv(args.fig_root / "metadata.csv", dtype={"pid": str, "pid_base": str})

    case = compute_features(index["path"].tolist(), [args.data_root / p for p in index["path"]],
                            args.processed / "hash_cache.npz")
    fig = compute_features(meta["id"].astype(str).tolist(), [args.fig_root / p for p in meta["image"]],
                           args.processed / "figshare_hash_cache.npz", desc="хеши Figshare")

    pairs, _ = candidate_pairs(case["hashes"][:, 0], fig["hashes"], args.cand_max, symmetric=False)
    pairs["corr"] = pair_corr(case["thumbs"], fig["thumbs"], pairs)
    m = best_matches(index, meta, pairs, args)

    # качество совмещения — для exact-совпавших оригиналов
    m["align_corr"] = np.nan
    todo = m[(m["level"] == "exact") & m["is_original"]]
    for k, r in tqdm(todo.iterrows(), total=len(todo), desc="совмещение"):
        fimg = read_image(args.fig_root / "images" / f"{r['fig_id']}.png")
        _, m.loc[k, "align_corr"] = transfer_mask(fimg, fimg, TF_NAMES.index(r["tf"]), read_image(args.data_root / r["path"]))
    m["align_corr"] = m["align_corr"].round(4)

    # группы пациентов
    dup = pd.read_csv(args.processed / "dup_pairs.csv", usecols=["i", "j", "is_near"])
    g = groups[["path", "group_id", "group_exact_id"]].copy()
    g["pid_base"] = np.where(m["level"].isin(args.pid_levels), m["pid_base"], "")
    g["patient_group_id"] = patient_groups(index, m, dup[dup["is_near"]], tuple(args.pid_levels))

    m.to_csv(args.processed / "figshare_matches.csv", index=False, encoding="utf-8")
    g.to_csv(args.processed / "groups_pid.csv", index=False, encoding="utf-8")
    ex = m[m["level"] == "exact"]
    ex[ex["cls"] != ex["fig_cls"]].to_csv(args.processed / "figshare_label_mismatch.csv", index=False, encoding="utf-8-sig")

    n_cls = index.assign(group_id=g["group_id"]).groupby("group_id")["cls"].nunique()
    conflict_groups = set(n_cls.index[n_cls > 1])

    rd = args.report_dir / "masks"
    rd.mkdir(parents=True, exist_ok=True)
    for old in rd.glob("match_check_*.png"):
        old.unlink()
    for level, n in (("exact", args.n_check), ("near", 10)):
        pool = m[(m["level"] == level) & m["is_original"]].sort_values("path")
        sel = pool.sample(min(n, len(pool)), random_state=SEED).sort_values(["cls", "path"])
        if len(sel):
            plot_check(sel, args.data_root, args.fig_root,
                       f"Кейс ↔ Figshare, уровень {level}: {len(sel)} случайных оригиналов из {len(pool)} (seed {SEED})",
                       rd / f"match_check_{level}")
            sel.to_csv(rd / f"match_check_{level}.csv", index=False, encoding="utf-8-sig")

    report = make_report(index, meta, m, g, conflict_groups, args)
    (args.report_dir / "figshare_match_report.md").write_text(report, encoding="utf-8")
    print(m[m["is_original"]].groupby(["cls", "level"]).size().unstack(fill_value=0).to_string())
    print(f"групп пациентов: {g['patient_group_id'].nunique()}; отчет: {args.report_dir / 'figshare_match_report.md'}")


if __name__ == "__main__":
    main()
