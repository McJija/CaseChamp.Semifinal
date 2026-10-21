"""Общий пул данных для обеих схем и его кэш.

Пул — все уникальные снимки, каждый один раз:
  - срез Figshare (PNG + исходная маска) — если снимок есть в Figshare; сюда входят и срезы
    Figshare, у которых нет пары в кейсе;
  - рабочий снимок кейса без маски — если пары в Figshare нет (опухоли из других источников
    и класс «нет опухоли»).
Часть train/val/test: у срезов Figshare — от сплита кейса по пациенту (`pid_base`), у снимков
кейса — из splits.csv. Пациенты Figshare, которых нет в кейсе, — в train.

Классификация учится на всем пуле, сегментация — на строках `has_mask == True`.
Колонки: root (figshare / case), path, mask, has_mask, cls, split, source (для метрик по источникам:
figshare — срез Figshare; figshare_crop — снимок кейса, тот же срез Figshare в другом кадрировании,
без маски; other — снимок, которого в Figshare нет), fig_id, pid_base, group (пациент или группа — единица бутстрепа),
case_path (рабочий снимок кейса — двойник среза Figshare, для оценки на JPEG-версиях).

После изменения splits.csv (например, новых исключений по разметке) пул и кэш нужно пересобрать.
Для снимков «нет опухоли» колонки nt_finding (норма / другая патология), nt_mode (T1 / другой режим)
и n_reviewers берутся из review_decisions.csv.

Пример:
    python -m src.build_pool --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.analyze_mask_sources import inherit_split
from src.dataset import DataConfig, MRIDataset, PreprocCache, build_cache, class_weights
from src.labels import CLASSES, CLS_RU

SEED = 42
PARTS = ["train", "val", "test"]


def build_table(processed: Path, fig_root: Path) -> pd.DataFrame:
    meta = pd.read_csv(fig_root / "metadata.csv", dtype={"pid": str, "pid_base": str})
    splits = pd.read_csv(processed / "splits.csv", dtype={"pid_base": str}).fillna({"pid_base": ""})
    matches = pd.read_csv(processed / "figshare_matches.csv", dtype={"fig_id": str, "pid_base": str})
    fig = inherit_split(meta, splits, matches)

    live = splits[splits["split"].isin(PARTS)]
    # группа пациента для срезов Figshare — та же, что у его снимков в кейсе
    pid_group = live[live["pid_base"] != ""].groupby("pid_base")["patient_group_id"].agg(lambda v: sorted(set(v)))
    if (pid_group.map(len) > 1).any():
        raise ValueError("пациент Figshare попал в несколько групп кейса")
    pid_group = pid_group.map(lambda v: v[0])
    # двойник среза Figshare среди рабочих снимков кейса
    ex = matches[matches["level"] == "exact"].merge(live[["path", "primary", "group_exact_id"]], on="path")
    twin = ex[ex["primary"]].assign(fig_id=lambda d: d["fig_id"].astype(int)).drop_duplicates("fig_id").set_index("fig_id")["path"]

    pos = pd.DataFrame({
        "root": "figshare", "path": fig["image"], "mask": fig["mask"], "has_mask": True, "cls": fig["cls"],
        "split": fig["split"], "source": "figshare", "fig_id": fig["id"], "pid_base": fig["pid_base"],
        "group": fig["pid_base"].map(pid_group).fillna("fig:" + fig["pid_base"]),
        "case_path": fig["id"].map(twin).fillna(""),
    })
    # рабочие снимки кейса, которых нет в Figshare: у их группы одинаковых снимков нет exact-совпадения
    matched_groups = set(ex["group_exact_id"])
    rest = live[live["primary"] & ~live["group_exact_id"].isin(matched_groups)]
    # near-совпадение: тот же срез Figshare в другом кадрировании — это не «другой источник»
    near = set(matches.loc[matches["level"] == "near", "path"])
    other = pd.DataFrame({
        "root": "case", "path": rest["path"], "mask": "", "has_mask": False, "cls": rest["cls"], "split": rest["split"],
        "source": np.where(rest["path"].isin(near), "figshare_crop", "other"), "fig_id": -1, "pid_base": rest["pid_base"], "group": rest["patient_group_id"], "case_path": rest["path"],
    })
    table = pd.concat([pos, other], ignore_index=True).sort_values(["root", "path"]).reset_index(drop=True)

    # подгруппы класса «нет опухоли» по разметке проверяющих (для раздельного подсчета specificity)
    table["nt_finding"], table["nt_mode"], table["n_reviewers"] = "", "", 0
    rv = processed / "review_decisions.csv"
    if rv.exists():
        r = pd.read_csv(rv, keep_default_na=False).set_index("path")
        nt = table["cls"] == "no_tumor"
        paths = table.loc[nt, "path"]
        if not paths.isin(r.index).all() or not r.loc[paths, "keep"].astype(bool).all():
            raise ValueError("в пуле есть снимки «нет опухоли», не прошедшие отбор по разметке: сначала запустите src.make_splits")
        for col in ("nt_finding", "nt_mode", "n_reviewers"):
            table.loc[nt, col] = r.loc[paths, col].to_numpy()
    return table


def verify(table: pd.DataFrame, processed: Path, cfg: DataConfig, roots: dict) -> dict:
    res = {"groups_in_several_parts": int((table.groupby("group")["split"].nunique() > 1).sum())}
    splits = pd.read_csv(processed / "splits.csv", usecols=["path", "split"]).set_index("path")["split"]
    tw = table[(table["root"] == "figshare") & (table["case_path"] != "")]
    res["twin_part_mismatch"] = int((tw["case_path"].map(splits) != tw["split"]).sum())
    res["twins"] = len(tw)
    res["duplicate_paths"] = int(table["path"].duplicated().sum())

    cache = PreprocCache(cfg.cache_dir, "pool", cfg.size)
    masks = np.load(cache.files["masks"], mmap_mode="r")
    px = masks.reshape(len(masks), -1).sum(1)
    has = cache.table["has_mask"].to_numpy(dtype=bool)
    res["empty_among_masked"] = int((px[has] == 0).sum())
    res["nonempty_among_unmasked"] = int((px[~has] > 0).sum())
    sample = cache.table.sample(24, random_state=SEED)
    same = True
    for root, part in sample.groupby("root"):
        cached = MRIDataset(part, cfg, None, cache, with_mask=True)
        fresh = MRIDataset(part, cfg, None, None, with_mask=True, image_root=roots[root], mask_root=roots[root])
        for i in range(len(part)):
            a, b = cached[i], fresh[i]
            same &= bool((a["image"] == b["image"]).all() and (a["mask"] == b["mask"]).all() and a["has_mask"] == b["has_mask"])
    res["cache_equals_fresh"] = same
    return res


def _md_table(df: pd.DataFrame) -> str:
    df = df.reset_index()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _counts(d: pd.DataFrame) -> pd.DataFrame:
    t = pd.crosstab(d["cls"], d["split"]).reindex(index=CLASSES, columns=PARTS, fill_value=0)
    t.loc["всего"] = t.sum()
    t["всего"] = t.sum(1)
    t.index = [CLS_RU.get(c, c) for c in t.index]
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--fig-root", type=Path, default=Path("data/external/figshare/png"))
    ap.add_argument("--report", type=Path, default=Path("reports/pool_report.md"))
    args = ap.parse_args()
    cfg = DataConfig.from_yaml(args.config)
    processed = Path(cfg.processed_dir)
    roots = {"figshare": args.fig_root, "case": Path(cfg.data_root)}

    table = build_table(processed, args.fig_root)
    table.to_csv(processed / "pool.csv", index=False, encoding="utf-8")
    t = time.perf_counter()
    files = build_cache(table, None, cfg.cache_dir, "pool", cfg.size, roots=roots)
    build_s = time.perf_counter() - t
    res = verify(table, processed, cfg, roots)

    train = table[table["split"] == "train"]
    w = class_weights(train)
    src = pd.crosstab([table["source"], table["cls"]], table["split"]).reindex(columns=PARTS, fill_value=0)
    src["всего"] = src.sum(1)
    ext = table[(table["source"] == "other") & (table["cls"] != "no_tumor")]["split"].value_counts().reindex(PARTS)
    groups = table.groupby("split")["group"].nunique().reindex(PARTS)
    size_mb = (files["images"].stat().st_size + files["masks"].stat().st_size) / 2**20
    yes = lambda ok: "да" if ok else "НЕТ"
    out = ["# Общий пул данных", "",
           "`data/processed/pool.csv` — все уникальные снимки, каждый один раз. Снимок, который есть в Figshare, входит как срез Figshare "
           "с исходной маской; снимок кейса без пары в Figshare — без маски. Обе схемы учатся на части train этого пула: "
           "классификация — на всех строках, сегментация — на строках с маской.", "",
           "## Все снимки (классификация)", "", _md_table(_counts(table)), "",
           "## Снимки с маской (сегментация)", "", _md_table(_counts(table[table["has_mask"]]).drop(index=CLS_RU["no_tumor"])), "",
           "## По источникам", "",
           "`figshare` — срез Figshare (с маской); `figshare_crop` — снимок кейса, тот же срез Figshare в другом кадрировании "
           "(без маски); `other` — снимок, которого в Figshare нет.", "", _md_table(src), "",
           f"- опухоли строго из других источников (`other`): train {int(ext['train'])}, val {int(ext['val'])}, test {int(ext['test'])} — "
           "только они годятся для внешней проверки сегментатора",
           f"- групп пациентов: train {int(groups['train'])}, val {int(groups['val'])}, test {int(groups['test'])}",
           "", "Класс «нет опухоли» по подгруппам (разметка проверяющих):", "",
           _md_table(pd.crosstab([table.loc[table["cls"] == "no_tumor", "nt_finding"], table.loc[table["cls"] == "no_tumor", "nt_mode"]],
                                 table.loc[table["cls"] == "no_tumor", "split"], margins=True, margins_name="всего").reindex(columns=PARTS + ["всего"])), "",
           "- веса классов по train пула: " + ", ".join(f"{CLS_RU[c]} {float(v):.3f}" for c, v in zip(CLASSES, w)), "",
           "## Проверки", "",
           f"- групп пациентов в нескольких частях: **{res['groups_in_several_parts']}**",
           f"- срезов Figshare не в той части, что их двойник в кейсе: **{res['twin_part_mismatch']}** (двойников: {res['twins']})",
           f"- повторяющихся путей: **{res['duplicate_paths']}**",
           f"- пустых масок у строк с маской: **{res['empty_among_masked']}**; непустых у строк без маски: **{res['nonempty_among_unmasked']}**",
           f"- кэш совпадает с расчетом на лету (24 случайные строки): {yes(res['cache_equals_fresh'])}", "",
           "## Кэш", "",
           f"`{files['images'].as_posix()}` и `{files['masks'].as_posix()}`: {len(table)} строк, {size_mb:.0f} МБ, сборка {build_s:.0f} с.", ""]
    args.report.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
