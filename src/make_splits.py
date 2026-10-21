"""Финальный сплит train/val/test по группам пациентов (исходное деление организаторов не используется).

Исходный сплит организаторов не используется (72,9% test — копии train).
1. К индексу применяются правки меток (label_fixes.csv): relabel / exclude.
2. В каждой группе одинаковых снимков (group_exact_id) выбирается один оригинал —
   primary (наибольшее разрешение, затем по имени). Обучение и оценка идут по primary;
   остальные файлы наследуют split своей группы (нужны для проверки устойчивости на копиях).
3. Группы пациентов (patient_group_id) делятся StratifiedGroupKFold(n_splits=7, seed=42);
   test и val — два фолда с самыми ровными долями классов, остальные — train (≈ 72 / 14 / 14 %).
4. Деление заморожено в data/processed/folds_frozen.csv (группа пациента → фолд и роль). Пока файл
   существует, фолды берутся из него и не пересчитываются. Удалить файл = осознанно разделить заново.
5. Исключения по разметке проверяющих (review_decisions.csv) применяются фильтром ПОСЛЕ деления:
   снимок получает split = excluded, остальные снимки остаются в своих частях.

Результат: data/processed/splits.csv, reports/splits_report.md.

Пример:
    python -m src.make_splits
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
N_SPLITS = 7
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
FROZEN = "folds_frozen.csv"
REVIEW = "review_decisions.csv"


def load(processed: Path) -> pd.DataFrame:
    index = pd.read_csv(processed / "index.csv")
    g = pd.read_csv(processed / "groups_pid.csv", dtype={"pid_base": str})
    sizes = pd.read_csv(processed / "groups.csv", usecols=["path", "h", "w"])
    assert (g["path"] == index["path"]).all() and (sizes["path"] == index["path"]).all()
    d = index.rename(columns={"split": "split_orig", "cls": "cls_orig"})
    d["group_exact_id"], d["patient_group_id"] = g["group_exact_id"], g["patient_group_id"]
    d["pid_base"] = g["pid_base"].fillna("")
    d["area"] = sizes["h"] * sizes["w"]

    fixes = pd.read_csv(processed / "label_fixes.csv").set_index("path")
    act = d["path"].map(fixes["action"])
    d["cls"] = np.where(act == "relabel", d["path"].map(fixes["cls_new"]), d["cls_orig"])
    d["excluded"] = (act == "exclude").to_numpy()

    # решения по разметке проверяющих: исключается вся группа одинаковых снимков (сам снимок и его копии)
    d["review_reason"] = ""
    if (processed / REVIEW).exists():
        r = pd.read_csv(processed / REVIEW, keep_default_na=False)
        bad = r[~r["keep"].astype(bool)]
        exact = d.set_index("path")["group_exact_id"]
        reason = dict(zip(bad["path"].map(exact), bad["reason"]))
        d["review_reason"] = d["group_exact_id"].map(reason).fillna("")
    return d


def assign(d: pd.DataFrame, frozen: pd.DataFrame | None = None) -> pd.DataFrame:
    """Назначает primary, фолд и часть. frozen — замороженное деление (patient_group_id, fold, role);
    без него фолды считаются заново. Исключения по разметке на деление не влияют."""
    use = d[~d["excluded"]]
    multi = use.groupby("patient_group_id")["cls"].nunique()
    if (multi > 1).any():
        raise ValueError(f"группы пациентов с разными классами: {list(multi.index[multi > 1])}")

    # один оригинал на уникальный снимок
    orig = use[use["is_original"]].sort_values(["group_exact_id", "area", "path"], ascending=[True, False, True])
    primary = orig.drop_duplicates("group_exact_id")
    d["primary"] = d["path"].isin(primary["path"])

    p = primary.sort_values("path").reset_index(drop=True)  # фиксированный порядок -> воспроизводимый сплит
    if frozen is None:
        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        fold_of = {}
        for fold, (_, idx) in enumerate(sgkf.split(p, p["cls"], p["patient_group_id"])):
            for grp in p.loc[idx, "patient_group_id"].unique():
                fold_of[grp] = fold
    else:
        fold_of = dict(zip(frozen["patient_group_id"], frozen["fold"].astype(int)))
        missing = sorted(set(p["patient_group_id"]) - set(fold_of))
        if missing:
            raise ValueError(f"в замороженном делении нет групп: {missing[:5]} (всего {len(missing)})")
    fold = d["patient_group_id"].map(fold_of)
    d["fold"] = fold.fillna(-1).astype(int)

    # Группы очень неравные по размеру, поэтому фолды неравны по числу снимков. test и val —
    # два фолда, где доли классов ближе всего к 1/N_SPLITS (правило фиксировано, от моделей не зависит).
    t = pd.crosstab(p["patient_group_id"].map(fold_of), p["cls"])
    dev = (t / t.sum() - 1 / N_SPLITS).abs().max(axis=1).sort_values(kind="stable")
    if frozen is None:
        test_fold, val_fold = int(dev.index[0]), int(dev.index[1])
    else:
        roles = frozen.drop_duplicates("fold").set_index("fold")["role"]
        test_fold, val_fold = int(roles[roles == "test"].index[0]), int(roles[roles == "val"].index[0])
    d.attrs.update(fold_table=t, fold_dev=dev, test_fold=test_fold, val_fold=val_fold, frozen=frozen is not None)
    reviewed_out = (d["review_reason"] != "").to_numpy()
    d["split"] = np.select([d["excluded"], fold.isna(), reviewed_out, fold == test_fold, fold == val_fold],
                           ["excluded", "unused", "excluded", "test", "val"], default="train")
    d["exclude_reason"] = np.where(d["split"] != "excluded", "", np.where(d["excluded"], "конфликт меток, случай 1", d["review_reason"]))
    return d


def _md_table(df: pd.DataFrame) -> str:
    df = df.reset_index()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _split_table(x: pd.DataFrame, value: str | None = None) -> pd.DataFrame:
    cols = ["train", "val", "test"]
    if value is None:
        t = pd.crosstab(x["cls"], x["split"])
    else:
        t = x.groupby(["cls", "split"])[value].nunique().unstack(fill_value=0)
    t = t.reindex(index=CLASSES, columns=cols, fill_value=0)
    t.loc["всего"] = t.sum()
    tot = t.sum(1)
    for c in cols:
        t[f"{c} %"] = (100 * t[c] / tot).round(1)
    t["всего"] = tot
    return t


def make_report(d: pd.DataFrame, processed: Path) -> str:
    live = d[d["split"].isin(["train", "val", "test"])]
    prim = live[live["primary"]]
    out = ["# Финальный сплит (вариант B)", ""]
    ft = d.attrs["fold_table"].reindex(columns=CLASSES)
    share = (100 * ft / ft.sum()).round(1)
    share["макс. отклонение, п.п."] = (100 * d.attrs["fold_dev"]).round(1)
    share["снимков"] = ft.sum(1)
    share["роль"] = ["test" if f == d.attrs["test_fold"] else "val" if f == d.attrs["val_fold"] else "train" for f in share.index]
    out += [
        ("Деление взято из замороженного файла `folds_frozen.csv`. " if d.attrs["frozen"] else "") +
        f"`StratifiedGroupKFold(n_splits={N_SPLITS}, shuffle=True, random_state={SEED})` по группам пациентов "
        f"(`patient_group_id`). Группы неравны по размеру, поэтому фолды неравны по числу снимков; "
        f"**test — фолд {d.attrs['test_fold']}, val — фолд {d.attrs['val_fold']}**: два фолда, где доли классов ближе всего "
        f"к 1/{N_SPLITS} (правило фиксировано заранее и не зависит от моделей). Остальные фолды — train. "
        "Исходный сплит организаторов не используется.",
        "",
        "Доля каждого класса в фолде, % (рабочие снимки до исключений по разметке проверяющих):",
        "",
        _md_table(share),
        "",
        f"- файлов всего: {len(d)}; в сплите: {len(live)}; исключено: {int((d['split'] == 'excluded').sum())}; "
        f"не используется (группа без оригинала): {int((d['split'] == 'unused').sum())}",
        f"- перемаркировано по Figshare: {int((d['cls'] != d['cls_orig']).sum())} файлов (гипофиз → менингиома)",
        f"- рабочих снимков (primary — один оригинал на уникальный снимок): **{len(prim)}**",
        "",
    ]
    exc = d[d["split"] == "excluded"]
    if len(exc):
        et = exc.groupby("exclude_reason").agg(файлов=("path", "size"), уникальных_снимков=("group_exact_id", "nunique"))
        role = np.select([exc["fold"] == d.attrs["test_fold"], exc["fold"] == d.attrs["val_fold"], exc["fold"] < 0], ["test", "val", "вне деления"],
                         default="train")
        work = exc[exc["primary"]]
        by_role = pd.crosstab(work["exclude_reason"], pd.Series(role, index=exc.index)[work.index]).add_prefix("рабочих снимков из ")
        out += ["## Исключено", "", "Исключения применяются после деления: снимок убирается из своей части, остальные остаются на местах.", "",
                _md_table(et.join(by_role, how="left").fillna(0).astype(int)), ""]
    out += ["## Рабочие снимки (primary) — на них идут обучение и оценка", "", _md_table(_split_table(prim)), ""]
    out += ["## Группы пациентов", "", _md_table(_split_table(prim, "patient_group_id")), ""]
    known = prim[prim["pid_base"] != ""]
    out += ["## Пациенты Figshare с известным PID", "", _md_table(_split_table(known, "pid_base").drop(index="no_tumor")), ""]
    out += ["## Все файлы (с дубликатами и аугментированными копиями)", "", _md_table(_split_table(live)), ""]

    aug = live[~live["is_original"] & (live["aug"] != "COPY")]
    out += ["## Аугментированные копии организаторов по split (для шага 6.3)", "",
            _md_table(pd.crosstab([aug["split"], aug["cls"]], aug["aug"])), ""]

    out += ["## Источник (схема имени) рабочих снимков по split, %", "",
            _md_table((100 * pd.crosstab(prim["source"], prim["split"], normalize="columns")).round(1)[["train", "val", "test"]]), ""]
    out += ["## Откуда пришли рабочие снимки (исходный сплит организаторов → новый)", "",
            _md_table(pd.crosstab(prim["split_orig"], prim["split"])[["train", "val", "test"]]), ""]

    # проверки утечки
    def crossing(col: str) -> int:
        return int((live.groupby(col)["split"].nunique() > 1).sum())

    pos = pd.Series(np.arange(len(d)), index=d["path"])
    sp = d["split"].to_numpy()
    dup = pd.read_csv(processed / "dup_pairs.csv", usecols=["i", "j", "is_exact", "is_near"])
    ok = np.isin(sp[dup["i"]], ["train", "val", "test"]) & np.isin(sp[dup["j"]], ["train", "val", "test"])
    cross = ok & (sp[dup["i"]] != sp[dup["j"]])
    fm = pd.read_csv(processed / "figshare_matches.csv", usecols=["path", "level", "fig_id"], dtype={"fig_id": str})
    fm = fm[fm["level"] != "none"].assign(split=sp[pos[fm.loc[fm["level"] != "none", "path"]].to_numpy()])
    fm = fm[fm["split"].isin(["train", "val", "test"])]
    out += [
        "## Проверки утечки (везде должно быть 0)",
        "",
        f"- групп пациентов в нескольких split: **{crossing('patient_group_id')}**",
        f"- уникальных снимков (group_exact_id) в нескольких split: **{crossing('group_exact_id')}**",
        f"- base_id в нескольких split: **{crossing('base_id')}**",
        f"- пар-дубликатов exact между разными split: **{int((cross & dup['is_exact']).sum())}**",
        f"- пар near между разными split: **{int((cross & dup['is_near']).sum())}**",
        f"- пациентов Figshare (pid_base) в нескольких split: **{int((live[live['pid_base'] != ''].groupby('pid_base')['split'].nunique() > 1).sum())}**",
        f"- срезов Figshare (fig_id) в нескольких split: **{int((fm.groupby('fig_id')['split'].nunique() > 1).sum())}**",
        f"- групп пациентов с разными классами: **{int((live.groupby('patient_group_id')['cls'].nunique() > 1).sum())}**",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--report", type=Path, default=Path("reports/splits_report.md"))
    args = ap.parse_args()

    frozen_path = args.processed / FROZEN
    frozen = pd.read_csv(frozen_path) if frozen_path.exists() else None
    d = assign(load(args.processed), frozen)
    if frozen is None:  # первое деление — замораживаем
        fz = d.loc[d["fold"] >= 0, ["patient_group_id", "fold"]].drop_duplicates().sort_values("patient_group_id")
        fz["role"] = np.select([fz["fold"] == d.attrs["test_fold"], fz["fold"] == d.attrs["val_fold"]], ["test", "val"], default="train")
        fz.to_csv(frozen_path, index=False, encoding="utf-8")
        print(f"деление заморожено: {frozen_path} ({len(fz)} групп)")
    cols = ["path", "split", "cls", "primary", "is_original", "aug", "source", "base_id", "group_exact_id",
            "patient_group_id", "pid_base", "fold", "cls_orig", "split_orig", "exclude_reason"]
    d[cols].to_csv(args.processed / "splits.csv", index=False, encoding="utf-8")
    report = make_report(d, args.processed)
    args.report.write_text(report, encoding="utf-8")
    prim = d[d["primary"] & d["split"].isin(["train", "val", "test"])]
    print(pd.crosstab(prim["cls"], prim["split"]).reindex(CLASSES)[["train", "val", "test"]].to_string())
    print(f"splits: {args.processed / 'splits.csv'}; отчет: {args.report}")


if __name__ == "__main__":
    main()
