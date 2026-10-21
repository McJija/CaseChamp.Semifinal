"""Таблица правок меток (решение команды после шага 2.3).

Правила:
  relabel — метка файла кейса не совпадает с Figshare (exact-совпадение): принимается
            метка Figshare; правка распространяется на все файлы того же base_id
            (аугментированные копии), у которых была та же исходная метка;
  exclude — конфликтная группа шага 1.2 (один снимок в двух классах), которой нет
            в Figshare: исключается из train/val/test.

Исходные файлы в DATA_ROOT не меняются: правки применяются при сборке сплита.

Пример:
    python -m src.make_label_fixes
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_fixes(index: pd.DataFrame, mismatch: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    # 1) перемаркировка по Figshare, с распространением на копии того же base_id
    mm = mismatch.merge(index[["path", "base_id"]], on="path")
    for (base_id, old_cls), x in mm.groupby(["base_id", "cls"]):
        new_cls = x["fig_cls"].unique()
        if len(new_cls) != 1:
            raise ValueError(f"{base_id}: неоднозначная метка Figshare {new_cls}")
        fig = ", ".join(sorted(f"{i}.mat" for i in x["fig_id"].astype(str).unique()))
        pid = ", ".join(sorted(x["pid_base"].astype(str).unique()))
        for path in index.loc[(index["base_id"] == base_id) & (index["cls"] == old_cls), "path"]:
            rows[path] = {"path": path, "action": "relabel", "cls_orig": old_cls, "cls_new": new_cls[0],
                          "reason": f"Figshare {fig} (пациент {pid}): {new_cls[0]}"}
    # 2) исключение конфликтных групп, не решенных по Figshare
    relabeled_cases = set(cases.loc[cases["path"].isin(rows), "case"])
    for r in cases[~cases["case"].isin(relabeled_cases)].itertuples(index=False):
        rows[r.path] = {"path": r.path, "action": "exclude", "cls_orig": r.cls, "cls_new": "",
                        "reason": f"конфликт меток, случай {r.case}: снимка нет в Figshare"}
    fixes = pd.DataFrame(rows.values()).sort_values(["action", "path"]).reset_index(drop=True)

    # проверка: в решенных случаях после правок остается один класс
    c = cases.merge(fixes[["path", "cls_new"]], on="path", how="left")
    c["cls_final"] = c["cls_new"].where(c["cls_new"].notna() & (c["cls_new"] != ""), c["cls"])
    left = c[c["case"].isin(relabeled_cases)].groupby("case")["cls_final"].nunique()
    if (left > 1).any():
        raise ValueError(f"после правок в случаях {list(left.index[left > 1])} остались разные классы")
    return fixes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    index = pd.read_csv(args.processed / "index.csv")
    mismatch = pd.read_csv(args.processed / "figshare_label_mismatch.csv", dtype={"fig_id": str, "pid_base": str})
    cases = pd.read_csv(args.processed / "label_conflict_cases.csv")
    fixes = build_fixes(index, mismatch, cases)
    fixes.to_csv(args.processed / "label_fixes.csv", index=False, encoding="utf-8-sig")

    print(fixes.groupby(["action", "cls_orig", "cls_new"]).size().rename("файлов").to_string())
    n_orig = fixes.merge(index[["path", "is_original"]], on="path").groupby("action")["is_original"].sum().to_dict()
    print("из них оригиналов:", n_orig)
    print(f"таблица: {args.processed / 'label_fixes.csv'}")


if __name__ == "__main__":
    main()
