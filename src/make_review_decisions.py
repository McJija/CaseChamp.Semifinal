"""Решения по снимкам класса «нет опухоли» на основе отметок проверяющих.

Правило (принято 7 октября 2026 г., до обучения моделей):
  снимок остается в классе «нет опухоли», если
    1) никто из проверяющих не отметил «КТ»;
    2) никто не отметил «Опухоль»;
    3) режим и находку отметил хотя бы один проверяющий, режим не сводится к «?»,
       и к снимку нет подписи о плохом качестве.
  Иначе снимок исключается из обучения и оценки (в классы опухолей не переносится).
При расхождении проверяющих действует осторожное объединение: одной отметки «КТ» или
«Опухоль» достаточно для исключения. Снимки с другой патологией без отметки «Опухоль»
остаются: это отрицательные примеры, которые система должна уметь отличать от опухоли.

Для оставшихся снимков записываются подгруппы — по ним отдельно считается specificity:
  nt_finding: «норма» (все отметили «Норма») / «другая патология»;
  nt_mode:    «T1» (у всех T1 или T1 с контрастом) / «другой режим».

Вход:  reports/doctor/no_tumor_review/marks_merged.csv
Выход: data/processed/review_decisions.csv (строка на каждый из проверенных снимков).
Исключения применяет src/make_splits.py — фильтром после замороженного деления.

Пример:
    python -m src.make_review_decisions
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PARTS = ["train", "val", "test"]
R_CT = "разметка: КТ, не МРТ"
R_TUMOR = "разметка: отметка «Опухоль» хотя бы у одного проверяющего"
R_QUALITY = "разметка: режим не определен, плохое качество или нет ответа"


def decide(m: pd.DataFrame) -> pd.DataFrame:
    b = lambda col: m[col].astype(bool)
    quality = (m["note_kinds"].str.contains("качество") | (m["mode_all"] == "?") | b("mode_missing") | b("finding_missing"))
    reason = np.select([b("ct_any"), b("tumor_any"), quality], [R_CT, R_TUMOR, R_QUALITY], default="")
    keep = reason == ""
    out = m[["num", "path", "split", "patient_group_id", "n_reviewers"]].copy()
    out["keep"] = keep
    out["reason"] = reason
    out["mri_type"], out["finding"] = m["mode_all"], m["finding_all"]
    out["nt_finding"] = np.where(~keep, "", np.where(m["finding_all"] == "Норма", "норма", "другая патология"))
    out["nt_mode"] = np.where(~keep, "", np.where(b("t1_family_all"), "T1", "другой режим"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--marks", type=Path, default=Path("reports/doctor/no_tumor_review/marks_merged.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/review_decisions.csv"))
    args = ap.parse_args()

    m = pd.read_csv(args.marks, keep_default_na=False)
    d = decide(m)
    d.to_csv(args.out, index=False, encoding="utf-8")

    def by_part(x: pd.DataFrame) -> str:
        c = x["split"].value_counts().reindex(PARTS).fillna(0).astype(int)
        return f"{len(x)} ({c['train']} / {c['val']} / {c['test']})"

    print(f"проверено снимков: {len(d)}; остается: {by_part(d[d['keep']])}; исключается: {by_part(d[~d['keep']])}  (train / val / test)")
    for r, x in d[~d["keep"]].groupby("reason"):
        print(f"  {r}: {by_part(x)}")
    k = d[d["keep"]]
    print("подгруппы оставшихся:")
    print(pd.crosstab([k["nt_finding"], k["nt_mode"]], k["split"], margins=True, margins_name="всего").reindex(columns=PARTS + ["всего"]).to_string())
    print(f"размечены двумя проверяющими: {int((k['n_reviewers'] == 2).sum())}, одним: {int((k['n_reviewers'] == 1).sum())}")
    print(f"таблица: {args.out}")


if __name__ == "__main__":
    main()
