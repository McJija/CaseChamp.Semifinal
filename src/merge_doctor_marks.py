"""Сведение отметок проверяющих по классу «нет опухоли» (после src/read_doctor_marks.py).

Наборы листов определяются по именам файлов и виду отметок (кто именно размечал, по файлам
установить нельзя):
  A — файлы 1.png … 8.png (красные точки), листы 1–8;
  B — файлы page_0N (красные галочки), листы 1–4;
  C — остальные файлы (темно-красные точки), листы 9–12.
Листы 1–4 размечены дважды (A и B), остальные — по одному разу.

Вход:  marks_raw.csv (программа), marks_overrides.csv (ручные поправки считывания),
       marks_notes.csv (подписи проверяющих, переписаны с листов вручную),
       marks_doubts.csv (отметки, которые стоит пересмотреть, — неэкспертная оценка).
Выход: marks_merged.csv (строка на снимок, ответы каждого набора и сводные признаки),
       answers.csv (заполненные колонки mri_type, finding, comment — объединение ответов),
       reports/doctor_marks_report.md.
Сплит, пул и кэши этот скрипт не меняет.

Пример:
    python -m src.merge_doctor_marks
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.plot_no_tumor_review import FINDING, MODE

SETS = ["A", "B", "C"]
PARTS = ["train", "val", "test"]


def reviewer_set(file: str) -> str:
    stem = Path(file).stem
    if stem.startswith("page_"):
        return "B"
    return "A" if stem.isdigit() else "C"


def parse_nums(spec: str) -> list[int]:
    out = []
    for part in str(spec).split(","):
        a, _, b = part.strip().partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def as_set(value: str) -> frozenset:
    return frozenset(v for v in str(value).split("|") if v)


def primary_mode(s: frozenset) -> str:
    """Один режим для сравнения: «?» рядом с другим вариантом считается сомнением, а не ответом."""
    core = s - {"?"}
    if len(core) == 1:
        return next(iter(core))
    return "?" if s == {"?"} else ("пусто" if not s else "несколько")


def primary_finding(s: frozenset) -> str:
    if len(s) == 1:
        return next(iter(s))
    if s == {"Опухоль", "Другое"}:
        return "Опухоль или другое"
    return "пусто" if not s else "несколько"


def merge(raw: pd.DataFrame, overrides: pd.DataFrame, notes: pd.DataFrame, answers: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["set"] = raw["file"].map(reviewer_set)
    for o in overrides.itertuples(index=False):
        hit = (raw["set"] == o.set) & (raw["num"] == o.num)
        if hit.sum() != 1:
            raise ValueError(f"поправка {o.set}-{o.num}: найдено строк {int(hit.sum())}")
        raw.loc[hit, o.field] = o.value
    if raw.duplicated(["set", "num"]).any():
        raise ValueError("один и тот же снимок дважды в одном наборе")

    note_rows = [{"set": n.set, "num": k, "note": n.note, "kind": n.kind} for n in notes.itertuples(index=False) for k in parse_nums(n.nums)]
    note_df = pd.DataFrame(note_rows)
    per_num = note_df.groupby("num").apply(lambda x: "; ".join(f"{s}: {t}" for s, t in zip(x["set"], x["note"])), include_groups=False)
    kinds = note_df.groupby("num")["kind"].agg(lambda v: "|".join(sorted(set(v))))

    rows = []
    by = {(r.set, r.num): r for r in raw.itertuples(index=False)}
    for a in answers.itertuples(index=False):
        row = {"num": a.num, "page": a.page, "path": a.path, "split": a.split, "patient_group_id": a.patient_group_id}
        modes, finds = [], []
        for s in SETS:
            r = by.get((s, a.num))
            row[f"mode_{s}"] = r.mode if r is not None else ""
            row[f"finding_{s}"] = r.finding if r is not None else ""
            row[f"seen_{s}"] = r is not None
            if r is not None:
                modes.append(as_set(r.mode))
                finds.append(as_set(r.finding))
        mode_all, find_all = frozenset().union(*modes), frozenset().union(*finds)
        row.update({
            "n_reviewers": len(modes),
            "mode_all": "|".join(o for o in MODE if o in mode_all),
            "finding_all": "|".join(o for o in FINDING if o in find_all),
            "mode_missing": all(len(m) == 0 for m in modes),
            "finding_missing": all(len(f) == 0 for f in finds),
            "ct_any": "КТ" in mode_all,
            "t1c_any": "T1+К" in mode_all,
            "t1c_all": all(m == {"T1+К"} for m in modes),
            "t1_family_all": all(len(m) > 0 and m <= {"T1+К", "T1"} for m in modes),
            "unsure_mode": "?" in mode_all,
            "tumor_any": "Опухоль" in find_all,
            "other_any": "Другое" in find_all,
            "norm_all": all(f == {"Норма"} for f in finds),
            "notes": per_num.get(a.num, ""),
            "note_kinds": kinds.get(a.num, ""),
        })
        if len(modes) == 2:
            row["mode_agree"] = "да" if modes[0] == modes[1] else ("частично" if modes[0] & modes[1] else "нет")
            row["finding_agree"] = "да" if finds[0] == finds[1] else ("частично" if finds[0] & finds[1] else "нет")
        else:
            row["mode_agree"] = row["finding_agree"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, index_name: str | None = None) -> str:
    df = df.reset_index()
    if index_name is not None:
        df = df.rename(columns={df.columns[0]: index_name})
    df = df.map(lambda v: v.replace("|", ", ") if isinstance(v, str) else v)  # «|» ломает таблицу Markdown
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _nums(values) -> str:
    """Список номеров с диапазонами: 1, 3–5, 9."""
    v = sorted(int(x) for x in values)
    if not v:
        return "нет"
    out, start, prev = [], v[0], v[0]
    for x in v[1:] + [None]:
        if x is None or x != prev + 1:
            out.append(str(start) if start == prev else f"{start}–{prev}")
            start = x
        prev = x
    return ", ".join(out)


def scenarios(m: pd.DataFrame) -> pd.DataFrame:
    quality = m["note_kinds"].str.contains("качество") | (m["mode_all"] == "?") | m["mode_missing"] | m["finding_missing"]
    rules = {
        "все снимки": pd.Series(True, index=m.index),
        "без КТ": ~m["ct_any"],
        "без КТ и без отметки «Опухоль»": ~m["ct_any"] & ~m["tumor_any"],
        "**принято:** без КТ, без отметки «Опухоль», режим определен, качество без замечаний": ~m["ct_any"] & ~m["tumor_any"] & ~quality,
        "МРТ в любом режиме, «Норма» у всех проверяющих": ~m["ct_any"] & m["norm_all"],
        "T1 или T1+К у всех, «Норма» у всех": m["t1_family_all"] & m["norm_all"],
        "T1+К хотя бы у одного, «Норма» у всех": m["t1c_any"] & ~m["ct_any"] & m["norm_all"],
        "T1+К у всех, «Норма» у всех": m["t1c_all"] & m["norm_all"],
    }
    rows = []
    for name, mask in rules.items():
        x = m[mask]
        c = x["split"].value_counts().reindex(PARTS).fillna(0).astype(int)
        rows.append({"правило": name, "train": c["train"], "val": c["val"], "test": c["test"], "всего": len(x),
                     "групп пациентов": x["patient_group_id"].nunique()})
    return pd.DataFrame(rows).set_index("правило")


def make_report(m: pd.DataFrame, raw: pd.DataFrame, overrides: pd.DataFrame, notes: pd.DataFrame, doubts: pd.DataFrame) -> str:
    out = ["# Отметки проверяющих по классу «нет опухоли»", "",
           "Сведение отметок с листов проверки (сканы листов в репозиторий не входят). Этот отчет — только чтение и сверка ответов; отбор снимков описан в конце раздела о правилах.", ""]

    # 1. что прочитано
    files = raw.assign(set=raw["file"].map(reviewer_set)).groupby("set").agg(файлов=("file", "nunique"), листы=("page", lambda v: _nums(v.unique())),
                                                                              снимков=("num", "nunique"))
    files.insert(0, "вид отметок", pd.Series({"A": "красные точки", "B": "красные галочки", "C": "темно-красные точки"}))
    out += ["## Что прочитано", "", _md_table(files, "набор"), "",
            f"- снимков всего: {len(m)}; размечены двумя наборами: {int((m['n_reviewers'] == 2).sum())} (листы 1–4); одним: {int((m['n_reviewers'] == 1).sum())}",
            "- кто из проверяющих врач, по файлам установить нельзя; наборы A, B, C названы по виду отметок", ""]

    # 2. надежность считывания
    px = raw[[c for c in raw.columns if c.startswith("px_")]].to_numpy()
    out += ["## Надежность считывания", "",
            "Отметка засчитывается, если в окне клетки не меньше 40 цветных пикселей.",
            f"- окон клеток всего: {px.size}; пустых (0 пикселей): {int((px == 0).sum())}; со следами 1–39 пикселей: {int(((px > 0) & (px < 40)).sum())}; "
            f"с отметкой (от 40): {int((px >= 40).sum())}",
            f"- ручных поправок считывания: {len(overrides)} (снимки {_nums(overrides['num'])}) — в окно клетки попала подпись к соседнему снимку; "
            "причины — в `marks_overrides.csv`",
            "- проверено глазами: все 16 листов целиком; клетка за клеткой сверены с программой лист 2 (наборы A и B), лист 3 (B) и лист 9 (C) — "
            "160 клеток, расхождений нет; все клетки с пропуском или двумя отметками просмотрены крупно",
            "- текст подписей программа не читает: подписи переписаны с листов вручную (`marks_notes.csv`), возможны неточности в отнесении подписи к номеру", ""]

    # 3. пропуски и двойные отметки
    long = raw.assign(set=raw["file"].map(reviewer_set))
    for o in overrides.itertuples(index=False):
        long.loc[(long["set"] == o.set) & (long["num"] == o.num), o.field] = o.value
    out += ["## Пропуски и двойные отметки", ""]
    t = []
    for s, x in long.groupby("set"):
        fm, ff = x["mode"].map(as_set), x["finding"].map(as_set)
        t.append({"набор": s, "режим не отмечен": _nums(x.loc[fm.map(len) == 0, "num"]),
                  "режим + «?»": _nums(x.loc[fm.map(lambda v: len(v) == 2 and "?" in v), "num"]),
                  "находка не отмечена": _nums(x.loc[ff.map(len) == 0, "num"]),
                  "«Опухоль» и «Другое» вместе": _nums(x.loc[ff.map(lambda v: v == {"Опухоль", "Другое"}), "num"]),
                  "«Норма» и «Другое» вместе": _nums(x.loc[ff.map(lambda v: v == {"Норма", "Другое"}), "num"])})
    out += [_md_table(pd.DataFrame(t).set_index("набор")), "",
            f"- снимков, у которых режим не отметил никто: {_nums(m.loc[m['mode_missing'], 'num'])}",
            f"- снимков, у которых находку не отметил никто: {_nums(m.loc[m['finding_missing'], 'num'])}", ""]

    # 4. согласие двух наборов
    d = m[m["n_reviewers"] == 2]
    pa, pb = d["mode_A"].map(as_set).map(primary_mode), d["mode_B"].map(as_set).map(primary_mode)
    fa, fb = d["finding_A"].map(as_set).map(primary_finding), d["finding_B"].map(as_set).map(primary_finding)
    okm = ~pa.isin(["пусто", "несколько"]) & ~pb.isin(["пусто", "несколько"])
    okf = (fa != "пусто") & (fb != "пусто")
    susp_a, susp_b = d["finding_A"].str.contains("Опухоль"), d["finding_B"].str.contains("Опухоль")
    path_a, path_b = d["finding_A"].map(as_set).map(lambda v: bool(v - {"Норма"})), d["finding_B"].map(as_set).map(lambda v: bool(v - {"Норма"}))
    out += [f"## Согласие двух наборов (листы 1–4, {len(d)} снимков)", "",
            f"- режим: совпал у {int((pa[okm] == pb[okm]).sum())} из {int(okm.sum())} ({(pa[okm] == pb[okm]).mean():.0%}), каппа Коэна {cohen_kappa_score(pa[okm], pb[okm]):.2f}",
            f"- находка (три варианта и «Опухоль или другое»): совпала у {int((fa[okf] == fb[okf]).sum())} из {int(okf.sum())} ({(fa[okf] == fb[okf]).mean():.0%}), "
            f"каппа {cohen_kappa_score(fa[okf], fb[okf]):.2f}",
            f"- «есть патология / норма»: совпало у {int((path_a[okf] == path_b[okf]).sum())} из {int(okf.sum())} ({(path_a[okf] == path_b[okf]).mean():.0%}), "
            f"каппа {cohen_kappa_score(path_a[okf], path_b[okf]):.2f}",
            f"- отметка «Опухоль» (одна или вместе с «Другое»): набор A — {int(susp_a.sum())} снимков, набор B — {int(susp_b.sum())}, оба — {int((susp_a & susp_b).sum())}", "",
            "Режим, A (строки) × B (столбцы):", "", _md_table(pd.crosstab(pa.rename("A"), pb.rename("B"))), "",
            "Находка, A (строки) × B (столбцы):", "", _md_table(pd.crosstab(fa.rename("A"), fb.rename("B"))), ""]

    # 5. распределения по ответам каждого набора
    per = []
    for s in SETS:
        x = m[m[f"seen_{s}"]]
        per.append(pd.Series(x[f"mode_{s}"].map(as_set).map(primary_mode).value_counts(), name=f"набор {s}"))
    mode_tab = pd.concat(per, axis=1).fillna(0).astype(int).reindex([*MODE, "несколько", "пусто"]).dropna(how="all").fillna(0).astype(int)
    per = []
    for s in SETS:
        x = m[m[f"seen_{s}"]]
        per.append(pd.Series(x[f"finding_{s}"].map(as_set).map(primary_finding).value_counts(), name=f"набор {s}"))
    find_tab = pd.concat(per, axis=1).fillna(0).astype(int).reindex([*FINDING, "Опухоль или другое", "несколько", "пусто"]).dropna(how="all").fillna(0).astype(int)
    out += ["## Режим съемки по ответам каждого набора", "", _md_table(mode_tab, "режим"), "",
            "## Находки по ответам каждого набора", "", _md_table(find_tab, "находка"), ""]

    # 6. сводно по снимкам
    out += ["## Сводно по снимкам (объединение ответов)", "",
            f"- КТ отметил хотя бы один проверяющий: {int(m['ct_any'].sum())} снимков",
            f"- T1 с контрастом отметил хотя бы один: {int(m['t1c_any'].sum())}; все проверяющие: {int(m['t1c_all'].sum())}",
            f"- «Норма» у всех проверяющих: {int(m['norm_all'].sum())}",
            f"- «Опухоль» отметил хотя бы один: {int(m['tumor_any'].sum())}",
            f"- «Другое» отметил хотя бы один: {int(m['other_any'].sum())}",
            f"- режим «?» отметил хотя бы один: {int(m['unsure_mode'].sum())}",
            f"- «Опухоль» отметили оба проверяющих (листы 1–4): {_nums(m.loc[(m['n_reviewers'] == 2) & m['finding_A'].str.contains('Опухоль') & m['finding_B'].str.contains('Опухоль'), 'num'])}",
            f"- T1 с контрастом и отметка «Опухоль» хотя бы у одного: {_nums(m.loc[m['t1c_any'] & m['tumor_any'], 'num'])}", "",
            "По частям (train / val / test): " + "; ".join(
                f"{name} {'/'.join(str(int(v)) for v in m.loc[m[col], 'split'].value_counts().reindex(PARTS).fillna(0))}"
                for name, col in (("КТ", "ct_any"), ("«Опухоль»", "tumor_any"), ("«Другое»", "other_any"), ("режим «?»", "unsure_mode"))) + ".", "",
            "## Сколько снимков останется при разных правилах отбора", "", _md_table(scenarios(m)), "",
            "Принятое правило выделено. Решение по каждому снимку — `data/processed/review_decisions.csv` (`src/make_review_decisions.py`); "
            "исключения применяются к замороженному сплиту в `src/make_splits.py`.", ""]

    # 7. T1 с контрастом
    t1c = m[m["t1c_any"]]
    out += [f"## Снимки, отмеченные как T1 с контрастом ({len(t1c)})", "",
            _md_table(t1c.assign(режим=t1c["mode_all"], находка=t1c["finding_all"], проверяющих=t1c["n_reviewers"], часть=t1c["split"],
                                 подпись=t1c["notes"]).set_index("num")[["часть", "проверяющих", "режим", "находка", "подпись"]], "№"), ""]

    # 8. согласованность внутри группы одного пациента
    rows = []
    for s in SETS:
        x = m[m[f"seen_{s}"]].assign(pm=lambda d_: d_[f"mode_{s}"].map(as_set).map(primary_mode))
        g = x.groupby("patient_group_id").agg(n=("num", "size"), modes=("pm", lambda v: "/".join(sorted(set(v)))), nums=("num", lambda v: _nums(v)),
                                              k=("pm", "nunique"))
        g = g[(g["n"] > 1)]
        rows.append({"набор": s, "групп с ≥ 2 снимками": len(g), "из них с разным режимом": int((g["k"] > 1).sum()),
                     "примеры (снимки: режимы)": "; ".join(f"{r.nums}: {r.modes}" for r in g[g["k"] > 1].head(6).itertuples())})
    out += ["## Согласованность внутри группы одного пациента", "",
            "В группе — копии одного снимка и соседние срезы; режим у них, как правило, один. Разный режим внутри группы у одного "
            "проверяющего — повод пересмотреть отметки (не доказательство ошибки).", "", _md_table(pd.DataFrame(rows).set_index("набор")), ""]

    # 9. подписи
    nk = pd.DataFrame([{"kind": n.kind, "num": k} for n in notes.itertuples(index=False) for k in parse_nums(n.nums)])
    kind_tab = nk.groupby("kind")["num"].agg(снимков="nunique", номера=lambda v: _nums(set(v)))
    out += ["## Подписи проверяющих", "", _md_table(kind_tab, "о чем подпись"), "",
            "Полный текст подписей — `reports/doctor/no_tumor_review/marks_notes.csv`.", "",
            f"## Отметки, которые стоит пересмотреть ({len(doubts)})", "",
            "Оценка не экспертная: это места, где отметка расходится с тем, что видно на снимке, или с подписью того же проверяющего.", "",
            _md_table(doubts.rename(columns={"set": "набор", "num": "№", "mark": "отметка", "doubt": "сомнение"}).set_index("№"), "№"), ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheets-dir", type=Path, default=Path("reports/doctor/no_tumor_review"))
    ap.add_argument("--report", type=Path, default=Path("reports/doctor_marks_report.md"))
    args = ap.parse_args()
    d = args.sheets_dir
    raw = pd.read_csv(d / "marks_raw.csv", keep_default_na=False)
    overrides = pd.read_csv(d / "marks_overrides.csv", keep_default_na=False)
    notes = pd.read_csv(d / "marks_notes.csv", keep_default_na=False)
    answers = pd.read_csv(d / "answers.csv", keep_default_na=False)
    doubts = pd.read_csv(d / "marks_doubts.csv", keep_default_na=False)

    m = merge(raw, overrides, notes, answers)
    m.to_csv(d / "marks_merged.csv", index=False, encoding="utf-8-sig")
    filled = answers.drop(columns=[c for c in ("n_reviewers",) if c in answers.columns]).copy()
    filled["mri_type"], filled["finding"], filled["comment"] = m["mode_all"].to_numpy(), m["finding_all"].to_numpy(), m["notes"].to_numpy()
    filled["n_reviewers"] = m["n_reviewers"].to_numpy()
    filled.to_csv(d / "answers.csv", index=False, encoding="utf-8-sig")
    args.report.write_text(make_report(m, raw, overrides, notes, doubts), encoding="utf-8")
    print(scenarios(m).to_string())
    print(f"\nотчет: {args.report}; таблицы: {d / 'marks_merged.csv'}, {d / 'answers.csv'}")


if __name__ == "__main__":
    main()
