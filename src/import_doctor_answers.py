"""Разбор ошибок с врачом: ответы врача из Ответы_врача.xlsx сводятся со случаями.

Вход: заполненная таблица врача (лист «Ответы»; таблица с комментариями врача в репозиторий не входит) и reports/doctor/error_review/cases.csv (57 случаев с test).
Выход:
  doctorReview/error_review_answers.csv  — ответы как есть + расшифровка + комментарии (внутренний файл);
  reports/doctor/error_review/summary.md — сводка в числах, без комментариев;
  reports/doctor/error_review/causes.png — причины ошибок (рисунок к слайду 13).

Расшифровка ответов (сначала — по комментариям; 19 октября 2026 г. подтверждена ответами врача на 4 вопроса):
  - группа C («неверный тип»), причина «другая болезнь» — «система назвала другой тип», а не «на снимке другое
    заболевание» (подтверждено); причина в этом случае выводится из остальных колонок (см. ниже);
  - «Метка верна? — нет» в случаях 3, 16, 23 относилась к ответу системы, метка набора верна (подтверждено);
    случай 36 («не уверен») — сомнение относится к лишним областям обводки системы, метка верна;
    случай 2 (пусто) — врач в комментарии описывает менингиому, метка верна;
  - случаи 21 и 57 — по мнению врача, подпись набора ошибочна (подтверждено): 21 — менингиома, а не глиома
    (четкие контуры, широкое основание, прилегает к оболочкам и кости); 57 — глиома, а не менингиома.
    Метки test не правим: в сводке приводятся оба числа.
Итоговая причина — первая подходящая по порядку:
  1) возможная ошибка метки; 2) снимок вне условий применения (не МРТ T1 с контрастом);
  3) плохое качество; 4) трудно и для врача («Вы бы ошиблись?» — да или не уверен, либо причина
  «очень маленькая опухоль» / «опухоль не видна на одном срезе»); 5) ошибка системы: метка верна, снимок
  подходящий, врач бы не ошибся.
В группе B случаи, где ответ системы совпал с меткой, — «верный ответ, сработал только флаг»: это не ошибка.

Пример:
    python -m src.import_doctor_answers --xlsx "doctorReview/Ответы_врача_заполнено.xlsx"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import slide_style  # noqa: E402
from src.labels import CLS_RU  # noqa: E402

COLS = ["case", "page", "grp", "label_ru", "answer_ru", "source_note", "mode", "label_ok", "visible", "quality",
        "would_miss", "cause_raw", "comment"]
T1C = "МРТ T1 с контрастом"
LABEL_OK_FROM_COMMENT = {2: "да", 3: "да", 16: "да", 23: "да", 36: "да"}  # см. описание модуля
DOCTOR_LABEL = {21: "meningioma", 57: "glioma"}  # подпись набора, по мнению врача, ошибочна; верный тип — этот
POSSIBLE_LABEL_ERROR = set(DOCTOR_LABEL)
# вопросы по расшифровке и ответы врача (19 октября 2026 г.)
DOCTOR_QA = [
    ("Группа C", "«Другая болезнь» — это «система назвала другой тип опухоли», а не «на снимке другое заболевание»?",
     "Да: ответ системы (тип) неверен, подпись набора верна."),
    ("Случаи 3, 16, 23", "«Метка верна? — нет» относилась к ответу системы, а подпись набора верна?",
     "Да: случай 3 — опухоли нет, подпись верна; 16 и 23 — подпись «менингиома» верна."),
    ("Случай 21", "На снимке менингиома, а подпись «глиома» в наборе ошибочна?",
     "Да, скорее менингиома: четкие контуры, широкое основание, прилегает к оболочкам и внутренней поверхности кости."),
    ("Случай 57", "Считать подпись «менингиома» в наборе ошибкой?", "Да."),
]
CAUSES = {
    "label": "ошибка подписи в наборе (по мнению врача)",
    "scope": "снимок не МРТ T1 с контрастом",
    "quality": "плохое качество снимка",
    "hard": "трудно и для врача по одному срезу",
    "system": "ошибка системы: врач бы не ошибся",
}
GROUPS = {"A": "пропуск или ложная тревога", "B": "флаг «требует внимания»", "C": "неверный тип опухоли"}


def final_cause(r) -> str:
    if r.case in POSSIBLE_LABEL_ERROR:
        return "label"
    if r["mode"] != T1C:
        return "scope"
    if r.quality == "плохое":
        return "quality"
    if r.would_miss in ("да", "не уверен") or r.cause_raw in ("очень маленькая опухоль", "опухоль не видна на одном срезе"):
        return "hard"
    return "system"


def plot_causes(err: pd.DataFrame, n_flag_only: int, path: Path) -> None:
    order = list(CAUSES)[::-1]
    counts = err["cause"].value_counts().reindex(order).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(6.0, 2.9), dpi=200)
    ax.barh(range(len(order)), counts.values, color=slide_style.MARK, height=0.6)
    for i, v in enumerate(counts.values):
        ax.text(v + 0.4, i, str(v), va="center", fontsize=10.5, color="#0b0b0b")
    ax.set_yticks(range(len(order)), [CAUSES[k] for k in order], fontsize=10, color="#0b0b0b")
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"Причины {len(err)} ошибок по разбору врача", fontsize=11, color="#0b0b0b", loc="left")
    fig.text(0.01, 0.02, f"Еще {n_flag_only} случаев — ответ верный, но стоит пометка «требует внимания врача»: это не ошибка.",
             fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", type=Path, default=Path("doctorReview/Ответы_врача_заполнено.xlsx"))
    ap.add_argument("--cases", type=Path, default=Path("reports/doctor/error_review/cases.csv"))
    ap.add_argument("--out-internal", type=Path, default=Path("doctorReview/error_review_answers.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/doctor/error_review"))
    ap.add_argument("--metrics", type=Path, default=Path("reports/eval_test/metrics.json"))
    args = ap.parse_args()
    slide_style.apply()  # шрифт и оформление — как у рисунков к слайдам

    a = pd.read_excel(args.xlsx, sheet_name="Ответы")
    assert len(a.columns) == len(COLS), f"ожидалось {len(COLS)} колонок, в таблице {len(a.columns)}"
    a.columns = COLS
    c = pd.read_csv(args.cases, keep_default_na=False)
    m = c.merge(a, on="case", validate="one_to_one")
    assert len(m) == len(c) == 57
    assert (m["label_ru"] == m["cls"].map(CLS_RU)).all() and (m["answer_ru"] == m["answer"].map(CLS_RU)).all(), "таблица врача не совпадает со случаями"
    for col in ("mode", "label_ok", "visible", "quality", "would_miss", "cause_raw", "comment"):
        m[col] = m[col].fillna("").astype(str).str.strip()

    m["label_ok_raw"] = m["label_ok"]
    m["label_ok"] = [LABEL_OK_FROM_COMMENT.get(k, v) for k, v in zip(m.case, m.label_ok)]
    m.loc[m.case.isin(POSSIBLE_LABEL_ERROR), "label_ok"] = "нет (по мнению врача)"
    m["doctor_label"] = m["case"].map(DOCTOR_LABEL).fillna("")
    m["answer_right"] = m["answer"] == m["cls"]
    m["flag_only"] = (m["category"] == "B") & m["answer_right"]
    m["cause"] = [("flag_only" if fo else final_cause(r)) for fo, (_, r) in zip(m.flag_only, m.iterrows())]
    m["cause_ru"] = m["cause"].map({**CAUSES, "flag_only": "верный ответ, только флаг"})
    m["type_mixed"] = (m["category"] == "C") & (m["cause_raw"] == "другая болезнь")
    args.out_internal.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.out_internal, index=False, encoding="utf-8")

    err = m[~m.flag_only]
    flag_only = int(m.flag_only.sum())
    plot_causes(err, flag_only, args.out_dir / "causes.png")

    def cnt(d: pd.DataFrame, cond) -> int:
        return int(cond(d).sum())

    def pct_ru(x: float) -> str:
        return f"{100 * x:.1f}".replace(".", ",") + "%"

    # «оба числа»: тип среди найденных опухолей по подписям набора и с поправкой врача (метки test не правим)
    tc = json.loads(args.metrics.read_text(encoding="utf-8"))["cls"]["type_correct_among_found"]
    tk, tn = int(tc["k"]), int(tc["n"])
    fixed = sum(int((m.loc[m.case == k, "answer"] == v).any()) for k, v in DOCTOR_LABEL.items())

    lines = ["# Разбор ошибок системы врачом: сводка", "",
             f"Случаев: {len(m)} с test (листы `page_01…15.png`); ответы врача получены 17 октября 2026 г., "
             "расшифровка части ответов подтверждена врачом 19 октября. "
             f"Ответ системы неверен в {len(err)} случаях, еще в {flag_only} ответ верный, но стоит флаг «требует внимания». "
             "Расшифровка ответов и порядок выбора итоговой причины — в описании `src/import_doctor_answers.py`. "
             "Метки test по итогам разбора не правились.", "",
             "## Итоговая причина по группам", "",
             "| Причина | " + " | ".join(f"{k}: {v}" for k, v in GROUPS.items()) + " | Всего |",
             "| --- | --- | --- | --- | --- |"]
    for key, name in {**CAUSES, "flag_only": "верный ответ, только флаг"}.items():
        row = [cnt(m[m.category == g], lambda d: d.cause == key) for g in GROUPS]
        lines.append(f"| {name} | " + " | ".join(map(str, row)) + f" | {sum(row)} |")
    lines.append("| **всего** | " + " | ".join(str(int((m.category == g).sum())) for g in GROUPS) + f" | {len(m)} |")

    lines += ["", "## Ответы врача по колонкам (все 57 случаев)", ""]
    for col, name in (("mode", "Режим снимка"), ("label_ok", "Метка верна? (после расшифровки)"), ("visible", "Опухоль видна на срезе («нет» — снимок без опухоли)"),
                      ("quality", "Качество снимка"), ("would_miss", "Врач бы ошибся по одному срезу")):
        vc = m[col].replace("", "нет ответа").value_counts()
        lines.append(f"- {name}: " + ", ".join(f"{k} — {v}" for k, v in vc.items()) + ".")
    out_scope = m[m["mode"] != T1C]
    lines += ["", "## Наблюдения", "",
              f"- Не МРТ T1 с контрастом — {len(out_scope)} снимков, все из других источников (не Figshare): "
              + ", ".join(f"случай {r.case} — {r['mode']}" for _, r in out_scope.iterrows())
              + f". Флаг стоял на {int(out_scope.flag.astype(str).str.lower().eq('true').sum())} из них.",
              f"- Врач бы ошибся или не уверен по одному срезу: {cnt(m, lambda d: d.would_miss.isin(['да', 'не уверен']))} случаев; "
              f"среди них опухоль гипофиза, названная менингиомой, — {cnt(m, lambda d: d.would_miss.isin(['да', 'не уверен']) & (d.cls == 'pituitary') & (d.answer == 'meningioma'))}.",
              f"- Неверный тип (группа C: случаев — {cnt(m, lambda d: d.category == 'C')}, пациентов — {m[m.category == 'C'].group.nunique()}): "
              f"врач бы не ошибся в {cnt(m, lambda d: (d.category == 'C') & (d.would_miss == 'нет'))}; "
              f"снимок подходящий и хорошего качества в {cnt(m, lambda d: (d.category == 'C') & (d['mode'] == T1C) & (d.quality == 'норма'))}.",
              f"- Пропуски и ложные тревоги (группа A, 6 случаев): " + "; ".join(f"случай {r.case} — {r.cause_ru}" for _, r in m[m.category == 'A'].iterrows()) + ".",
              f"- Ошибка подписи в наборе, по мнению врача: "
              + "; ".join(f"случай {k} — {CLS_RU[v]}, а не {CLS_RU[m.loc[m.case == k, 'cls'].iloc[0]]}" for k, v in sorted(DOCTOR_LABEL.items()))
              + f". Оба числа: тип верен у {tk} из {tn} найденных опухолей по подписям набора ({pct_ru(tk / tn)}) "
              f"и у {tk + fixed} из {tn} с поправкой врача ({pct_ru((tk + fixed) / tn)})."]
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    qa = ["# Вопросы врачу по расшифровке ответов и ответы (19 октября 2026 г.)", ""]
    for who, q, ans in DOCTOR_QA:
        qa += [f"- **{who}.** {q}", f"  Ответ: {ans}"]
    (args.out_internal.parent / "questions_for_doctor.md").write_text("\n".join(qa) + "\n", encoding="utf-8")
    print("\n".join(lines + [""] + qa))


if __name__ == "__main__":
    main()
