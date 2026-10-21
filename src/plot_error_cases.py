"""Листы ошибок системы на test для разбора врачом.

Берутся ответы финальной оценки (reports/eval_test/predictions.csv) и замороженная система (models/final/).
Отбор (seed 42):
  A — все пропущенные опухоли и все ложные тревоги;
  B — снимки с флагом «требует внимания врача»: все, где ответ неверен, и 10 случайных, где ответ верен;
  C — неверный тип опухоли без флага: по 2 среза у каждого пациента с повторяющимися ошибками и все одиночные.
Каждый случай — строка листа: снимок | обводка (модель — красным, разметка Figshare — зеленым) | тепловая
карта Grad-CAM | ответ и клетки для отметок врача. Ответы вносятся в answers.csv.

Результат: reports/doctor/error_review/page_XX.png, cases.csv (что на листах), answers.csv (пустые колонки).

Пример:
    python -m src.plot_error_cases
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.inference import load_predictor  # noqa: E402
from src.io_utils import read_image  # noqa: E402
from src.labels import CLS_RU  # noqa: E402

SEED = 42
PER_PAGE = 4
CAT = {"A": "пропуск или ложная тревога", "B": "флаг «требует внимания»", "C": "неверный тип опухоли"}
SOURCE_RU = {"figshare": "Figshare", "figshare_crop": "Figshare, другое кадрирование", "other": "другой источник"}


def select(d: pd.DataFrame) -> pd.DataFrame:
    tu = d["cls"] != "no_tumor"
    wrong = d["answer"] != d["cls"]
    a = d[(tu & ~d["present"]) | (~tu & d["present"])].assign(category="A")
    rest = d.drop(a.index)
    fl = rest[rest["flag"]]
    b = pd.concat([fl[fl["answer"] != fl["cls"]],
                   fl[fl["answer"] == fl["cls"]].sample(n=min(10, int((fl["answer"] == fl["cls"]).sum())), random_state=SEED)]).assign(category="B")
    rest = rest.drop(fl.index)
    wt = rest[tu.loc[rest.index] & rest["present"] & wrong.loc[rest.index]]
    c = wt.groupby("group", group_keys=False).apply(lambda x: x.sample(n=min(2, len(x)), random_state=SEED)).assign(category="C")
    out = pd.concat([a, b, c])
    return out.sort_values(["category", "group", "path"]).reset_index(drop=True)


def text_block(ax, k: int, r) -> None:
    ax.axis("off")
    truth = CLS_RU[r.cls]
    src = SOURCE_RU.get(r.source, r.source) if r.cls != "no_tumor" else f"режим {r.nt_mode}, {r.nt_finding}"
    lines = [(f"Случай {k} · {CAT[r.category]}", 11, "bold"),
             (f"Правильный ответ: {truth}", 10, "normal"),
             (f"({src}; пациент {r.group})", 8.5, "normal"),
             (f"Ответ системы: {CLS_RU[r.answer]} (вероятность этого класса у классификатора {100 * r.confidence:.0f}%)", 9.5, "normal"),
             (f"Вероятность опухоли {100 * r.p_tumor:.0f}%; область {int(r.mask_area)} px", 8.5, "normal"),
             (f"Флаг: {r.flag_reason}" if r.flag else "Флага нет", 8.5, "normal"),
             ("", 6, "normal"),
             ("Метка верна:        да  /  нет  /  ?", 9.5, "normal"),
             ("Опухоль видна:     да  /  нет  /  только на соседних", 9.5, "normal"),
             ("Качество снимка:  норма  /  артефакт  /  плохое", 9.5, "normal"),
             ("Ошиблись бы вы:   да  /  нет", 9.5, "normal"),
             ("Причина: ______________________________", 9.5, "normal")]
    y = 0.98
    for s, size, weight in lines:
        ax.text(0.0, y, s, fontsize=size, fontweight=weight, va="top", wrap=True, transform=ax.transAxes)
        y -= 0.085 if size >= 9.5 else 0.07


def draw_page(chunk: pd.DataFrame, page: int, n_pages: int, roots: dict, fig_root: Path, predictor):
    """Лист из PER_PAGE случаев. Снимки рисуются без интерполяции: в PDF они встают в исходном разрешении."""
    fig, axs = plt.subplots(PER_PAGE, 4, figsize=(17, 4.4 * PER_PAGE), gridspec_kw={"width_ratios": [1, 1, 1, 1.45]})
    for row, r in enumerate(chunk.itertuples()):
        img = read_image(roots[r.root] / r.path)
        res = predictor.predict(img, heatmap=True)
        for j in range(3):
            axs[row, j].imshow(img, cmap="gray", interpolation="none")
            axs[row, j].set_xticks([]); axs[row, j].set_yticks([])
        axs[row, 0].set_title(f"снимок (файл case_{r.case:02d})", fontsize=9)
        if r.has_mask:
            true = read_image(fig_root / r.mask) > 0
            axs[row, 1].contour(true, [0.5], colors="#00c853", linewidths=1.3)
        if res["mask"] is not None and res["mask"].any():
            axs[row, 1].contour(res["mask"], [0.5], colors="#ff1744", linewidths=1.3)
        axs[row, 1].set_title("обводка: модель — красным" + (", врач — зеленым" if r.has_mask else ""), fontsize=9)
        axs[row, 2].imshow(res["heatmap"], cmap="jet", alpha=0.4, vmin=0, vmax=1, interpolation="none")
        axs[row, 2].set_title(f"куда смотрел классификатор ({CLS_RU[res['heatmap_class']]})", fontsize=9)
        text_block(axs[row, 3], r.case, r)
    for row in range(len(chunk), PER_PAGE):
        for j in range(4):
            axs[row, j].axis("off")
    fig.suptitle(f"Ошибки системы на тестовых снимках · лист {page} из {n_pages} · отметки — в файле «Ответы_врача.xlsx»", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, default=Path("reports/eval_test/predictions.csv"))
    ap.add_argument("--data-root", type=Path, default=Path("mrt"))
    ap.add_argument("--fig-root", type=Path, default=Path("data/external/figshare/png"))
    ap.add_argument("--out", type=Path, default=Path("reports/doctor/error_review"))
    ap.add_argument("--pdf", type=Path, default=None, help="дополнительно сохранить листы в векторный PDF")
    args = ap.parse_args()

    d = pd.read_csv(args.predictions, keep_default_na=False)
    for c in ("present", "flag", "has_mask"):
        d[c] = d[c].astype(str).str.lower() == "true"
    cases = select(d)
    cases.insert(0, "case", np.arange(1, len(cases) + 1))
    cases.insert(1, "page", (cases["case"] - 1) // PER_PAGE + 1)
    args.out.mkdir(parents=True, exist_ok=True)
    roots = {"figshare": args.fig_root, "case": args.data_root}
    predictor = load_predictor()

    pdf = PdfPages(args.pdf) if args.pdf else None
    n_pages = int(cases["page"].max())
    for page, chunk in cases.groupby("page"):
        fig = draw_page(chunk, page, n_pages, roots, args.fig_root, predictor)
        fig.savefig(args.out / f"page_{page:02d}.png", dpi=90)
        if pdf is not None:
            pdf.savefig(fig)
        plt.close(fig)
        print(f"лист {page}: случаи {chunk['case'].min()}–{chunk['case'].max()}", flush=True)
    if pdf is not None:
        pdf.close()

    keep = ["case", "page", "category", "path", "root", "mask", "has_mask", "group", "cls", "answer", "confidence", "p_tumor", "mask_area", "flag", "flag_reason",
            "source", "nt_mode", "nt_finding", "dice"]
    cases[keep].to_csv(args.out / "cases.csv", index=False, encoding="utf-8-sig")
    ans = cases[["case", "page", "category", "path", "group", "cls", "answer"]].copy()
    for c in ("label_ok", "tumor_visible", "quality", "doctor_would_miss", "cause", "comment"):
        ans[c] = ""
    ans.to_csv(args.out / "answers.csv", index=False, encoding="utf-8-sig")
    print(f"случаев {len(cases)} (" + ", ".join(f"{k}: {int((cases['category'] == k).sum())}" for k in CAT) + f"), листов {cases['page'].max()} → {args.out}")


if __name__ == "__main__":
    main()
