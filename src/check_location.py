"""Проверка идеи команды (2026-10-17): отличаются ли типы опухоли расположением — менингиома прилегает
к краю (оболочки, кость), глиома лежит внутри мозга, опухоль гипофиза — в центре у основания черепа.

Определение записано до первого запуска (коммит скрипта — до прогона):
- Данные: срезы Figshare с эталонной обводкой из общего пула, части train (2229) и val (391). Тест не открывается.
- Все считается в координатах модели: 256×256 после src/preprocess.py (кэш pool_256).
- Голова: снимок > 10 (из 255) → закрытие 7×7 → заполнение дыр → наибольшая связная область.
  Масштаб — радиус головы r = sqrt(площадь головы / π).
- Признаки обводки:
    d_edge   — наименьшее расстояние от опухоли до края головы / r («прилегает к краю»);
    depth    — расстояние от центра масс опухоли до края головы / r;
    dx, dy   — смещение центра опухоли от центра головы: по горизонтали по модулю, по вертикали (вниз +) / r;
    area     — площадь опухоли / площадь головы;
    solidity — площадь / площадь выпуклой оболочки; compact — 4πA / P²;
    edge     — резкость края: (яркость полосы 3 px внутри обводки − полосы 3 px снаружи) / 255;
    core     — «темный центр»: (яркость внешней половины опухоли − внутренней) / 255;
    bright   — средняя яркость опухоли / медиана яркости головы.
- Главная мера: менингиома против глиомы по одному признаку d_edge («ближе к краю → менингиома»).
  Порог — на train (наибольшая сбалансированная точность), качество — на val: сбалансированная точность и AUC;
  95% интервал — бутстреп по пациентам (колонка group, 2000 повторов, seed 42).
- Дополнительно:
    (а) логистическая регрессия на всех признаках (стандартизация, class_weight="balanced", C = 1), 3 типа:
        обучение на train, оценка на val — сбалансированная точность, полнота по типам;
    (б) то же на обводках сегментатора системы (models/final/segmenter.pt) на val; срезы, где область
        не найдена (площадь ≤ 50 px), считаются отдельно;
    (в) сравнение с типом, который назвал классификатор системы на тех же срезах val
        (reports/eval_val/predictions.csv), и на его ошибках; простое среднее вероятностей двух моделей
        (веса 1:1, без подбора) — только как описание.
Модели системы не меняются.

Пример:
    python -m src.check_location
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy import ndimage  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.dataset import PreprocCache, to_tensor  # noqa: E402
from src.inference import _load  # noqa: E402
from src import slide_style  # noqa: E402
from src.labels import CLS_RU  # noqa: E402

TYPES = ["glioma", "meningioma", "pituitary"]
FEATURES = ["d_edge", "depth", "dx", "dy", "area", "solidity", "compact", "edge", "core", "bright"]
# цвета типов — первые три слота проверенной палитры (синий, оранжевый, бирюзовый); одни и те же во всех рисунках
COLORS = {"glioma": "#2a78d6", "meningioma": "#eb6834", "pituitary": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
MIN_AREA = 50


def head_mask(img: np.ndarray) -> np.ndarray:
    m = (img > 10).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n == 0:
        return m.astype(bool)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def features(img: np.ndarray, tumor: np.ndarray) -> dict:
    """Признаки одной обводки (определения — в описании модуля)."""
    head = head_mask(img)
    r = np.sqrt(head.sum() / np.pi)
    dist = cv2.distanceTransform(head.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.nonzero(tumor)
    hy, hx = np.nonzero(head)
    cy, cx = ys.mean(), xs.mean()
    contours, _ = cv2.findContours(tumor.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c, True) for c in contours)
    hull = cv2.convexHull(np.vstack(contours))
    k = np.ones((3, 3), np.uint8)
    t8 = tumor.astype(np.uint8)
    inner = (t8 - cv2.erode(t8, k, iterations=3)).astype(bool)
    outer = (cv2.dilate(t8, k, iterations=3) - t8).astype(bool) & head
    tdist = cv2.distanceTransform(t8, cv2.DIST_L2, 5)
    half = tdist > np.median(tdist[tumor])
    f = img.astype(np.float32)
    area = tumor.sum()
    return {
        "d_edge": float(dist[tumor].min() / r),
        "depth": float(dist[int(round(cy)), int(round(cx))] / r),
        "dx": float(abs(cx - hx.mean()) / r),
        "dy": float((cy - hy.mean()) / r),
        "area": float(area / head.sum()),
        "solidity": float(area / max(cv2.contourArea(hull), 1.0)),
        "compact": float(4 * np.pi * area / max(perim, 1.0) ** 2),
        "edge": float((f[inner].mean() - f[outer].mean()) / 255) if inner.any() and outer.any() else 0.0,
        "core": float((f[tumor & ~half].mean() - f[half].mean()) / 255) if half.any() and (tumor & ~half).any() else 0.0,
        "bright": float(f[tumor].mean() / max(np.median(f[head]), 1.0)),
    }


def group_boot(df: pd.DataFrame, stat, n: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = df["group"].unique()
    idx = {g: np.flatnonzero(df["group"].to_numpy() == g) for g in groups}
    vals = []
    for _ in range(n):
        take = np.concatenate([idx[g] for g in rng.choice(groups, len(groups))])
        sub = df.iloc[take]
        try:
            vals.append(stat(sub))
        except ValueError:  # в повторе один класс
            continue
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def best_threshold(x: np.ndarray, is_men: np.ndarray) -> float:
    """Порог «x ≤ t → менингиома» с наибольшей сбалансированной точностью (кандидаты — середины)."""
    u = np.unique(x)
    cands = (u[:-1] + u[1:]) / 2
    scores = [balanced_accuracy_score(is_men, x <= t) for t in cands]
    return float(cands[int(np.argmax(scores))])


def seg_masks(cache: PreprocCache, paths: list[str], seg_path: Path) -> dict[str, np.ndarray]:
    model, _ = _load(seg_path, torch.device("cpu"))
    out = {}
    with torch.no_grad():
        for i in range(0, len(paths), 16):
            batch = paths[i:i + 16]
            x = torch.stack([to_tensor(cache.image(p)) for p in batch])
            prob = torch.sigmoid(model(x).float())[:, 0].numpy()
            for p, m in zip(batch, prob):
                out[p] = m > 0.5
    return out


def plot_distribution(val: pd.DataFrame, thr: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=200)
    rng = np.random.default_rng(42)
    for i, t in enumerate(TYPES):
        x = val.loc[val.cls == t, "d_edge"].to_numpy()
        y = i + rng.uniform(-0.18, 0.18, len(x))
        ax.scatter(x, y, s=9, color=COLORS[t], alpha=0.55, linewidths=0)
        med = np.median(x)
        ax.plot([med, med], [i - 0.3, i + 0.3], color=INK, lw=2, solid_capstyle="round")
        ax.text(med, i + 0.36, f"медиана {med:.2f}", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.axvline(thr, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(thr, 2.75, f" порог {thr:.2f}", ha="left", va="top", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(TYPES)), [f"{CLS_RU[t]} ({(val.cls == t).sum()})" for t in TYPES], fontsize=9, color=INK)
    ax.set_ylim(-0.6, 2.85)
    ax.set_xlabel("расстояние от опухоли до края головы (доля радиуса головы)", fontsize=9, color=MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.set_title("Где лежит опухоль: срезы Figshare, проверочная часть", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_examples(cache: PreprocCache, val: pd.DataFrame, path: Path) -> None:
    """По одному срезу каждого типа (ближайший к медиане d_edge своего типа) с кратчайшим отрезком до края головы."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4), dpi=200)
    for ax, t in zip(axes, TYPES):
        sub = val[val.cls == t]
        row = sub.iloc[int(np.argmin(np.abs(sub.d_edge - sub.d_edge.median())))]
        img, tumor = cache.image(row.path), cache.mask(row.path) > 0
        head = head_mask(img)
        ax.imshow(img, cmap="gray", interpolation="none")
        for m, c in ((head, "#c3c2b7"), (tumor, COLORS[t])):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cc in cs:
                ax.plot(cc[:, 0, 0], cc[:, 0, 1], color=c, lw=1.2)
        # кратчайший отрезок: точка опухоли, ближайшая к краю, и ближайшая к ней точка вне головы
        dist, (iy, ix) = ndimage.distance_transform_edt(head, return_indices=True)
        ys, xs = np.nonzero(tumor)
        j = int(np.argmin(dist[ys, xs]))
        y0, x0 = ys[j], xs[j]
        ax.plot([x0, ix[y0, x0]], [y0, iy[y0, x0]], color="#ffffff", lw=1.5)
        ax.set_title(f"{CLS_RU[t]}: {row.d_edge:.2f}", fontsize=9, color=INK)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=Path("data/processed/cache"))
    ap.add_argument("--segmenter", type=Path, default=Path("models/final/segmenter.pt"))
    ap.add_argument("--val-pred", type=Path, default=Path("reports/eval_val/predictions.csv"))
    ap.add_argument("--out", type=Path, default=Path("reports/location"))
    ap.add_argument("--report", type=Path, default=Path("reports/location_report.md"))
    args = ap.parse_args()
    slide_style.apply()  # шрифт и оформление — как у рисунков к слайдам
    args.out.mkdir(parents=True, exist_ok=True)

    cache = PreprocCache(args.cache_dir, "pool", 256)
    t = cache.table
    t = t[(t.root == "figshare") & t.has_mask & t.split.isin(["train", "val"])].reset_index(drop=True)  # тест не берем
    assert set(t.split) == {"train", "val"}

    rows = []
    for r in t.itertuples():
        f = features(cache.image(r.path), cache.mask(r.path) > 0)
        rows.append({"path": r.path, "split": r.split, "cls": r.cls, "group": r.group, **f})
    feat = pd.DataFrame(rows)
    feat.to_csv(args.out / "features.csv", index=False)
    tr, va = feat[feat.split == "train"].reset_index(drop=True), feat[feat.split == "val"].reset_index(drop=True)

    # главная мера: менингиома против глиомы по d_edge
    mg_tr = tr[tr.cls.isin(["glioma", "meningioma"])]
    mg_va = va[va.cls.isin(["glioma", "meningioma"])].reset_index(drop=True)
    thr = best_threshold(mg_tr.d_edge.to_numpy(), (mg_tr.cls == "meningioma").to_numpy())

    def bacc(d: pd.DataFrame) -> float:
        return balanced_accuracy_score(d.cls == "meningioma", d.d_edge <= thr)

    def auc(d: pd.DataFrame) -> float:
        return roc_auc_score(d.cls == "meningioma", -d.d_edge)

    main_res = {"thr": thr, "bacc_train": bacc(mg_tr), "bacc_val": bacc(mg_va), "bacc_ci": group_boot(mg_va, bacc),
                "auc_val": auc(mg_va), "auc_ci": group_boot(mg_va, auc),
                "men_close": int(((mg_va.cls == "meningioma") & (mg_va.d_edge <= thr)).sum()), "n_men": int((mg_va.cls == "meningioma").sum()),
                "gli_far": int(((mg_va.cls == "glioma") & (mg_va.d_edge > thr)).sum()), "n_gli": int((mg_va.cls == "glioma").sum())}

    # (а) три типа, все признаки, эталонные обводки
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
    clf.fit(tr[FEATURES], tr.cls)
    va["loc_pred"] = clf.predict(va[FEATURES])
    proba = clf.predict_proba(va[FEATURES])
    va[[f"loc_{c}" for c in clf.classes_]] = proba

    def bacc3(d: pd.DataFrame, col: str = "loc_pred") -> float:
        return balanced_accuracy_score(d.cls, d[col])

    # (б) обводки сегментатора на val
    segm = seg_masks(cache, va.path.tolist(), args.segmenter)
    srows = []
    for r in va.itertuples():
        m = segm[r.path]
        if m.sum() > MIN_AREA:
            srows.append({"path": r.path, "found": True, **features(cache.image(r.path), m)})
        else:
            srows.append({"path": r.path, "found": False})
    sf = pd.DataFrame(srows)
    vs = va[["path", "cls", "group"]].merge(sf, on="path")
    found = vs[vs.found].reset_index(drop=True)
    found["loc_pred"] = clf.predict(found[FEATURES])
    mg_seg = found[found.cls.isin(["glioma", "meningioma"])].reset_index(drop=True)

    # (в) классификатор системы на тех же срезах
    pred = pd.read_csv(args.val_pred, keep_default_na=False)
    pred = pred[(pred.root == "figshare")][["path", "answer", "p_glioma", "p_meningioma", "p_pituitary"]]
    va = va.merge(pred, on="path", how="left")
    assert va.answer.notna().all()
    found_sys = va.answer.isin(TYPES)
    cnn = va[found_sys].reset_index(drop=True)
    p_cnn = cnn[[f"p_{c}" for c in TYPES]].to_numpy()
    p_cnn = p_cnn / p_cnn.sum(1, keepdims=True)
    p_loc = cnn[[f"loc_{c}" for c in TYPES]].to_numpy()
    cnn["mix_pred"] = np.array(TYPES)[(p_cnn + p_loc).argmax(1)]
    wrong = cnn[cnn.answer != cnn.cls]

    # отчет
    def pct(a, b):
        return f"{a} из {b} ({100 * a / b:.1f}%)" if b else "—"

    def rec_line(d: pd.DataFrame, col: str) -> str:
        return ", ".join(f"{CLS_RU[c]} {pct(int(((d.cls == c) & (d[col] == c)).sum()), int((d.cls == c).sum()))}" for c in TYPES)

    cm = confusion_matrix(va.cls, va.loc_pred, labels=TYPES)
    groups_val = va.groupby("cls").group.nunique()
    coef = pd.DataFrame(clf[-1].coef_, index=clf.classes_, columns=FEATURES).round(2)
    med = feat.groupby(["split", "cls"])[FEATURES].median().round(3)

    plot_distribution(va, thr, args.out / "d_edge_val.png")
    plot_examples(cache, va, args.out / "examples_val.png")

    lines = [
        "# Проверка: отличаются ли типы опухоли расположением (2026-10-17)",
        "",
        "Идея команды: менингиома прилегает к краю (растет из оболочек), глиома лежит внутри мозга, опухоль гипофиза — "
        "в центре у основания черепа. Проверяем по эталонным обводкам Figshare, считая признаки обводки "
        "(определения — в `src/check_location.py`, записаны до запуска). Обучение — train, оценка — val; "
        "тест не открывался. Модели системы не менялись.",
        "",
        f"Срезов: train {len(tr)} ({tr.group.nunique()} пациентов), val {len(va)} ({va.group.nunique()} пациентов; "
        + ", ".join(f"{CLS_RU[c]} — {groups_val.get(c, 0)}" for c in TYPES) + "). Интервалы — бутстреп по пациентам; "
        "пациентов в val мало, поэтому интервалы широкие.",
        "",
        "## Главная мера: «ближе к краю головы → менингиома» (менингиома против глиомы)",
        "",
        f"- Порог, подобранный на train: расстояние до края ≤ {thr:.3f} радиуса головы → менингиома.",
        f"- Сбалансированная точность: train {main_res['bacc_train']:.3f}; **val {main_res['bacc_val']:.3f}** "
        f"(95% ДИ {main_res['bacc_ci'][0]:.3f}–{main_res['bacc_ci'][1]:.3f}).",
        f"- AUC на val: {main_res['auc_val']:.3f} (95% ДИ {main_res['auc_ci'][0]:.3f}–{main_res['auc_ci'][1]:.3f}).",
        f"- На val: менингиом у края {pct(main_res['men_close'], main_res['n_men'])}; глиом глубже порога {pct(main_res['gli_far'], main_res['n_gli'])}.",
        "",
        "## Три типа по всем признакам обводки (логистическая регрессия)",
        "",
        f"- Эталонные обводки, val: сбалансированная точность **{bacc3(va):.3f}** "
        f"(95% ДИ {group_boot(va, bacc3)[0]:.3f}–{group_boot(va, bacc3)[1]:.3f}); случайный выбор — 0,333.",
        f"- Тип верен: {rec_line(va, 'loc_pred')}.",
        f"- Обводки сегментатора системы, val: область найдена у {pct(len(found), len(vs))}; на них сбалансированная точность "
        f"{bacc3(found):.3f}; тип верен: {rec_line(found, 'loc_pred')}. "
        f"Менингиома против глиомы по одному порогу: {balanced_accuracy_score(mg_seg.cls == 'meningioma', mg_seg.d_edge <= thr):.3f}.",
        "",
        "Матрица ошибок (эталонные обводки, val; строки — истина, столбцы — ответ):",
        "",
        "| Истина \\ ответ | " + " | ".join(CLS_RU[c] for c in TYPES) + " |",
        "| --- | " + " | ".join("---" for _ in TYPES) + " |",
        *[f"| {CLS_RU[c]} | " + " | ".join(str(v) for v in cm[i]) + " |" for i, c in enumerate(TYPES)],
        "",
        "## Сравнение с классификатором системы (те же срезы val)",
        "",
        f"- Система нашла опухоль на {pct(len(cnn), len(va))} срезов; тип у классификатора верен: {rec_line(cnn, 'answer')}; "
        f"сбалансированная точность {bacc3(cnn, 'answer'):.3f}.",
        f"- Модель по расположению на тех же срезах: {bacc3(cnn):.3f}.",
        f"- Ошибок типа у классификатора: {len(wrong)}. Модель по расположению на них права в "
        f"{pct(int((wrong.loc_pred == wrong.cls).sum()), len(wrong))}.",
        f"- Среднее вероятностей двух моделей (1:1, без подбора — только описание): сбалансированная точность "
        f"{bacc3(cnn, 'mix_pred'):.3f}, ошибок типа {int((cnn.mix_pred != cnn.cls).sum())} вместо {len(wrong)}.",
        "",
        "## Медианы признаков по типам",
        "",
        "| Часть | Тип | " + " | ".join(FEATURES) + " |",
        "| --- | --- | " + " | ".join("---" for _ in FEATURES) + " |",
        *[f"| {s} | {CLS_RU[c]} | " + " | ".join(f"{v:.3f}" for v in med.loc[(s, c)]) + " |" for s in ("train", "val") for c in TYPES],
        "",
        "Коэффициенты логистической регрессии (после стандартизации; знак — в пользу типа):",
        "",
        "| Тип | " + " | ".join(FEATURES) + " |",
        "| --- | " + " | ".join("---" for _ in FEATURES) + " |",
        *[f"| {CLS_RU[c]} | " + " | ".join(f"{v:.2f}" for v in coef.loc[c]) + " |" for c in TYPES],
        "",
        "Рисунки: `reports/location/d_edge_val.png` (расстояние до края по типам), `reports/location/examples_val.png` (примеры).",
    ]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
