"""Проверка shortcut: можно ли угадать класс по техническим признакам снимка.

Считается по рабочим снимкам (splits.csv, primary == True):
  raw_* — признаки файла «как есть»: размер, пропорции, цветность, яркость, контраст,
          черные поля, число уровней серого, резкость, фон в углах, байт на пиксель;
  pp_*  — признаки после реального препроцессинга (src/preprocess.py), по содержимому
          снимка без полей приведения к квадрату: то, что останется модели.
Проба: градиентный бустинг только на этих признаках (без пикселей), обучение на train,
оценка на val. Тест не используется.

Результаты: data/processed/image_stats.csv, reports/shortcut_report.md, reports/shortcut/*.png

Пример:
    python -m src.check_shortcut
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm import tqdm

from src.imhash import black_bbox
from src.preprocess import preprocess

SEED = 42
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
CLS_RU = {"glioma": "глиома", "meningioma": "менингиома", "pituitary": "гипофиз", "no_tumor": "нет опухоли"}
CLS_COLOR = {"glioma": "#2a78d6", "meningioma": "#eb6834", "pituitary": "#1baf7a", "no_tumor": "#eda100"}
INK, MUTED = "#0b0b0b", "#898781"

FEATURES_RU = {
    "raw_h": "высота, px", "raw_w": "ширина, px", "raw_aspect": "пропорции кадра (ш/в)",
    "raw_bpp": "байт на пиксель (сжатие JPEG)", "raw_color": "цветность (разница каналов)",
    "raw_mean": "средняя яркость", "raw_std": "контраст (СКО яркости)",
    "raw_border": "доля черных полей", "raw_bbox_aspect": "пропорции содержимого (ш/в)",
    "raw_levels": "число уровней серого", "raw_sat": "доля пересвеченных пикселей",
    "raw_sharp": "резкость (лапласиан)", "raw_corner": "яркость фона в углах",
    "pp_mean": "средняя яркость", "pp_std": "контраст (СКО яркости)", "pp_dark": "доля темных пикселей",
    "pp_sat": "доля пересвеченных пикселей", "pp_levels": "число уровней серого",
    "pp_entropy": "энтропия гистограммы", "pp_sharp": "резкость (лапласиан)",
    "pp_bbox_aspect": "пропорции содержимого (ш/в)", "pp_upscale": "масштаб при приведении к 256",
}
RAW = [f for f in FEATURES_RU if f.startswith("raw_")]
PP = [f for f in FEATURES_RU if f.startswith("pp_")]
SIZE_ONLY = ["raw_h", "raw_w"]
PP_VERSION = "preprocess-1.5"  # при изменении препроцессинга кэш признаков пересчитывается
PP_NO_SCALE = [f for f in PP if f != "pp_upscale"]


def image_stats(path: Path) -> dict:
    buf = np.fromfile(str(path), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    b, g, r = (bgr[..., k].astype(np.int16) for k in range(3))
    y0, y1, x0, x1 = black_bbox(gray)
    c = max(4, min(h, w) // 20)
    corners = np.concatenate([gray[:c, :c].ravel(), gray[:c, -c:].ravel(), gray[-c:, :c].ravel(), gray[-c:, -c:].ravel()])
    out = {
        "raw_h": h, "raw_w": w, "raw_aspect": w / h, "raw_bpp": len(buf) / (h * w),
        "raw_color": float((np.abs(r - g) + np.abs(g - b)).mean()),
        "raw_mean": float(gray.mean()), "raw_std": float(gray.std()),
        "raw_border": 1 - (y1 - y0) * (x1 - x0) / (h * w),
        "raw_bbox_aspect": (x1 - x0) / (y1 - y0),
        "raw_levels": int(np.count_nonzero(np.bincount(gray.ravel(), minlength=256))),
        "raw_sat": float((gray >= 250).mean()),
        "raw_sharp": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "raw_corner": float(corners.mean()),
    }
    # реальный препроцессинг (src/preprocess.py); статистика — по содержимому, без полей приведения к квадрату
    full, m = preprocess(gray)
    q = np.ascontiguousarray(full[m.top : m.top + m.new_h, m.left : m.left + m.new_w])
    pp = q.astype(np.float32) / 255
    hist = np.bincount(q.ravel(), minlength=256) / q.size
    nz = hist[hist > 0]
    out.update({
        "pp_mean": float(pp.mean()), "pp_std": float(pp.std()), "pp_dark": float((pp < 0.05).mean()),
        "pp_sat": float((pp > 0.95).mean()), "pp_levels": int(len(nz)), "pp_entropy": float(-(nz * np.log2(nz)).sum()),
        "pp_sharp": float(cv2.Laplacian(q, cv2.CV_64F).var()),
        "pp_bbox_aspect": (m.x1 - m.x0) / (m.y1 - m.y0), "pp_upscale": m.scale,
        "pp_version": PP_VERSION,
    })
    return out


def probe(d: pd.DataFrame, feats: list[str], binary: bool) -> dict:
    """Бустинг на технических признаках: обучение на train, оценка на val."""
    tr, va = d[d["split"] == "train"], d[d["split"] == "val"]
    y_tr = (tr["cls"] != "no_tumor").astype(int) if binary else tr["cls"]
    y_va = (va["cls"] != "no_tumor").astype(int) if binary else va["cls"]
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight="balanced", random_state=SEED)
    clf.fit(tr[feats], y_tr)
    pred = clf.predict(va[feats])
    res = {"bal_acc": balanced_accuracy_score(y_va, pred), "clf": clf, "va": va, "y_va": y_va}
    if binary:
        res["auc"] = roc_auc_score(y_va, clf.predict_proba(va[feats])[:, 1])
        res["recall_tumor"] = float((pred[y_va == 1] == 1).mean())
        res["specificity"] = float((pred[y_va == 0] == 0).mean())
    return res


# ---------------------------------------------------------------- графики


def plot_distributions(d: pd.DataFrame, feats: list[str], title: str, path: Path, log: tuple = ()) -> None:
    ncols = 3
    nrows = int(np.ceil(len(feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 2.15 * nrows + 0.6), dpi=110)
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.set_visible(False)
    for k, f in enumerate(feats):
        ax = axes[k // ncols, k % ncols]
        ax.set_visible(True)
        data = [d.loc[d["cls"] == c, f].to_numpy() for c in CLASSES]
        bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=False, whis=(5, 95),
                        medianprops={"color": INK, "lw": 1.4}, whiskerprops={"color": MUTED, "lw": 1},
                        capprops={"color": MUTED, "lw": 1}, boxprops={"lw": 0})
        for patch, c in zip(bp["boxes"], CLASSES):
            patch.set_facecolor(CLS_COLOR[c])
        ax.set_yticks(range(1, len(CLASSES) + 1), [CLS_RU[c] for c in CLASSES], fontsize=8)
        ax.invert_yaxis()
        if f in log:
            ax.set_xscale("log")
        ax.set_title(FEATURES_RU[f], fontsize=9, loc="left", color=INK)
        ax.tick_params(axis="x", labelsize=7.5, colors=MUTED)
        ax.grid(axis="x", alpha=0.25, lw=0.5)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle(f"{title}\nящик — 25–75-й перцентили, черта — медиана, усы — 5–95-й перцентили", fontsize=10.5,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path)
    plt.close(fig)


def plot_importance(imp: pd.Series, title: str, path: Path) -> None:
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(imp) + 1.1), dpi=110)
    ax.barh([FEATURES_RU[f] for f in imp.index], imp.to_numpy(), color="#2a78d6", height=0.62)
    for y, v in enumerate(imp.to_numpy()):
        ax.text(max(v, 0) + imp.max() * 0.015, y, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_xlabel("падение сбалансированной точности на val при перемешивании признака", fontsize=8.5, color=MUTED)
    ax.set_xlim(min(0, imp.min()), imp.max() * 1.15)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- отчет


def _md_table(df: pd.DataFrame) -> str:
    df = df.reset_index()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _medians(d: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    t = d.groupby("cls")[feats].median().reindex(CLASSES).T
    t.index = [FEATURES_RU[f] for f in t.index]
    t.columns = [CLS_RU[c] for c in t.columns]
    return t.map(lambda v: f"{v:.3g}")


def single_feature_auc(d: pd.DataFrame, feats: list[str]) -> pd.Series:
    """AUC «опухоль / нет опухоли» по одному признаку (на train+val), 0,5 — признак бесполезен."""
    x = d[d["split"].isin(["train", "val"])]
    y = (x["cls"] != "no_tumor").astype(int)
    auc = {f: roc_auc_score(y, x[f]) for f in feats}
    return pd.Series({f: max(a, 1 - a) for f, a in auc.items()}).sort_values(ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("mrt"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    s = pd.read_csv(args.processed / "splits.csv")
    d = s[s["primary"]].reset_index(drop=True)
    stats_path = args.processed / "image_stats.csv"
    st = pd.read_csv(stats_path) if stats_path.exists() else None
    if st is not None and st["path"].tolist() == d["path"].tolist() and "pp_version" in st and (st["pp_version"] == PP_VERSION).all():
        print(f"кэш: {stats_path}")
    else:
        st = pd.DataFrame([image_stats(args.data_root / p) for p in tqdm(d["path"], desc="признаки")])
        st.insert(0, "path", d["path"])
        st.to_csv(stats_path, index=False, encoding="utf-8")
    d = d.merge(st, on="path")

    rd = args.report_dir / "shortcut"
    rd.mkdir(parents=True, exist_ok=True)
    plot_distributions(d, RAW[2:], "Технические признаки снимков «как есть», по классам", rd / "features_raw.png",
                       log=("raw_sharp", "raw_bpp"))
    plot_distributions(d, PP, "Признаки после препроцессинга (обрезка полей → 256×256 → нормализация яркости)",
                       rd / "features_preprocessed.png", log=("pp_sharp",))

    probes = {}
    for name, feats in (("только размер файла (высота, ширина)", SIZE_ONLY), ("все признаки «как есть»", RAW),
                        ("после препроцессинга", PP), ("после препроцессинга, без масштаба", PP_NO_SCALE)):
        b, m = probe(d, feats, binary=True), probe(d, feats, binary=False)
        probes[name] = (feats, b, m)
    rows = [{"набор признаков": n, "AUC опухоль/нет": round(b["auc"], 3), "recall опухоли": round(b["recall_tumor"], 3),
             "specificity": round(b["specificity"], 3), "сбаланс. точность, 4 класса": round(m["bal_acc"], 3)}
            for n, (_, b, m) in probes.items()]
    probe_table = pd.DataFrame(rows).set_index("набор признаков")

    # контроль: «нет опухоли» против опухолей НЕ из Figshare (тот же круг источников)
    fm = pd.read_csv(args.processed / "figshare_matches.csv", usecols=["path", "level"])
    d = d.merge(fm, on="path")
    d["origin"] = np.where(d["level"] == "none", "не Figshare", "Figshare")
    ctl = d[(d["cls"] == "no_tumor") | (d["origin"] == "не Figshare")]
    ctl_rows = []
    for name, feats in (("все признаки «как есть»", RAW), ("после препроцессинга, без масштаба", PP_NO_SCALE)):
        b = probe(ctl, feats, binary=True)
        ctl_rows.append({"набор признаков": name, "AUC опухоль/нет": round(b["auc"], 3),
                         "recall опухоли": round(b["recall_tumor"], 3), "specificity": round(b["specificity"], 3)})
    ctl_table = pd.DataFrame(ctl_rows).set_index("набор признаков")
    ctl_n = ctl[ctl["split"] == "val"]["cls"].ne("no_tumor").value_counts()
    grp = np.where(d["cls"] == "no_tumor", "нет опухоли", np.where(d["origin"] == "Figshare", "опухоль, Figshare", "опухоль, не Figshare"))
    grp_cols = ["raw_h", "raw_w", "raw_bpp", "raw_sharp", "pp_levels", "pp_sharp", "pp_upscale"]
    grp_tab = d.groupby(grp)[grp_cols].median().round(3).rename(
        columns={c: ("как есть: " if c.startswith("raw_") else "после препроцессинга: ") + FEATURES_RU[c] for c in grp_cols})
    grp_tab.insert(0, "снимков", pd.Series(grp).value_counts())
    grp_tab.insert(1, "512×512, %", (100 * ((d["raw_h"] == 512) & (d["raw_w"] == 512)).groupby(grp).mean()).round(1))

    imps = {}
    for key, name, fname in (("все признаки «как есть»", "«как есть»", "importance_raw.png"),
                             ("после препроцессинга", "после препроцессинга", "importance_preprocessed.png")):
        feats, b, _ = probes[key]
        pi = permutation_importance(b["clf"], b["va"][feats], b["y_va"], scoring="balanced_accuracy", n_repeats=20,
                                    random_state=SEED)
        imps[key] = pd.Series(pi.importances_mean, index=feats)
        plot_importance(imps[key], f"Какие признаки выдают класс «нет опухоли» ({name})", rd / fname)

    # размеры
    size = d.assign(size=d["raw_w"].astype(str) + "×" + d["raw_h"].astype(str))
    top = size["size"].value_counts().head(6).index
    size_tab = (100 * pd.crosstab(size["size"].where(size["size"].isin(top), "другие"), size["cls"], normalize="columns")).round(1)
    size_tab = size_tab.reindex(index=list(top) + ["другие"], columns=CLASSES).rename(columns=CLS_RU)
    n_sizes = size.groupby("cls")["size"].nunique().reindex(CLASSES).rename(CLS_RU)
    color_share = (100 * (d["raw_color"] > 1).groupby(d["cls"]).mean()).round(1).reindex(CLASSES).rename(CLS_RU)
    # Figshare vs прочие источники внутри классов опухолей
    origin_tab = pd.crosstab(d["cls"], d["origin"]).reindex(CLASSES).rename(index=CLS_RU)

    auc_raw, auc_pp = single_feature_auc(d, RAW), single_feature_auc(d, PP)
    out = ["# Проверка shortcut", "",
           f"Рабочие снимки: {len(d)} (train {int((d['split'] == 'train').sum())}, val {int((d['split'] == 'val').sum())}, "
           f"test {int((d['split'] == 'test').sum())}). Проба обучается на train и оценивается на val; test не используется.", "",
           "## Проба: угадывается ли класс без пикселей", "",
           "Градиентный бустинг только на технических признаках. Если класс угадывается — нейросеть тоже может "
           "выучить эти признаки вместо опухоли. Случайное угадывание: AUC 0,5; сбалансированная точность 4 классов 0,25.", "",
           _md_table(probe_table), "",
           "«Масштаб» сеть напрямую не видит (он проявляется только через резкость и текстуру), поэтому есть строка без него.", "",
           "### Контроль: «нет опухоли» против опухолей не из Figshare", "",
           f"Опухоли не из Figshare взяты из того же круга источников, что и «нет опухоли» (val: {int(ctl_n.get(True, 0))} опухолей, "
           f"{int(ctl_n.get(False, 0))} без опухоли). Если класс угадывается и здесь — дело не только в Figshare.", "",
           _md_table(ctl_table), "",
           "Медианы по трем группам снимков:", "", _md_table(grp_tab.T), "",
           "![](shortcut/importance_raw.png)", "", "![](shortcut/importance_preprocessed.png)", "",
           "## AUC «опухоль / нет опухоли» по одному признаку (train+val)", "",
           "«Как есть»:", "", _md_table(auc_raw.rename(FEATURES_RU).round(3).rename("AUC").to_frame()), "",
           "После препроцессинга:", "", _md_table(auc_pp.rename(FEATURES_RU).round(3).rename("AUC").to_frame()), "",
           "## Размеры файлов, % снимков класса", "", _md_table(size_tab), "",
           "Разных размеров в классе: " + ", ".join(f"{k} — {v}" for k, v in n_sizes.items()) + ".", "",
           "Доля цветных файлов (каналы различаются), %: " + ", ".join(f"{k} — {v}" for k, v in color_share.items()) + ".", "",
           "## Происхождение рабочих снимков", "", _md_table(origin_tab), "",
           "## Медианы признаков «как есть»", "", _md_table(_medians(d, RAW)), "",
           "![](shortcut/features_raw.png)", "",
           "## Медианы признаков после препроцессинга", "", _md_table(_medians(d, PP)), "",
           "![](shortcut/features_preprocessed.png)", ""]
    (args.report_dir / "shortcut_report.md").write_text("\n".join(out), encoding="utf-8")
    print(probe_table.to_string())
    print(f"отчет: {args.report_dir / 'shortcut_report.md'}")


if __name__ == "__main__":
    main()
