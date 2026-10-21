"""Сравнение источников данных для сегментации. Test не используется.

Варианты:
  А — снимки кейса + маски, перенесенные из Figshare (только exact-совпадения);
  Б — Figshare с собственным делением по пациентам (как в исходном плане);
  В — Figshare в исходном виде, деление наследуется от сплита кейса по пациенту (pid_base).

Считает объемы по частям, утечку варианта Б относительно сплита кейса и — на снимках
train и val — насколько совпадают входы и маски вариантов А и В после препроцессинга.

Результат: data/processed/seg_figshare_split.csv (срезы Figshare с унаследованным split),
reports/masks_report.md, reports/masks/sources_compare.png.

Пример:
    python -m src.analyze_mask_sources
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from src.imhash import TF_NAMES, TFS
from src.io_utils import read_image
from src.labels import CLS_RU, TUMOR_CLASSES
from src.match_figshare import transfer_mask
from src.preprocess import mask_to_model, preprocess

SEED = 42
PARTS = ["train", "val", "test"]
C_ORANGE, C_BLUE = "#eb6834", "#2a78d6"


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float32).ravel(), b.astype(np.float32).ravel()
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


def inherit_split(meta: pd.DataFrame, splits: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Срезам Figshare присваивается split пациента из сплита кейса."""
    live = splits[splits["split"].isin(PARTS) & splits["pid_base"].notna() & (splits["pid_base"] != "")]
    per_pid = live.groupby("pid_base")["split"].agg(lambda s: sorted(set(s)))
    if (per_pid.map(len) > 1).any():
        raise ValueError(f"пациенты в нескольких частях: {list(per_pid.index[per_pid.map(len) > 1])}")
    pid_split = per_pid.map(lambda v: v[0])
    out = meta.copy()
    out["split"] = out["pid_base"].map(pid_split)
    out["patient_in_case"] = out["split"].notna()
    out["split"] = out["split"].fillna("train")  # пациентов нет в кейсе — утечки в test кейса быть не может
    ex = matches[matches["level"] == "exact"]
    out["in_case"] = out["id"].isin(set(ex["fig_id"].astype(int)))
    return out


def _count_table(d: pd.DataFrame, unit: str | None = None) -> pd.DataFrame:
    t = (pd.crosstab(d["cls"], d["split"]) if unit is None else d.groupby(["cls", "split"])[unit].nunique().unstack(fill_value=0))
    t = t.reindex(index=TUMOR_CLASSES, columns=PARTS, fill_value=0)
    t.loc["всего"] = t.sum()
    t["всего"] = t.sum(1)
    t.index = [CLS_RU.get(c, c) for c in t.index]
    return t


def compare_a_vs_c(pairs: pd.DataFrame, data_root: Path, fig_root: Path) -> pd.DataFrame:
    rows = []
    for r in tqdm(pairs.itertuples(index=False), total=len(pairs), desc="А против В"):
        case = read_image(data_root / r.path)
        fimg = read_image(fig_root / "images" / f"{r.fig_id}.png")
        fmask = read_image(fig_root / "masks" / f"{r.fig_id}.png")
        tf = TF_NAMES.index(r.tf)
        pc, mc = preprocess(case)
        pf, mf = preprocess(fimg)
        mask_c = TFS[tf](mask_to_model(fmask, mf)).astype(bool)  # вариант В: маска Figshare в координатах модели
        moved, _ = transfer_mask(fimg, fmask, tf, case)  # вариант А: перенос на снимок кейса
        mask_a = mask_to_model(moved, mc).astype(bool)
        rows.append({"path": r.path, "fig_id": r.fig_id, "cls": r.cls, "tf": r.tf,
                     "input_corr": _corr(pc, TFS[tf](pf)), "mask_dice": _dice(mask_a, mask_c),
                     "px_a": int(mask_a.sum()), "px_c": int(mask_c.sum()),
                     "case_h": case.shape[0], "case_w": case.shape[1]})
    return pd.DataFrame(rows)


def plot_compare(cmp: pd.DataFrame, data_root: Path, fig_root: Path, out: Path, n: int = 8) -> None:
    pick = pd.concat([cmp.nsmallest(n // 2, "mask_dice"), cmp.sample(n - n // 2, random_state=SEED)])
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5.9), dpi=100, facecolor="white")
    for k, r in enumerate(pick.itertuples(index=False)):
        case, fimg = read_image(data_root / r.path), read_image(fig_root / "images" / f"{r.fig_id}.png")
        fmask = read_image(fig_root / "masks" / f"{r.fig_id}.png")
        tf = TF_NAMES.index(r.tf)
        pc, mc = preprocess(case)
        pf, mf = preprocess(fimg)
        mask_c = TFS[tf](mask_to_model(fmask, mf))
        mask_a = mask_to_model(transfer_mask(fimg, fmask, tf, case)[0], mc)
        for ax, img, mask, color, ttl in (
                (axes[0, k], pc, mask_a, C_ORANGE, f"А: {Path(r.path).name}\n{r.case_w}×{r.case_h}, Dice с В {r.mask_dice:.3f}"),
                (axes[1, k], TFS[tf](pf), mask_c, C_BLUE, f"В: {r.fig_id}.mat\nсовпадение входов r={r.input_corr:.3f}")):
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            if mask.any():
                ax.contour(mask > 0, levels=[0.5], colors=[color], linewidths=1.2)
            ax.set_axis_off()
            ax.set_title(ttl, fontsize=7.5, pad=3)
    fig.suptitle(f"Вход и маска после препроцессинга: вверху вариант А (снимок кейса + перенесенная маска), внизу вариант В (Figshare). "
                 f"Слева {n // 2} худших по совпадению масок, справа случайные", fontsize=10.5, fontweight="bold", x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=0.2, h_pad=1.6)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def _md_table(df: pd.DataFrame) -> str:
    df = df.reset_index()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("mrt"))
    ap.add_argument("--fig-root", type=Path, default=Path("data/external/figshare/png"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports"))
    ap.add_argument("--n-compare", type=int, default=800, help="сколько пар train+val сравнивать (А против В)")
    args = ap.parse_args()

    meta = pd.read_csv(args.fig_root / "metadata.csv", dtype={"pid": str, "pid_base": str})
    splits = pd.read_csv(args.processed / "splits.csv", dtype={"pid_base": str})
    matches = pd.read_csv(args.processed / "figshare_matches.csv", dtype={"fig_id": str, "pid_base": str})

    # вариант В: Figshare с унаследованным делением
    fig = inherit_split(meta, splits, matches)
    cols = ["id", "image", "mask", "cls", "pid", "pid_base", "split", "patient_in_case", "in_case", "h", "w", "mask_px", "mask_frac"]
    fig[cols].to_csv(args.processed / "seg_figshare_split.csv", index=False, encoding="utf-8")

    # вариант А: рабочие снимки кейса с маской (exact)
    prim = splits[splits["primary"] & splits["split"].isin(PARTS) & (splits["cls"] != "no_tumor")].merge(
        matches[["path", "level", "fig_id", "tf"]], on="path")
    has_mask = set(splits.merge(matches[matches["level"] == "exact"][["path"]], on="path")["group_exact_id"])
    prim["has_mask"] = prim["group_exact_id"].isin(has_mask)
    a = prim[prim["has_mask"]]

    # вариант Б: собственное деление Figshare по пациентам — сколько срезов окажется «не в той» части
    sg = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    own = np.full(len(meta), "train", dtype=object)
    folds = list(sg.split(meta, meta["cls"], meta["pid_base"]))
    own[folds[0][1]], own[folds[1][1]] = "test", "val"
    b = fig.assign(own=own)
    known = b[b["patient_in_case"]]
    leak_tab = pd.crosstab(known["own"], known["split"]).reindex(index=PARTS, columns=PARTS, fill_value=0)
    leak_tab.index = [f"часть в варианте Б: {i}" for i in leak_tab.index]
    leak_tab.columns = [f"в сплите кейса: {c}" for c in leak_tab.columns]
    train_b_in_case_test = int(((known["own"] == "train") & (known["split"] == "test")).sum())
    test_b_in_case_train = int(((known["own"] == "test") & (known["split"] == "train")).sum())

    # А против В на train и val
    direct = prim[(prim["level"] == "exact") & prim["split"].isin(["train", "val"])].sort_values("path")
    pairs = direct.sample(min(args.n_compare, len(direct)), random_state=SEED)
    cmp = compare_a_vs_c(pairs, args.data_root, args.fig_root)
    plot_compare(cmp, args.data_root, args.fig_root, args.report_dir / "masks" / "sources_compare.png")
    q = lambda s, p: float(s.quantile(p))
    cmp_tab = cmp.groupby("cls").agg(
        пар=("path", "size"), совпадение_входов_медиана=("input_corr", "median"),
        совпадение_входов_5й_перцентиль=("input_corr", lambda s: q(s, 0.05)),
        dice_масок_медиана=("mask_dice", "median"), dice_масок_5й_перцентиль=("mask_dice", lambda s: q(s, 0.05)),
        доля_dice_от_0_9=("mask_dice", lambda s: (s >= 0.9).mean())).reindex(TUMOR_CLASSES).round(3)
    cmp_tab.index = [CLS_RU[c] for c in cmp_tab.index]
    not512 = cmp[(cmp["case_h"] != 512) | (cmp["case_w"] != 512)]

    neg = splits[splits["primary"] & (splits["cls"] == "no_tumor")]["split"].value_counts().reindex(PARTS)
    extra = fig[~fig["in_case"]]
    out = ["# Источник масок для сегментации", "",
           "Три варианта. Test в расчетах не участвует (сравнение А и В — на снимках train и val).", "",
           "- **А** — снимки кейса + маски, перенесенные из Figshare (только exact-совпадения);",
           "- **Б** — Figshare с собственным делением по пациентам (как в исходном плане);",
           "- **В** — Figshare в исходном виде (PNG без сжатия + исходная маска), а часть train/val/test наследуется "
           "от сплита кейса по пациенту (`pid_base`).", "",
           "## Объем по частям", "",
           "Вариант А — рабочие снимки кейса с маской:", "", _md_table(_count_table(a)), "",
           f"Рабочих снимков опухолей без маски (их нет в Figshare): {int((~prim['has_mask']).sum())} из {len(prim)} — "
           "в обучении сегментатора не участвуют ни в одном варианте.", "",
           "Вариант В — срезы Figshare:", "", _md_table(_count_table(fig)), "",
           "Вариант В — пациенты:", "", _md_table(_count_table(fig, "pid_base")), "",
           f"- срезов Figshare, которых нет в кейсе: {len(extra)} (train {int((extra['split'] == 'train').sum())}, "
           f"val {int((extra['split'] == 'val').sum())}, test {int((extra['split'] == 'test').sum())}) — вариант В добавляет их к обучению и оценке",
           f"- пациентов Figshare, которых нет в кейсе: {fig.loc[~fig['patient_in_case'], 'pid_base'].nunique()} "
           f"({int((~fig['patient_in_case']).sum())} срезов) — отнесены в train", "",
           "## Вариант Б дает утечку относительно сплита кейса", "",
           "Собственное деление Figshare (тот же метод, seed 42) против сплита кейса, срезы:", "", _md_table(leak_tab), "",
           f"- {train_b_in_case_test} срезов попали бы в обучение сегментатора, хотя их пациенты — в test кейса;",
           f"- {test_b_in_case_train} срезов попали бы в тест сегментатора, хотя их пациенты — в train кейса.",
           "Итоговая система оценивается на test кейса, поэтому вариант Б непригоден.", "",
           f"## А против В: одно и то же? ({len(cmp)} случайных пар из train и val)", "",
           "Для снимка кейса и его двойника из Figshare после препроцессинга сравниваются входы (корреляция) и маски (Dice).", "",
           _md_table(cmp_tab), "",
           f"- Dice масок ниже 0,9: {int((cmp['mask_dice'] < 0.9).sum())} из {len(cmp)}; ниже 0,8: {int((cmp['mask_dice'] < 0.8).sum())}",
           f"- снимков кейса не 512×512 в выборке: {len(not512)}"
           + (f"; у них Dice масок — медиана {not512['mask_dice'].median():.3f}, совпадение входов {not512['input_corr'].median():.3f}" if len(not512) else ""),
           "", "![](masks/sources_compare.png)", "",
           "## Снимки без опухоли для сегментатора", "",
           f"В Figshare опухоль есть на каждом срезе. Снимки без опухоли есть только в кейсе: train {int(neg['train'])}, "
           f"val {int(neg['val'])}, test {int(neg['test'])} (до чистки по разметке врача). Они могут идти в сегментацию с пустой маской, "
           "но взяты из других источников — сегментатор может выучить «чужой источник → пустая маска» (тот же shortcut).", "",
           "## Предложение", "",
           "**Вариант В.** Обучать сегментатор на Figshare в исходном виде, часть train/val/test наследовать от сплита кейса по пациенту.", "",
           f"- данных больше: train {int((fig['split'] == 'train').sum())} срезов против {int((a['split'] == 'train').sum())} в варианте А;",
           "- маски точные: это исходная разметка для своего снимка, без переноса "
           f"(в варианте А у {int((cmp['mask_dice'] < 0.9).sum())} из {len(cmp)} проверенных пар маска после переноса расходится с исходной, Dice ниже 0,9);",
           f"- входы те же: после препроцессинга снимок кейса и его двойник из Figshare совпадают (корреляция по медиане "
           f"{cmp['input_corr'].median():.3f}); сжатие JPEG, которого нет в PNG, добавляется аугментацией;",
           "- утечки нет: пациент целиком в одной части, той же, что в сплите кейса.", "",
           "Оценка сегментации: основная — на срезах Figshare пациентов из test; дополнительная — на снимках кейса из test "
           "(те же срезы после JPEG), чтобы показать работу на файлах в том виде, в каком они приходят.", "",
           f"Рабочие снимки опухолей без маски ({int((~prim['has_mask']).sum())} шт., не из Figshare) — проверка обобщения: "
           "сегментатор, обученный на Figshare, должен находить на них область опухоли (доля снимков с маской больше порога).", "",
           "Снимки без опухоли: обучить сегментатор в двух вариантах — без них и с ними (пустая маска) — и сравнить по ложным "
           "маскам на снимках «нет опухоли» из val и по стресс-тесту.", ""]
    (args.report_dir / "masks_report.md").write_text("\n".join(out), encoding="utf-8")
    cmp.to_csv(args.processed / "seg_a_vs_c.csv", index=False, encoding="utf-8")
    print("\n".join(l for l in out if not l.startswith("![")))


if __name__ == "__main__":
    main()
