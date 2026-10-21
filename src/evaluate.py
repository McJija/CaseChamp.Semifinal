"""Финальная оценка замороженной системы: классификатор + сегментатор + правило «опухоль есть».

Система загружается из models/final/ через src/inference.py — ровно та, что в прототипе. Каждый снимок
читается из исходного файла и проходит обычный препроцессинг. Считается:
  - «опухоль есть / нет»: recall и specificity с интервалом Уилсона, в целом и по подгруппам
    (источник опухоли; режим и находка у «нет опухоли»), отдельно классификатор без сегментатора;
  - 4 класса: таблица ошибок, F1 по классам, macro-F1, верный тип среди найденных опухолей; AUC p_tumor;
  - флаг «требует внимания врача»; уверенность классификатора против доли верных ответов;
  - сегментация: Dice и IoU в координатах исходного снимка, по срезам и по пациентам, 95% интервал —
    бутстреп по группам (2000, seed 42); доля опухолей без маски, где найдена область;
  - попадание Grad-CAM истинного класса в маску, расширенную на 16 пикселей (configs/eval_definitions.yaml);
  - дополнительно: JPEG-версии тех же срезов из кейса; копии организаторов (BR, DA, HF, VF, RO, SP)
    у групп этой части; для --split test — «цена утечки» на всем тесте организаторов.
Тест организаторов содержит снимки нашего test, поэтому при --split val он не прогоняется.

Результат: reports/eval_<split>/ (predictions.csv, jpeg.csv, copies.csv, organizers_test.csv, metrics.json)
и отчет reports/eval_<split>/metrics.md.

Пример:
    python -m src.evaluate --split val      # сверка с известными числами
    python -m src.evaluate --split test     # финальная оценка, один раз
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import roc_auc_score

from src.config import load_config
from src.dataset import to_tensor
from src.inference import load_predictor
from src.io_utils import read_image
from src.labels import CLASSES, CLS_RU
from src.metrics import wilson
from src.preprocess import mask_to_model, preprocess
from src.stress import DEFS

AUGS = ["BR", "DA", "HF", "VF", "RO", "SP"]
SEED, N_BOOT = 42, 2000


def dice_iou(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    inter, s, union = int((pred & true).sum()), int(pred.sum() + true.sum()), int((pred | true).sum())
    return (2 * inter / s if s else 1.0), (inter / union if union else 1.0)


def run(predictor, path: Path) -> dict:
    r = predictor.predict(read_image(path), heatmap=False)
    return {"answer": r["class"], "present": r["class"] != "no_tumor", "p_tumor": r["p_tumor"], "confidence": r["confidence"],
            "cls_argmax": max(r["probs"], key=r["probs"].get), "mask_area": r["mask_area_px"],
            "flag": bool(r["flags"]), "flag_reason": r["flag_reason"], **{f"p_{c}": r["probs"][c] for c in CLASSES}, "_mask": r["mask"]}


def cam_hit(predictor, image: np.ndarray, true_mask: np.ndarray, cls: str, dilate_px: int) -> bool:
    """Максимум Grad-CAM истинного класса внутри маски, расширенной на dilate_px (в координатах модели)."""
    x256, meta = preprocess(image)
    x = to_tensor(x256).unsqueeze(0).to(predictor.device)
    cam = predictor.cam(input_tensor=x, targets=[ClassifierOutputTarget(CLASSES.index(cls))])[0]
    m = mask_to_model(true_mask.astype(np.uint8) * 255, meta) > 0
    m = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))) > 0
    y, x_ = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return bool(m[y, x_])


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
    return {"k": int(k), "n": int(n), "rate": (k / n if n else float("nan")), "lo": lo, "hi": hi}


def boot_ci(df: pd.DataFrame, fn, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    if df.empty:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    groups = df["group"].to_numpy()
    uniq, inv = np.unique(groups, return_inverse=True)
    rows = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    vals = [fn(df.iloc[np.concatenate([rows[g] for g in rng.integers(0, len(uniq), len(uniq))])]) for _ in range(n_boot)]
    return float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


def presence_table(d: pd.DataFrame, col: str) -> list[dict]:
    tu, nt = d[d["cls"] != "no_tumor"], d[d["cls"] == "no_tumor"]
    rows = [("опухоли: все", tu, True)]
    rows += [(f"опухоли: {s}", tu[tu["source"] == s], True) for s in ("figshare", "figshare_crop", "other")]
    rows += [("нет опухоли: все", nt, False)]
    rows += [(f"нет опухоли: {m}", nt[nt["nt_mode"] == m], False) for m in sorted(nt["nt_mode"].unique())]
    rows += [(f"нет опухоли: {f}", nt[nt["nt_finding"] == f], False) for f in sorted(nt["nt_finding"].unique())]
    out = []
    for name, x, tumor in rows:
        k = int(x[col].sum()) if tumor else int((~x[col]).sum())
        out.append({"группа": name, "мера": "найдено опухолей" if tumor else "верно «нет опухоли»", **rate(k, len(x))})
    return out


def classification(d: pd.DataFrame, col: str) -> dict:
    cm = pd.crosstab(pd.Categorical(d["cls"], CLASSES), pd.Categorical(d[col], CLASSES), dropna=False)
    per = {}
    for c in CLASSES:
        tp, fp, fn = int(cm.loc[c, c]), int(cm[c].sum() - cm.loc[c, c]), int(cm.loc[c].sum() - cm.loc[c, c])
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per[c] = {"precision": p, "recall": r, "f1": (2 * p * r / (p + r) if p + r else 0.0), "n": int(cm.loc[c].sum())}
    found = d[(d["cls"] != "no_tumor") & (d[col] != "no_tumor")]
    return {"confusion": cm.values.tolist(), "per_class": per, "macro_f1": float(np.mean([v["f1"] for v in per.values()])),
            "accuracy": float((d["cls"] == d[col]).mean()), "type_correct_among_found": rate(int((found["cls"] == found[col]).sum()), len(found))}


def calibration(d: pd.DataFrame) -> list[dict]:
    conf = d[[f"p_{c}" for c in CLASSES]].max(1)
    ok = d["cls_argmax"] == d["cls"]
    bins = [0, .5, .7, .9, .97, .99, 1.0001]
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        out.append({"уверенность": f"{lo:.2f}–{min(hi, 1):.2f}", "n": int(m.sum()),
                    "средняя_уверенность": float(conf[m].mean()) if m.any() else float("nan"), "доля_верных": float(ok[m].mean()) if m.any() else float("nan")})
    return out


def segmentation(d: pd.DataFrame) -> dict:
    m = d[d["has_mask"]]
    out = {"n_slices": len(m), "n_patients": int(m["group"].nunique())}
    by_patient = lambda x: x.groupby("group")["dice"].mean().mean()  # noqa: E731
    out["dice_slice"], out["dice_slice_ci"] = float(m["dice"].mean()), boot_ci(m, lambda x: x["dice"].mean())
    out["dice_patient"], out["dice_patient_ci"] = float(by_patient(m)), boot_ci(m, by_patient)
    out["iou_slice"] = float(m["iou"].mean())
    out["per_class"] = {c: {"n": int((m["cls"] == c).sum()), "dice": float(m.loc[m["cls"] == c, "dice"].mean()),
                            "dice_ci": boot_ci(m[m["cls"] == c], lambda x: x["dice"].mean()),
                            "patients": int(m.loc[m["cls"] == c, "group"].nunique())} for c in CLASSES[:3]}
    out["area_found_masked"] = rate(int((m["mask_area"] > 0).sum()), len(m))
    un = d[(d["cls"] != "no_tumor") & ~d["has_mask"]]
    out["area_found_unmasked"] = {s: rate(int((un.loc[un["source"] == s, "mask_area"] > 0).sum()), int((un["source"] == s).sum()))
                                  for s in ("figshare_crop", "other")}
    nt = d[d["cls"] == "no_tumor"]
    out["false_mask_no_tumor"] = rate(int((nt["mask_area"] > 0).sum()), len(nt))
    return out


def fmt(r: dict) -> str:
    return f"{r['k']} из {r['n']} ({100 * r['rate']:.1f}%; 95% ДИ {100 * r['lo']:.1f}–{100 * r['hi']:.1f})" if r["n"] else "—"


def report(split: str, M: dict, path: Path) -> None:
    L = [f"# Оценка системы на {split}", "", f"Система: {M['system']['classifier']} + {M['system']['segmenter']}, порог p_tumor ≥ {M['system']['threshold']}, "
         f"площадь маски > {M['system']['min_area']} px. Снимков: {M['n']} (опухолей {M['n_tumor']}, «нет опухоли» {M['n_no_tumor']}). "
         f"Посчитано {M['date']}, {M['minutes']:.1f} мин на CPU.", "",
         "## «Опухоль есть / нет» — система (две модели)", "", "| Группа | Мера | Результат |", "| --- | --- | --- |"]
    L += [f"| {r['группа']} | {r['мера']} | {fmt(r)} |" for r in M["presence"]]
    L += ["", "Классификатор отдельно (argmax), для сравнения:", "", "| Группа | Мера | Результат |", "| --- | --- | --- |"]
    L += [f"| {r['группа']} | {r['мера']} | {fmt(r)} |" for r in M["presence_classifier_only"]]
    L += ["", f"AUC вероятности опухоли у классификатора: {M['auc_p_tumor']:.3f}.", "",
          "## Четыре класса — ответ системы", "", f"Доля верных ответов {M['cls']['accuracy']:.3f}, macro-F1 {M['cls']['macro_f1']:.3f}. "
          f"Тип верен среди найденных опухолей: {fmt(M['cls']['type_correct_among_found'])}.", "",
          "| Истина \\ ответ | " + " | ".join(CLS_RU[c] for c in CLASSES) + " |", "| --- |" + " --- |" * 4]
    L += [f"| {CLS_RU[c]} | " + " | ".join(str(v) for v in row) + " |" for c, row in zip(CLASSES, M["cls"]["confusion"])]
    L += ["", "| Класс | Precision | Recall | F1 | Снимков |", "| --- | --- | --- | --- | --- |"]
    L += [f"| {CLS_RU[c]} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['n']} |" for c, v in M["cls"]["per_class"].items()]
    f = M["flag"]
    L += ["", "## Флаг «требует внимания врача» (модели расходятся)", "", f"Опухоли: {fmt(f['tumor'])}; «нет опухоли»: {fmt(f['no_tumor'])}. "
          f"Среди снимков с флагом ответ системы неверен (наличие или тип) у {fmt(f['wrong_among_flagged'])}.", "",
          "## Уверенность классификатора и доля верных ответов", "", "| Уверенность | Снимков | Средняя уверенность | Доля верных |", "| --- | --- | --- | --- |"]
    L += [f"| {r['уверенность']} | {r['n']} | {r['средняя_уверенность']:.3f} | {r['доля_верных']:.3f} |" for r in M["calibration"] if r["n"]]
    s = M["seg"]
    L += ["", "## Сегментация (в координатах исходного снимка)", "",
          f"Срезов с маской {s['n_slices']} от {s['n_patients']} пациентов. Dice по срезам {s['dice_slice']:.3f} (95% ДИ {s['dice_slice_ci'][0]:.3f}–{s['dice_slice_ci'][1]:.3f}), "
          f"по пациентам {s['dice_patient']:.3f} ({s['dice_patient_ci'][0]:.3f}–{s['dice_patient_ci'][1]:.3f}); IoU {s['iou_slice']:.3f}.", "",
          "| Тип | Срезов | Пациентов | Dice | 95% ДИ |", "| --- | --- | --- | --- | --- |"]
    L += [f"| {CLS_RU[c]} | {v['n']} | {v['patients']} | {v['dice']:.3f} | {v['dice_ci'][0]:.3f}–{v['dice_ci'][1]:.3f} |" for c, v in s["per_class"].items()]
    L += ["", f"Область найдена: на срезах с маской {fmt(s['area_found_masked'])}; на опухолях без маски — figshare_crop {fmt(s['area_found_unmasked']['figshare_crop'])}, "
          f"other {fmt(s['area_found_unmasked']['other'])}. Ложная маска на снимках «нет опухоли»: {fmt(s['false_mask_no_tumor'])}.", "",
          f"Попадание максимума Grad-CAM истинного класса в маску (+16 px): {fmt(M['cam_hit'])}."]
    if "jpeg" in M:
        j = M["jpeg"]
        L += ["", "## JPEG-версии тех же срезов из кейса", "", f"Срезов {j['n']}: опухоль найдена {fmt(j['found'])}; тип верен {fmt(j['type'])}; "
              f"ответ совпал с ответом на PNG-версии {fmt(j['agree'])}."]
    if "copies" in M:
        L += ["", "## Копии организаторов (устойчивость)", "", "Оговорка: копии сжаты до 224×224 без сохранения пропорций.", "",
              "| Тип копии | Снимков | Наличие верно | Тип/класс верен | Ответ как на оригинале |", "| --- | --- | --- | --- | --- |"]
        L += [f"| {a} | {v['n']} | {v['present_ok']:.3f} | {v['class_ok']:.3f} | {v['same_as_original']:.3f} |" for a, v in M["copies"].items()]
    if "organizers_test" in M:
        L += ["", "## Цена утечки: тест организаторов как есть", "", "Метки — исходные папки организаторов; файлы — все, включая копии.", "",
              "| Чьи снимки | Файлов | Доля верных (4 класса) | Наличие опухоли верно |", "| --- | --- | --- | --- |"]
        L += [f"| {k} | {v['n']} | {v['acc']:.3f} | {v['present_ok']:.3f} |" for k, v in M["organizers_test"].items()]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--data-root", type=Path, default=Path("mrt"))
    ap.add_argument("--fig-root", type=Path, default=Path("data/external/figshare/png"))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="первые N снимков каждого класса (отладка)")
    ap.add_argument("--no-extra", action="store_true", help="без JPEG-версий, копий и теста организаторов")
    args = ap.parse_args()
    out = args.out_dir or Path("reports") / f"eval_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    t0 = time.perf_counter()

    predictor = load_predictor()
    final = Path("models/final/manifest.json")
    system = {"classifier": "models/final/classifier.pt", "segmenter": "models/final/segmenter.pt"} if final.exists() else {"classifier": "?", "segmenter": "?"}
    system.update(threshold=predictor.presence_threshold, min_area=predictor.min_mask_area_px)
    if predictor.presence_threshold is None:
        raise SystemExit("порог решения «опухоль есть» не загружен — сначала python -m src.presence и python -m src.freeze")
    dilate = int(load_config(DEFS)["gradcam_hit"]["dilate_px"]) if "gradcam_hit" in load_config(DEFS) else 16
    roots = {"figshare": args.fig_root, "case": args.data_root}

    pool = pd.read_csv(args.processed / "pool.csv", keep_default_na=False)
    pool["has_mask"] = pool["has_mask"].astype(str).str.lower() == "true"
    d = pool[pool["split"] == args.split].reset_index(drop=True)
    if args.limit:
        d = d.groupby("cls").head(args.limit).reset_index(drop=True)

    rows, hits = [], []
    for i, r in enumerate(d.itertuples()):
        res = run(predictor, roots[r.root] / r.path)
        mask = res.pop("_mask")
        dice = iou = np.nan
        if r.has_mask:
            true = read_image(args.fig_root / r.mask) > 0
            dice, iou = dice_iou(mask, true)
            hits.append(cam_hit(predictor, read_image(roots[r.root] / r.path), true, r.cls, dilate))
        rows.append({**res, "dice": dice, "iou": iou})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} из {len(d)}", flush=True)
    d = pd.concat([d, pd.DataFrame(rows)], axis=1)
    d["cls_present"] = d["cls_argmax"] != "no_tumor"
    d.to_csv(out / "predictions.csv", index=False, encoding="utf-8")

    tu, nt = d[d["cls"] != "no_tumor"], d[d["cls"] == "no_tumor"]
    flagged = d[d["flag"]]
    wrong = (flagged["answer"] != flagged["cls"])
    M = {"split": args.split, "system": system, "n": len(d), "n_tumor": len(tu), "n_no_tumor": len(nt), "date": time.strftime("%Y-%m-%d %H:%M"),
         "presence": presence_table(d, "present"), "presence_classifier_only": presence_table(d, "cls_present"),
         "auc_p_tumor": float(roc_auc_score(d["cls"] != "no_tumor", d["p_tumor"])),
         "cls": classification(d, "answer"), "calibration": calibration(d), "seg": segmentation(d),
         "flag": {"tumor": rate(int(tu["flag"].sum()), len(tu)), "no_tumor": rate(int(nt["flag"].sum()), len(nt)),
                  "wrong_among_flagged": rate(int(wrong.sum()), len(flagged))},
         "cam_hit": rate(int(sum(hits)), len(hits))}

    if not args.no_extra:
        # JPEG-версии срезов Figshare из кейса
        jp = d[(d["root"] == "figshare") & (d["case_path"] != "")]
        jr = pd.DataFrame([run(predictor, args.data_root / p) for p in jp["case_path"]], columns=["answer", "present", "_mask"]
                          if jp.empty else None).drop(columns="_mask")
        jr.insert(0, "path", jp["path"].to_numpy())
        jr.to_csv(out / "jpeg.csv", index=False, encoding="utf-8")
        M["jpeg"] = {"n": len(jr), "found": rate(int(jr["present"].sum()), len(jr)),
                     "type": rate(int((jr["answer"].to_numpy() == jp["cls"].to_numpy()).sum()), len(jr)),
                     "agree": rate(int((jr["answer"].to_numpy() == jp["answer"].to_numpy()).sum()), len(jr))}
        # копии организаторов у групп этой части и их оригиналы
        s = pd.read_csv(args.processed / "splits.csv", keep_default_na=False)
        cp = s[(s["split"] == args.split) & s["aug"].isin(AUGS)]
        orig = s[(s["aug"] == "orig") & s["base_id"].isin(cp["base_id"])].drop_duplicates("base_id").set_index("base_id")["path"]
        cache = {}
        def pred_case(p):  # noqa: E306
            if p not in cache:
                r = run(predictor, args.data_root / p)
                r.pop("_mask")
                cache[p] = r
            return cache[p]
        crow = []
        for r in cp.itertuples():
            a = pred_case(r.path)
            o = pred_case(orig[r.base_id]) if r.base_id in orig.index else None
            crow.append({"path": r.path, "aug": r.aug, "cls": r.cls, "answer": a["answer"], "present": a["present"],
                         "orig_answer": o["answer"] if o else "", "flag": a["flag"]})
        c = pd.DataFrame(crow)
        c.to_csv(out / "copies.csv", index=False, encoding="utf-8")
        M["copies"] = {a: {"n": int((c["aug"] == a).sum()),
                           "present_ok": float((c.loc[c["aug"] == a, "present"] == (c.loc[c["aug"] == a, "cls"] != "no_tumor")).mean()),
                           "class_ok": float((c.loc[c["aug"] == a, "answer"] == c.loc[c["aug"] == a, "cls"]).mean()),
                           "same_as_original": float((c.loc[(c["aug"] == a) & (c["orig_answer"] != ""), "answer"]
                                                      == c.loc[(c["aug"] == a) & (c["orig_answer"] != ""), "orig_answer"]).mean())} for a in AUGS}
        # цена утечки — только при финальной оценке (тест организаторов содержит наш test)
        if args.split == "test":
            ot = s[s["split_orig"] == "test"]
            orow = []
            for j, r in enumerate(ot.itertuples()):
                a = pred_case(r.path)
                orow.append({"path": r.path, "cls_orig": r.cls_orig, "our_split": r.split, "answer": a["answer"], "present": a["present"]})
                if (j + 1) % 500 == 0:
                    print(f"  тест организаторов: {j + 1} из {len(ot)}", flush=True)
            o = pd.DataFrame(orow)
            o.to_csv(out / "organizers_test.csv", index=False, encoding="utf-8")
            who = {"весь тест организаторов": o, "пациенты из нашего train и val": o[o["our_split"].isin(["train", "val"])],
                   "наши тестовые пациенты": o[o["our_split"] == "test"], "исключенные и неиспользуемые": o[o["our_split"].isin(["excluded", "unused"])]}
            M["organizers_test"] = {k: {"n": len(x), "acc": float((x["answer"] == x["cls_orig"]).mean()),
                                        "present_ok": float((x["present"] == (x["cls_orig"] != "no_tumor")).mean())} for k, x in who.items()}

    M["minutes"] = (time.perf_counter() - t0) / 60
    (out / "metrics.json").write_text(json.dumps(M, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    report(args.split, M, out / "metrics.md")
    print(f"готово за {M['minutes']:.1f} мин: {out / 'metrics.md'}")
    for r in M["presence"][:1] + [M["presence"][4]]:
        print(f"  {r['группа']}: {fmt(r)}")
    print(f"  тип верен среди найденных: {fmt(M['cls']['type_correct_among_found'])}; Dice по срезам {M['seg']['dice_slice']:.3f}")


if __name__ == "__main__":
    main()
