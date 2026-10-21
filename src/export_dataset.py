"""Сборка датасета для публикации (два архива для GitHub Release).

1. brain_mri_dataset.zip — исходные снимки и обводки + таблица dataset.csv и описание README.md:
     brain_mri_dataset/images/<id>.<ext>, masks/<id>.png (только у срезов Figshare), dataset.csv, README.md.
   Кэш для обучения из него собирает python -m src.import_dataset <папка>.
2. brain_mri_cache256.zip — ровно то, на чем обучались модели: data/processed/pool.csv и кэш подготовки
   data/processed/cache/pool_256.* (снимки и маски 256×256 после src/preprocess.py). Распаковать в корень проекта.

Части train / val / test — как в pool.csv (деление по пациентам). Снимки организаторов и срезы Figshare
(Cheng et al., CC BY 4.0) копируются побайтно.

Пример:
    python -m src.export_dataset --out ../release
"""
from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

import pandas as pd

from src.labels import CLASSES, CLS_RU

CACHE_FILES = ["pool_256.npy", "pool_256_mask.npy", "pool_256.csv", "pool_256.json"]
COLUMNS = ["id", "image", "mask", "class", "split", "source", "patient_group", "figshare_id", "figshare_pid",
           "nt_mode", "nt_finding", "original_path"]
SOURCE_RU = {"figshare": "срез Figshare с ручной обводкой", "figshare_crop": "снимок организаторов: тот же срез Figshare в другом кадрировании",
             "other": "снимок организаторов, которого нет в Figshare"}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_table(pool: pd.DataFrame) -> pd.DataFrame:
    rows, n_case = [], 0
    for r in pool.sort_values(["root", "path"]).itertuples():
        if r.root == "figshare":
            rid = f"fig_{int(r.fig_id):04d}"
            original = f"figshare/{r.path}"
        else:
            n_case += 1
            rid = f"case_{n_case:04d}"
            original = f"organizers/{r.path}"
        ext = Path(r.path).suffix.lower()
        rows.append({"id": rid, "image": f"images/{rid}{ext}", "mask": f"masks/{rid}.png" if r.has_mask else "",
                     "class": r.cls, "split": r.split, "source": r.source, "patient_group": r.group,
                     "figshare_id": int(r.fig_id) if int(r.fig_id) >= 0 else "", "figshare_pid": r.pid_base,
                     "nt_mode": r.nt_mode, "nt_finding": r.nt_finding, "original_path": original,
                     "src_root": r.root, "src_path": r.path, "src_mask": r.mask})
    return pd.DataFrame(rows)


def dataset_readme(t: pd.DataFrame) -> str:
    by = pd.crosstab(t["class"].map(CLS_RU), t["split"]).reindex(index=[CLS_RU[c] for c in CLASSES], columns=["train", "val", "test"]).fillna(0).astype(int)
    src = t["source"].value_counts()
    lines = [
        "# Датасет: опухоли головного мозга на МРТ (T1 с контрастом), 2D-срезы",
        "",
        "Набор, на котором обучали и проверяли модели проекта. Каждый уникальный снимок — один раз; копии и искаженные версии "
        "убраны, данные поделены по пациентам: один пациент — только в одной части.",
        "",
        f"Снимков: {len(t)}; из них с ручной обводкой опухоли — {int((t['mask'] != '').sum())} (срезы Figshare).",
        "",
        "| Класс | train | val | test |",
        "| --- | --- | --- | --- |",
        *[f"| {c} | {r['train']} | {r['val']} | {r['test']} |" for c, r in by.iterrows()],
        "",
        "Источники: " + "; ".join(f"{SOURCE_RU[k]} — {int(v)}" for k, v in src.items()) + ".",
        "",
        "## Файлы",
        "",
        "- `images/<id>.<ext>` — снимок как в источнике (PNG — Figshare, JPG — данные организаторов кейса).",
        "- `masks/<id>.png` — обводка опухоли (0 — фон, 255 — опухоль), только у срезов Figshare.",
        "- `dataset.csv` — по строке на снимок:",
        "  - `class` — glioma / meningioma / pituitary / no_tumor;",
        "  - `split` — train / val / test (val — выбор моделей и порога, test — один раз для итоговой оценки);",
        "  - `source` — figshare / figshare_crop / other;",
        "  - `patient_group` — пациент или группа снимков одного пациента (по ней делили данные);",
        "  - `figshare_id`, `figshare_pid` — номер среза и пациента в Figshare;",
        "  - `nt_mode`, `nt_finding` — для класса «нет опухоли»: режим МРТ и находка по ручной проверке;",
        "  - `original_path` — откуда взят снимок.",
        "",
        "## Как получен",
        "",
        "1. Из 12 064 файлов организаторов убраны искаженные копии (яркость, затемнение, отражения, поворот, шум) и повторы "
        "одного снимка под разными именами — по «отпечатку изображения» с учетом поворотов и отражений.",
        "2. Снимки организаторов сопоставлены со срезами открытого набора Figshare: совпавшие заменены исходными срезами "
        "с ручной обводкой, совпадения дали номера пациентов.",
        "3. Класс «нет опухоли» проверен вручную: исключены снимки КТ, снимки с подозрением на опухоль и снимки плохого качества "
        "(правило — `src/make_review_decisions.py`, ответы — `reports/doctor/no_tumor_review/answers.csv`).",
        "4. Деление по пациентам: `StratifiedGroupKFold` (7 частей, seed 42), test — часть 1, val — часть 3.",
        "",
        "Код всех шагов — в `src/` репозитория; порядок команд — в README репозитория.",
        "",
        "## Источники и условия",
        "",
        "- Cheng J. et al. Brain tumor dataset. Figshare, version 8. doi:10.6084/m9.figshare.1512427. Лицензия CC BY 4.0 — "
        "при использовании указывайте авторов.",
        "- Снимки организаторов — датасет кейс-чемпионата «Инновации в медицине» (обезличенные снимки).",
        "- Класс «нет опухоли» собран из других источников и режимов МРТ, чем опухоли; здоровых снимков T1 с контрастом в наборе "
        "почти нет. Это ограничение набора, его нужно учитывать при обучении.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="папка для архивов (вне репозитория)")
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--data-root", type=Path, default=Path("mrt"))
    ap.add_argument("--fig-root", type=Path, default=Path("data/external/figshare/png"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(args.processed / "pool.csv", keep_default_na=False)
    pool["has_mask"] = pool["has_mask"].astype(str).str.lower() == "true"
    t = dataset_table(pool)
    roots = {"figshare": args.fig_root, "case": args.data_root}

    # 1. исходные снимки и обводки
    zp = args.out / "brain_mri_dataset.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:  # PNG и JPG уже сжаты
        for r in t.itertuples():
            z.write(roots[r.src_root] / r.src_path, f"brain_mri_dataset/{r.image}")
            if r.mask:
                z.write(roots[r.src_root] / r.src_mask, f"brain_mri_dataset/{r.mask}")
        z.writestr("brain_mri_dataset/dataset.csv", t[COLUMNS].to_csv(index=False).encode("utf-8"))
        z.writestr("brain_mri_dataset/README.md", dataset_readme(t).encode("utf-8"))
    print(f"{zp.name}: {len(t)} снимков, {int((t['mask'] != '').sum())} обводок, {zp.stat().st_size / 2**20:.0f} МБ")

    # 2. кэш, на котором шло обучение
    zc = args.out / "brain_mri_cache256.zip"
    with zipfile.ZipFile(zc, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(args.processed / "pool.csv", "data/processed/pool.csv")
        for n in CACHE_FILES:
            z.write(args.processed / "cache" / n, f"data/processed/cache/{n}")
    print(f"{zc.name}: pool.csv + кэш 256x256, {zc.stat().st_size / 2**20:.0f} МБ")
    (args.out / "checksums.md5").write_text("".join(f"{md5(p)}  {p.name}\n" for p in (zp, zc)), encoding="utf-8")


if __name__ == "__main__":
    main()
