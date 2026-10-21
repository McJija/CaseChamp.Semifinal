"""Индекс файлов датасета МРТ.

Обходит DATA_ROOT/<split>/<класс>/*.jpg и по имени файла определяет схему
именования (source), группу снимка (base_id) и тип аугментации (aug).

base_id строится только по имени файла. Он надежен для схем aug_GMPN
(G_704 + G_704_BR_ ...) и kaggle_TrTe (Tr-gl_0001). Для остальных схем
имена неуникальны между классами, поэтому base_id включает класс; реальные
дубликаты между схемами и сплитами ищутся по хешам изображений.

Пример:
    python -m src.build_index --data-root mrt --out data/processed/index.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Папки на диске -> коды. Имена папок нормализуются через strip()
# (в тесте у части классов ведущий пробел).
SPLIT_PREFIXES = {"Обуч": "train", "Тест": "test"}
CLASS_CODES = {
    "Глиома": "glioma",
    "Менингиома": "meningioma",
    "Гипофиз": "pituitary",
    "Опухоли нет": "no_tumor",
}
GMPN_LETTER_TO_CLS = {"G": "glioma", "M": "meningioma", "P": "pituitary", "N": "no_tumor"}
KAGGLE_CODE_TO_CLS = {"gl": "glioma", "me": "meningioma", "pi": "pituitary", "no": "no_tumor"}
AUG_TYPES = ("BR", "DA", "HF", "VF", "RO", "SP")

# Порядок важен: первая подходящая схема побеждает.
RE_GMPN = re.compile(r"^(?P<L>[GMPN])_(?P<n>\d+)(?:_(?P<aug>BR|DA|HF|VF|RO|SP)_)?$")
RE_KAGGLE = re.compile(r"^(?P<sub>Tr|Te)-(?P<code>gl|me|pi|no)_(?P<n>\d+)$")
RE_NUMERIC = re.compile(r"^(?P<n>\d+)$")
RE_PAREN = re.compile(r"^(?P<p>gg|m\d*|p) ?\((?P<n>\d+)\)$")
RE_IMAGE = re.compile(r"^image(?:\((?P<n>\d+)\))?(?P<copy> - Copy)?$")
RE_NO = re.compile(r"^(?:(?P<a>\d+) ?no|no ?(?P<b>\d+)|no|N(?P<c>\d+))$", re.IGNORECASE)


def parse_name(stem: str, cls: str) -> dict:
    """Разбирает имя файла (без расширения) в source / base_id / aug / name_cls."""
    if m := RE_GMPN.match(stem):
        return {
            "source": "aug_GMPN",
            "base_id": f"{m['L']}_{int(m['n'])}",
            "aug": m["aug"] or "orig",
            "name_cls": GMPN_LETTER_TO_CLS[m["L"]],
        }
    if m := RE_KAGGLE.match(stem):
        return {
            "source": "kaggle_TrTe",
            "base_id": stem,
            "aug": "orig",
            "name_cls": KAGGLE_CODE_TO_CLS[m["code"]],
        }
    if m := RE_NUMERIC.match(stem):
        return {"source": "numeric", "base_id": f"num:{cls}:{int(m['n'])}", "aug": "orig", "name_cls": None}
    if m := RE_PAREN.match(stem):
        # "m1 (3)" и "m1(3)" сводим к одной группе: лишнее объединение безопаснее пропущенной утечки
        return {"source": "paren", "base_id": f"paren:{cls}:{m['p']}({int(m['n'])})", "aug": "orig", "name_cls": None}
    if m := RE_IMAGE.match(stem):
        n = m["n"] if m["n"] is not None else "x"
        return {
            "source": "image_paren",
            "base_id": f"image:{cls}:{n}",
            "aug": "COPY" if m["copy"] else "orig",
            "name_cls": None,
        }
    if m := RE_NO.match(stem):
        n = m["a"] or m["b"] or m["c"]
        # вариант написания сохраняем в ключе: "1 no" и "No1" в исходниках — разные файлы
        variant = re.sub(r"\d+", "#", stem).lower()
        return {
            "source": "no_named",
            "base_id": f"no:{cls}:{variant}:{n or 'x'}",
            "aug": "orig",
            "name_cls": "no_tumor",
        }
    return {"source": "unknown", "base_id": f"unk:{cls}:{stem}", "aug": "unknown", "name_cls": None}


def detect_split(folder: str) -> str:
    for prefix, split in SPLIT_PREFIXES.items():
        if folder.strip().startswith(prefix):
            return split
    raise ValueError(f"Не удалось определить split по папке: {folder!r}")


def build_index(data_root: Path) -> pd.DataFrame:
    rows = []
    for split_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        split = detect_split(split_dir.name)
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            cls_name = cls_dir.name.strip()
            if cls_name not in CLASS_CODES:
                raise ValueError(f"Неизвестный класс: {cls_dir.name!r}")
            cls = CLASS_CODES[cls_name]
            for f in sorted(cls_dir.iterdir()):
                if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                    continue
                info = parse_name(f.stem, cls)
                rows.append({
                    "path": f.relative_to(data_root).as_posix(),
                    "split": split,
                    "cls": cls,
                    "base_id": info["base_id"],
                    "aug": info["aug"],
                    "is_original": info["aug"] == "orig",
                    "source": info["source"],
                    "name_cls_mismatch": info["name_cls"] is not None and info["name_cls"] != cls,
                })
    df = pd.DataFrame(rows)
    return df.sort_values(["split", "cls", "path"], kind="stable").reset_index(drop=True)


def _md_table(df: pd.DataFrame) -> str:
    df = df.reset_index()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _crosstab(index, columns) -> pd.DataFrame:
    return pd.crosstab(index, columns, margins=True, margins_name="всего")


def make_report(df: pd.DataFrame) -> str:
    out = ["# Индекс датасета", "", f"Всего файлов: **{len(df)}**", ""]

    out += ["## Файлы по split и классам", "", _md_table(_crosstab(df["cls"], df["split"])), ""]
    orig = df[df["is_original"]]
    out += ["## Только оригиналы (aug = orig)", "", _md_table(_crosstab(orig["cls"], orig["split"])), ""]

    out += ["## Схемы имен (source) по split", "", _md_table(_crosstab(df["source"], df["split"])), ""]
    for split in ("train", "test"):
        s = df[df["split"] == split]
        out += [f"### source × класс, {split}", "", _md_table(_crosstab(s["source"], s["cls"])), ""]

    out += ["## Типы аугментаций по split", "", _md_table(_crosstab(df["aug"], df["split"])), ""]

    # Полнота групп aug_GMPN внутри одного split
    g = df[df["source"] == "aug_GMPN"]
    grp = g.groupby(["split", "base_id"]).agg(n=("aug", "size"), has_orig=("is_original", "any"))
    comp = grp.groupby("split").agg(
        groups=("n", "size"),
        full_7=("n", lambda s: int((s == 7).sum())),
        partial=("n", lambda s: int((s < 7).sum())),
        no_original=("has_orig", lambda s: int((~s).sum())),
    )
    out += [
        "## Группы aug_GMPN внутри split",
        "",
        "full_7 — оригинал + 6 аугментаций; partial — неполные; no_original — в этом split нет оригинала группы.",
        "",
        _md_table(comp),
        "",
    ]

    mism = df[df["name_cls_mismatch"]]
    out += [f"## Класс в имени ≠ папке: {len(mism)}", ""]
    out += [f"- `{p}`" for p in mism["path"]] or ["нет"]
    out += [""]

    unk = df[df["source"] == "unknown"]
    out += [f"## Нераспознанные имена: {len(unk)}", ""]
    out += [f"- `{p}`" for p in unk["path"]] or ["нет"]
    out += [""]

    dup = df[df.duplicated("base_id", keep=False) & (df["source"] != "aug_GMPN")]
    out += [
        f"## Совпадения base_id вне схемы aug_GMPN: {dup['base_id'].nunique()} групп, {len(dup)} файлов",
        "",
        "Совпадение по имени — кандидат в дубликат; подтверждается по хешам на шаге 1.2.",
        "",
    ]
    if len(dup):
        out += [_md_table(_crosstab(dup["source"], dup["split"])), ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("mrt"), help="корень исходного датасета (read-only)")
    ap.add_argument("--out", type=Path, default=Path("data/processed/index.csv"))
    ap.add_argument("--report", type=Path, default=Path("reports/index_report.md"))
    args = ap.parse_args()

    df = build_index(args.data_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")

    report = make_report(df)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"index: {args.out} ({len(df)} строк)\nreport: {args.report}")


if __name__ == "__main__":
    main()
