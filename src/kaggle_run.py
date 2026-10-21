"""Запуск обучения в Kaggle (или в любой папке без git): находит данные, запускает список обучений,
упаковывает результаты.

Запуск задается как «конфиг:seed», например run1_cls:43 → configs/run1_cls.yaml, seed 43.
Данные ищутся в --input-dir: папка, где лежат pool.csv и pool_256.npy (кэш препроцессинга);
если набор данных остался архивом data_bundle.zip, он распаковывается в --work-dir/data.
Результаты пишутся в --work-dir: models/<запуск>/, experiments.csv и архив results_<время>.zip.

Пример (ячейка ноутбука Kaggle):
    python -m src.kaggle_run --runs run0_cls:42 run0_seg:42
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import zipfile
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from src.config import load_config

REPO = Path(__file__).resolve().parents[1]


def find_data(input_dir: Path, work_dir: Path) -> Path:
    def ready(d: Path) -> bool:
        return all((d / n).exists() for n in ("pool.csv", "pool_256.npy", "pool_256_mask.npy", "pool_256.csv", "pool_256.json"))

    for root in (input_dir, work_dir / "data"):
        for f in sorted(root.rglob("pool.csv")) if root.exists() else []:
            if ready(f.parent):
                return f.parent
    for z in sorted(input_dir.rglob("data_bundle.zip")):
        target = work_dir / "data"
        print(f"распаковка {z} → {target}", flush=True)
        with zipfile.ZipFile(z) as arc:
            arc.extractall(target)
        if ready(target):
            return target
    raise SystemExit(f"в {input_dir} нет папки с pool.csv и кэшем pool_256.* — подключите набор данных с data_bundle.zip (Add Input)")


def pinned() -> dict:
    """Версии из requirements-kaggle.txt: {пакет: версия}."""
    req = REPO / "requirements-kaggle.txt"
    lines = [x.split("#")[0].strip() for x in req.read_text(encoding="utf-8").splitlines()] if req.exists() else []
    return dict(x.split("==") for x in lines if "==" in x)


def environment() -> dict:
    import torch

    info = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "нет",
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if torch.cuda.is_available() else 0}
    for pkg, want in pinned().items():
        try:
            info[pkg] = version(pkg)
        except PackageNotFoundError:
            info[pkg] = "не установлен"
        if info[pkg] != want:
            print(f"ВНИМАНИЕ: {pkg} {info[pkg]}, а код проверен на {want} — выполните pip install -r requirements-kaggle.txt")
    commit = REPO / "COMMIT"
    info["commit"] = commit.read_text(encoding="utf-8").strip() if commit.exists() else "файла COMMIT нет"
    return info


def check_weights(configs: list[dict]) -> None:
    """Предобученные веса скачиваются из интернета — проверяем это до начала обучения."""
    import timm

    for enc in sorted({c["model"]["encoder"] for c in configs if c["model"]["pretrained"]}):
        try:
            timm.create_model(enc, pretrained=True, num_classes=0)
        except Exception as e:  # noqa: BLE001 — любая ошибка загрузки здесь означает одно: весов нет
            raise SystemExit(f"не удалось загрузить предобученные веса {enc}: {type(e).__name__}: {e}\n"
                             "включите интернет в настройках ноутбука (Settings → Internet → On)") from e
        print(f"предобученные веса {enc}: загружены", flush=True)


def pack(work: Path, with_weights: bool) -> Path | None:
    files = [f for f in sorted((work / "models").rglob("*")) if f.is_file() and (with_weights or f.suffix != ".pt")] if (work / "models").exists() else []
    if not files:
        return None
    out = work / f"results_{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        if (work / "experiments.csv").exists():
            z.write(work / "experiments.csv", "experiments.csv")
        for f in files:
            z.write(f, f.relative_to(work).as_posix())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="список «конфиг:seed», например run0_cls:42")
    ap.add_argument("--input-dir", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--work-dir", type=Path, default=Path("/kaggle/working"))
    ap.add_argument("--num-workers", type=int, default=3)
    ap.add_argument("--extra", default="", help="дополнительные аргументы src.train (в кавычках)")
    ap.add_argument("--no-weights", action="store_true", help="не класть веса в архив результатов")
    ap.add_argument("--allow-cpu", action="store_true", help="разрешить запуск без видеокарты (отладка)")
    args = ap.parse_args()

    runs = []
    for spec in args.runs:
        name, _, seed = spec.partition(":")
        config = REPO / "configs" / f"{name}.yaml"
        if not config.exists() or not (seed or "42").isdigit():
            raise SystemExit(f"запуск {spec!r}: нужен вид «конфиг:seed», конфиг {config} {'найден' if config.exists() else 'не найден'}")
        runs.append((spec, f"configs/{name}.yaml", seed or "42", load_config(config)))

    env = environment()
    print("окружение:", json.dumps(env, ensure_ascii=False), flush=True)
    if not env["cuda"] and not args.allow_cpu:
        raise SystemExit("видеокарта не найдена: включите GPU в настройках ноутбука (Settings → Accelerator → GPU T4)")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    data = find_data(args.input_dir, args.work_dir)
    info = json.loads((data / "data_info.json").read_text(encoding="utf-8")) if (data / "data_info.json").exists() else {}
    print("данные:", data, json.dumps(info, ensure_ascii=False), flush=True)
    extra = shlex.split(args.extra)
    if "--no-pretrained" not in extra:
        check_weights([c for *_, c in runs])

    failed = None
    for spec, config, seed, _ in runs:
        cmd = [sys.executable, "-m", "src.train", "--config", config, "--seed", seed,
               "--processed-dir", str(data), "--cache-dir", str(data), "--out-dir", str(args.work_dir / "models"),
               "--experiments", str(args.work_dir / "experiments.csv"), "--num-workers", str(args.num_workers), *extra]
        print(f"\n=== {spec} ===", flush=True)
        code = subprocess.run(cmd, cwd=REPO).returncode
        if code != 0:
            failed = f"запуск {spec} завершился с кодом {code}; следующие запуски не выполнялись"
            break
    out = pack(args.work_dir, not args.no_weights)
    if out:
        print(f"\nрезультаты: {out} ({out.stat().st_size / 2**20:.0f} МБ) — скачайте этот файл и внесите в проект: python -m src.import_kaggle <файл>")
    if failed:
        raise SystemExit(failed)


if __name__ == "__main__":
    main()
