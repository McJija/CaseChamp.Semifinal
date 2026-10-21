"""Загрузка датасета Figshare (Cheng et al., brain tumor dataset, CC BY 4.0).

Список файлов и md5 берутся из API Figshare; уже скачанные файлы с верным md5
пропускаются. Архивы распаковываются в <out>/mat/.

Пример:
    python -m src.download_figshare --out data/external/figshare
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
import zipfile
from pathlib import Path

ARTICLE_ID = 1512427
API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
CHUNK = 1 << 20
RETRIES = 12


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total, done, step = int(r.headers.get("Content-Length", 0)), 0, 0
        while chunk := r.read(CHUNK):
            f.write(chunk)
            done += len(chunk)
            if done // (50 * CHUNK) > step:  # сообщение каждые 50 МБ
                step = done // (50 * CHUNK)
                print(f"  {dst.name}: {done >> 20} / {total >> 20} МБ", flush=True)
    tmp.replace(dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/external/figshare"))
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()

    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(API_URL, headers={"User-Agent": "Mozilla/5.0"})
    meta = json.load(urllib.request.urlopen(req, timeout=60))
    (args.out / "figshare_article.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{meta['title']} | версия {meta.get('version')} | {meta.get('license', {}).get('name')}", flush=True)

    for f in meta["files"]:
        dst = raw / f["name"]
        want = f.get("computed_md5") or f.get("supplied_md5")
        if dst.exists() and md5_of(dst) == want:
            print(f"OK (уже есть)  {f['name']}", flush=True)
            continue
        print(f"скачиваю {f['name']} ({f['size'] >> 20} МБ)", flush=True)
        for attempt in range(1, RETRIES + 1):
            try:
                download(f["download_url"], dst)
                break
            except OSError as e:  # URLError, обрыв соединения, временный сбой DNS
                print(f"  попытка {attempt}/{RETRIES} не удалась: {e}", flush=True)
                if attempt == RETRIES:
                    raise
                time.sleep(min(15 * attempt, 90))
        got = md5_of(dst)
        if got != want:
            raise SystemExit(f"ОШИБКА md5: {f['name']}: {got} != {want}")
        print(f"OK md5  {f['name']}", flush=True)

    if not args.no_extract:
        mat = args.out / "mat"
        mat.mkdir(exist_ok=True)
        for z in sorted(raw.glob("*.zip")):
            with zipfile.ZipFile(z) as zf:
                names = [n for n in zf.namelist() if n.endswith(".mat")]
                todo = [n for n in names if not (mat / Path(n).name).exists()]
                for n in todo:
                    (mat / Path(n).name).write_bytes(zf.read(n))
                print(f"распаковано {z.name}: {len(names)} .mat (новых {len(todo)})", flush=True)
        print(f"всего .mat: {len(list(mat.glob('*.mat')))}", flush=True)


if __name__ == "__main__":
    main()
