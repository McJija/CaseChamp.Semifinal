"""Обучение классификатора, сегментатора или многозадачной сети.

Данные — общий пул (data/processed/pool.csv) и кэш препроцессинга `pool_<size>`; test не используется.
Деление данных фиксировано в пуле; seed обучения задает инициализацию голов, порядок батчей и аугментации.
Дообучение двухфазное: первые `epochs_frozen` эпох энкодер заморожен, дальше учится вся сеть.
Эпоха выбирается по правилам из src/metrics.py.

Результат запуска — папка <out_dir>/<run>_s<seed>_<время>/:
  best.pt (веса выбранной эпохи, конфиг, метрики), history.csv, val_predictions.csv, config.yaml
и строка в reports/experiments.csv.

Примеры:
    python -m src.train --config configs/run1_cls.yaml
    python -m src.train --config configs/run1_seg.yaml --seed 43
    python -m src.train --config configs/run0_cls.yaml --limit-train 64 --limit-val 48 --no-pretrained --device cpu
"""
from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from src.config import load_config
from src.dataset import CACHE_VERSION, DataConfig, PreprocCache, class_weights, make_loader, seed_everything
from src.labels import CLASSES
from src.metrics import candidate_epochs, classification_metrics, dice_iou, segmentation_metrics, select_epoch
from src.models import build_model, forward, set_encoder_frozen, set_train_mode, split_params

REPO = Path(__file__).resolve().parents[1]


def get_commit() -> str:
    """Хеш коммита: из файла COMMIT рядом с кодом (архив для Kaggle), иначе из git."""
    f = REPO / "COMMIT"
    if f.exists():
        return f.read_text(encoding="utf-8").strip()
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=15)
        if h.returncode == 0 and h.stdout.strip():
            dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, capture_output=True, text=True, timeout=15)
            return h.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def limit(table: pd.DataFrame, n: int | None) -> pd.DataFrame:
    """Небольшая подвыборка для отладки: поровну из каждого класса, сначала снимки с маской."""
    if not n or n >= len(table):
        return table
    per = int(np.ceil(n / table["cls"].nunique()))
    parts = [x.sort_values("has_mask", ascending=False, kind="stable").head(per) for _, x in table.groupby("cls")]
    return pd.concat(parts).sort_index()


def seg_loss(mask_logits: torch.Tensor, masks: torch.Tensor, has_mask: torch.Tensor) -> torch.Tensor:
    """Dice + BCE по снимкам, у которых маска известна."""
    if not bool(has_mask.any()):
        return mask_logits.sum() * 0.0
    ml, mt = mask_logits[has_mask].float(), masks[has_mask].float()
    bce = F.binary_cross_entropy_with_logits(ml, mt)
    p = torch.sigmoid(ml)
    inter = (p * mt).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + mt.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


def compute_loss(out: dict, batch: dict, kind: str, weights: torch.Tensor | None, tc: dict, device) -> torch.Tensor:
    loss = 0.0
    if "logits" in out:
        cls = F.cross_entropy(out["logits"].float(), batch["label"].to(device), weight=weights, label_smoothing=tc["label_smoothing"])
        loss = loss + (tc["cls_loss_weight"] if kind == "multitask" else 1.0) * cls
    if "mask_logits" in out:
        loss = loss + seg_loss(out["mask_logits"], batch["mask"].to(device), batch["has_mask"].to(device))
    return loss


@torch.no_grad()
def evaluate(model, kind: str, loader, table: pd.DataFrame, device, ec: dict) -> tuple[dict, pd.DataFrame]:
    """Метрики на val и таблица предсказаний (строка на снимок, порядок как в table).

    Считается в обычной точности, без смешанной: так числа из Kaggle совпадают с пересчетом на CPU.
    """
    model.eval()
    probs, rows = [], []
    for batch in loader:
        out = forward(model, kind, batch["image"].to(device, non_blocking=True))
        if "logits" in out:
            probs.append(torch.softmax(out["logits"].float(), 1).cpu().numpy())
        if "mask_logits" in out:
            pred = (torch.sigmoid(out["mask_logits"].float()) > ec["mask_threshold"]).cpu().numpy()[:, 0]
            true = batch["mask"].numpy()[:, 0] > 0.5
            for k in range(len(pred)):
                d, i = dice_iou(pred[k], true[k]) if bool(batch["has_mask"][k]) else (np.nan, np.nan)
                rows.append({"dice": d, "iou": i, "pred_area": int(pred[k].sum()), "true_area": int(true[k].sum())})
    frame = table[["path", "cls", "has_mask", "source"]].reset_index(drop=True).copy()
    frame["label"] = [CLASSES.index(c) for c in frame["cls"]]
    metrics = {}
    if probs:
        p = np.concatenate(probs)
        for k, c in enumerate(CLASSES):
            frame[f"p_{c}"] = p[:, k]
        frame["pred"] = [CLASSES[k] for k in p.argmax(1)]
        metrics.update(classification_metrics(frame["label"].to_numpy(), p))
    if rows:
        frame = pd.concat([frame, pd.DataFrame(rows)], axis=1)
        metrics.update(segmentation_metrics(frame, ec["min_mask_area_px"]))
    return metrics, frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=None, help="seed обучения (деление данных от него не зависит)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--epochs-frozen", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--limit-train", type=int, default=None, help="подвыборка train для отладки")
    ap.add_argument("--limit-val", type=int, default=None)
    ap.add_argument("--no-pretrained", action="store_true", help="не загружать предобученные веса (отладка)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("models"))
    ap.add_argument("--experiments", type=Path, default=Path("reports/experiments.csv"))
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    kind, mc, tc, ec = cfg["kind"], cfg["model"], cfg["train"], cfg["eval"]
    for key, val in (("seed", args.seed), ("epochs", args.epochs), ("epochs_frozen", args.epochs_frozen)):
        if val is not None:
            tc[key] = val
    if args.no_pretrained:
        mc["pretrained"] = False
    known = {f.name for f in fields(DataConfig)}
    over = {"batch_size": args.batch_size, "num_workers": args.num_workers, "processed_dir": args.processed_dir, "cache_dir": args.cache_dir}
    data = DataConfig(**{**{k: v for k, v in cfg["data"].items() if k in known}, **{k: v for k, v in over.items() if v is not None}})
    data = replace(data, seed=tc["seed"])  # порядок батчей и аугментации следуют seed обучения
    cfg["data"] = {f.name: getattr(data, f.name) for f in fields(DataConfig)}

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(tc["amp"]) and device.type == "cuda"
    seed_everything(tc["seed"])
    commit = get_commit()

    # ---------- данные
    pool = pd.read_csv(Path(data.processed_dir) / "pool.csv", keep_default_na=False)
    pool["has_mask"] = pool["has_mask"].astype(str).str.lower() == "true"
    cache = PreprocCache(data.cache_dir, "pool", data.size)
    missing = ~pool["path"].isin(cache.table["path"])
    if missing.any():
        raise ValueError(f"в кэше нет {int(missing.sum())} снимков пула — пересоберите его: python -m src.build_pool")
    train, val = pool[pool["split"] == "train"], pool[pool["split"] == "val"]
    with_mask = kind != "classifier"
    negatives = kind == "segmenter" and tc["seg_negatives"]["enabled"]
    if kind == "segmenter":
        if negatives:  # снимки «нет опухоли»: маска известна и пуста, ошибка сегментации по ним считается
            train = train[train["has_mask"] | (train["cls"] == "no_tumor")].copy()
            train["has_mask"] = True
        else:
            train = train[train["has_mask"]]
    train, val = limit(train, args.limit_train).reset_index(drop=True), limit(val, args.limit_val).reset_index(drop=True)
    crop = tc["seg_crop"] if with_mask and tc["seg_crop"]["enabled"] else None
    cf = tc["counterfactual"] if tc["counterfactual"]["enabled"] else None
    train_loader = make_loader(train, data, train=True, cache=cache, with_mask=with_mask, crop=crop, counterfactual=cf)
    val_loader = make_loader(val, replace(data, domain_randomization=False), train=False, cache=cache, with_mask=with_mask)
    weights = class_weights(train, cf["p_remove"] if cf else 0.0).to(device) if kind != "segmenter" and tc["class_weights"] else None

    # ---------- модель и оптимизатор
    model = build_model(kind, mc["encoder"], mc["pretrained"], mc.get("dropout", 0.2)).to(device)
    enc_params, head_params = split_params(model, kind)
    opt = torch.optim.AdamW([{"params": enc_params, "lr": tc["lr_encoder"]}, {"params": head_params, "lr": tc["lr_head"]}],
                            weight_decay=tc["weight_decay"])
    steps = len(train_loader) * tc["epochs"]
    warm = max(1, len(train_loader) * tc["warmup_epochs"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / max(steps, 1))))
    scaler = torch.amp.GradScaler(enabled=use_amp)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / f"{cfg['run']}_s{tc['seed']}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    dirty_note = " (есть незакоммиченные правки)" if commit.endswith("-dirty") else ""
    print(f"запуск {run_dir.name}: {kind}, {mc['encoder']}, устройство {device}, коммит {commit[:12]}{dirty_note}, "
          f"train {len(train)}, val {len(val)}, батчей в эпохе {len(train_loader)}", flush=True)

    # ---------- обучение
    history, t_start = [], time.perf_counter()
    for epoch in range(1, tc["epochs"] + 1):
        frozen = epoch <= tc["epochs_frozen"]
        set_encoder_frozen(model, kind, frozen)
        set_train_mode(model, kind, frozen)
        t0, total, n = time.perf_counter(), 0.0, 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = forward(model, kind, batch["image"].to(device, non_blocking=True))
                loss = compute_loss(out, batch, kind, weights, tc, device)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += float(loss.detach()) * len(batch["label"])
            n += len(batch["label"])
        t1 = time.perf_counter()
        metrics, _ = evaluate(model, kind, val_loader, val, device, ec)
        row = {"epoch": epoch, "phase": "головы" if frozen else "вся сеть", "train_loss": total / max(n, 1),
               "lr_head": opt.param_groups[1]["lr"], "sec_train": t1 - t0, "sec_val": time.perf_counter() - t1, **metrics}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8")
        torch.save(model.state_dict(), run_dir / f"epoch_{epoch:03d}.pt")
        keep = candidate_epochs(history, kind, ec["recall_floor"])
        for f in run_dir.glob("epoch_*.pt"):  # храним только веса эпох, которые еще могут быть выбраны
            if int(f.stem.split("_")[1]) not in keep:
                f.unlink()
        shown = {k: row[k] for k in ("macro_f1", "recall_tumor", "specificity", "dice", "false_mask_rate") if k in row}
        print(f"эпоха {epoch:3d} ({row['phase']}): loss {row['train_loss']:.4f} | "
              + ", ".join(f"{k} {v:.3f}" for k, v in shown.items()) + f" | {row['sec_train']:.0f} с + {row['sec_val']:.0f} с", flush=True)
    train_sec = time.perf_counter() - t_start

    # ---------- выбор эпохи и проверка сохраненных весов
    best = select_epoch(history, kind, ec["recall_floor"], ec["dice_tolerance"])
    best_row = next(r for r in history if r["epoch"] == best)
    state = torch.load(run_dir / f"epoch_{best:03d}.pt", map_location=device)
    model.load_state_dict(state)
    metrics, frame = evaluate(model, kind, val_loader, val, device, ec)
    key = "dice" if kind == "segmenter" else "macro_f1"
    if abs(metrics[key] - best_row[key]) > 1e-4:  # допуск на численный шум
        raise RuntimeError(f"веса эпохи {best} после загрузки дают {key} = {metrics[key]:.6f}, а при обучении было {best_row[key]:.6f}")
    frame.to_csv(run_dir / "val_predictions.csv", index=False, encoding="utf-8")
    torch.save({"model": state, "kind": kind, "encoder": mc["encoder"], "classes": CLASSES, "config": cfg, "epoch": best,
                "metrics": metrics, "commit": commit, "seed": tc["seed"], "preprocess_version": CACHE_VERSION}, run_dir / "best.pt")
    for f in run_dir.glob("epoch_*.pt"):
        f.unlink()

    # ---------- строка в таблице экспериментов
    sec_epoch = float(np.mean([r["sec_train"] + r["sec_val"] for r in history]))
    exp = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "run": cfg["run"], "kind": kind, "encoder": mc["encoder"],
           "pretrained": mc["pretrained"], "seed": tc["seed"], "commit": commit, "config": args.config.as_posix(),
           "epochs": tc["epochs"], "epochs_frozen": tc["epochs_frozen"], "lr_head": tc["lr_head"], "lr_encoder": tc["lr_encoder"],
           "batch_size": data.batch_size, "domain_randomization": data.domain_randomization, "seg_crop": bool(crop), "counterfactual": bool(cf), "seg_negatives": negatives,
           "n_train": len(train), "n_val": len(val), "best_epoch": best,
           **{f"val_{k}": (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
           "sec_per_epoch": round(sec_epoch, 1), "train_min": round(train_sec / 60, 1), "device": str(device),
           "weights": (run_dir / "best.pt").as_posix(), "note": args.note}
    args.experiments.parent.mkdir(parents=True, exist_ok=True)
    table = pd.concat([pd.read_csv(args.experiments), pd.DataFrame([exp])]) if args.experiments.exists() else pd.DataFrame([exp])
    table.to_csv(args.experiments, index=False, encoding="utf-8")
    print(f"выбрана эпоха {best}: " + ", ".join(f"{k} {v:.3f}" for k, v in metrics.items() if isinstance(v, float) and not np.isnan(v)), flush=True)
    print(f"веса: {run_dir / 'best.pt'}; время {train_sec / 60:.1f} мин ({sec_epoch:.0f} с на эпоху); строка добавлена в {args.experiments}", flush=True)


if __name__ == "__main__":
    main()
