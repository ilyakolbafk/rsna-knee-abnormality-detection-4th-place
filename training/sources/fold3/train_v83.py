"""Fixed fold/full training for the verified global AUX60 v8.3 cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import h1_base as h
from models import loss_parts, make_model, positive_weights
from resnet_model import make as make_resnet
from v83_data import V83Bags, attach_cache_paths, gold_auc, parent_key, pseudo_auc

ROOT = Path(__file__).resolve().parent
EXPECTED_BUILDER = "935625ae385df67cba32d11aca08a3cce7f3ec27d42313be79edc0236e9c483a"
EXPECTED_POLICY = "knee-laterality-v8.3"
EXPECTED_STATE = "canonical_L_v8_3"
SEED = 20260908


def sha(path):
    hsh = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            hsh.update(block)
    return hsh.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def guard():
    path = ROOT / "BUDGET_GUARD.json"
    if not path.exists():
        raise RuntimeError("BUDGET_GUARD.json is required on paid GPU")
    value = json.loads(path.read_text())
    if time.time() >= float(value["hard_stop_unix"]) - 600:
        raise RuntimeError("Funded deadline reached, including 10-minute backup reserve")
    return value


def code_hashes():
    names = ["train_v83.py", "v83_data.py", "h1_base.py", "models.py", "resnet_model.py"]
    return {name: sha(ROOT / name) for name in names}


def load_data(cache_root: Path, aux_names):
    receipt = json.loads((cache_root / "GLOBAL_RELEASE_RECEIPT.json").read_text())
    required = {
        "status": "PASS_GLOBAL_AUX60_V83_RELEASE",
        "builder_sha256": EXPECTED_BUILDER,
        "orientation_policy": EXPECTED_POLICY,
        "orientation_state": EXPECTED_STATE,
        "train_parents": 4345,
        "train_samples": 4352,
        "gold_studies": 58,
    }
    for key, want in required.items():
        if receipt.get(key) != want:
            raise ValueError(f"Cache release mismatch for {key}: {receipt.get(key)!r} != {want!r}")
    labels = pd.read_csv(cache_root / "TRAIN_LABELS.csv")
    manifest = pd.read_csv(cache_root / "TRAIN_MANIFEST.csv")
    train = attach_cache_paths(labels, manifest, cache_root)
    if len(train) != 4352 or train.SampleID.nunique() != 4352:
        raise ValueError("Expected 4,352 unique v8.3 train samples")
    if train.StudyInstanceUID.nunique() != 4345:
        raise ValueError("Expected 4,345 train parents")
    if train.groupby("report_group").fold.nunique().max() != 1:
        raise ValueError("A report group crosses folds")
    active = train[["label_mask__" + t for t in h.TARGETS]].astype(bool).any(axis=1)
    if int(active.sum()) != 4350:
        raise ValueError(f"Expected 4,350 rows with active main labels, got {int(active.sum())}")
    gold_labels = pd.read_csv(cache_root / "GOLD58_LABELS.csv")
    gold_manifest = pd.read_csv(cache_root / "GOLD58_MANIFEST.csv")
    gold = gold_labels.copy()
    gold["SampleID"] = gold.StudyInstanceUID.astype(str)
    gold["ParentStudyInstanceUID"] = gold.StudyInstanceUID.astype(str)
    gold["report_group"] = gold.StudyInstanceUID.astype(str)
    gold["fold"] = -1
    for target in h.TARGETS:
        gold["label_mask__" + target] = 1
        gold["confidence__" + target] = 1.0
    for name in aux_names:
        gold["aux__" + name] = 0.0
        gold["confidence__aux__" + name] = 0.0
        gold["mask__aux__" + name] = 0.0
    gold = attach_cache_paths(gold, gold_manifest, cache_root)
    if len(gold) != 58 or set(gold.StudyInstanceUID) & set(train.StudyInstanceUID):
        raise ValueError("Gold58 cardinality or train overlap failure")
    return train, gold, receipt


def make_loader(frame, aux_names, train, epoch, batch, workers):
    ds = V83Bags(frame, aux_names, train=train, seed=SEED, epoch=epoch)
    return h.loader(ds, batch if train else 1, workers, SEED + epoch, train)


@torch.inference_mode()
def predict(model, frame, aux_names, device, workers):
    model.eval()
    result = np.full((len(frame), 60), np.nan, np.float32)
    data = make_loader(frame, aux_names, False, 0, 1, workers)
    started = time.monotonic()
    for x, mask, meta, _y, _lw, index, gain in data:
        guard()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(
                x.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
                meta.to(device, non_blocking=True),
                gain.to(device, non_blocking=True),
            )
        result[index.numpy()] = logits.float().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("Incomplete validation predictions")
    return result, time.monotonic() - started


def save_predictions(path, frame, aux_names, logits):
    out = frame[["SampleID", "StudyInstanceUID", "ParentStudyInstanceUID", "report_group"]].copy()
    probabilities = torch.from_numpy(logits).sigmoid().numpy()
    for i, name in enumerate(h.TARGETS + list(aux_names)):
        out[name] = probabilities[:, i]
    out.to_csv(path, index=False)


def evaluate_fold(frame, probabilities):
    return {
        "sample": pseudo_auc(frame, probabilities, None),
        "parent": pseudo_auc(frame, probabilities, parent_key(frame)),
        "report_group": pseudo_auc(frame, probabilities, "report_group"),
    }


def make_network(model_name, assets):
    if model_name == "coatnet":
        return make_model("AUX60", assets, chunk=96), {
            "family": "CoAtNet-rmlp-2-384 AUX60",
            "initialization": "ImageNet-12k-to-1k",
            "pretrained_sha256": sha(assets / "coatnet_imagenet.safetensors"),
        }
    model, identity = make_resnet("R5", assets)
    return model, {"family": "ResNet50-224 AUX60 R5", **identity}


def train(args):
    budget = guard()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Each process must see exactly one physical GPU through CUDA_VISIBLE_DEVICES")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("RTX 5090 BF16 support is required")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    h.seed_all(SEED)

    assets = ROOT / "assets"
    aux_names = json.loads((assets / "aux_names.json").read_text())
    if len(aux_names) != 48:
        raise ValueError("AUX60 requires exactly 48 auxiliary heads")
    cache_root = Path(args.cache_root)
    frame, gold, cache_receipt = load_data(cache_root, aux_names)
    val = frame.loc[frame.fold == args.fold].copy().reset_index(drop=True)
    fit = frame.copy() if args.full else frame.loc[frame.fold != args.fold].copy().reset_index(drop=True)
    if not args.full and set(fit.report_group) & set(val.report_group):
        raise ValueError("Fold leakage through report_group")
    if args.full:
        val = val.iloc[:0].copy()

    cfg = (
        {"schedule": 24, "stop": 20, "average": [18, 19, 20], "batch": 8, "weight_decay": .02}
        if args.model == "coatnet"
        else {"schedule": 48, "stop": 40, "average": [38, 39, 40], "batch": 8, "weight_decay": .05}
    )
    model, model_identity = make_network(args.model, assets)
    model = model.to(device)
    steps = math.ceil(len(fit) / cfg["batch"])
    if args.model == "coatnet":
        from models import optimizer

        opt, scheduler = optimizer(model, cfg["schedule"], steps, False)
    else:
        backbone = list(model.backbone.parameters())
        backbone_ids = {id(p) for p in backbone}
        head = [p for p in model.parameters() if id(p) not in backbone_ids]
        opt = torch.optim.AdamW(
            [{"params": backbone, "lr": 1e-4}, {"params": head, "lr": 1e-3}],
            weight_decay=cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[1e-4, 1e-3], total_steps=cfg["schedule"] * steps, pct_start=.15
        )
    pos_weight = positive_weights(fit, aux_names).to(device)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    identity = {
        "run": args.run_name,
        "model": args.model,
        "full": bool(args.full),
        "fold": None if args.full else args.fold,
        "seed": SEED,
        "recipe": cfg,
        "aux_weight": .25,
        "windows_per_train_sample": 12,
        "validation_tokens": "all valid up to 64",
        "validation_each_epoch": True,
        "fixed_checkpoint_average": cfg["average"],
        "gold_fit_or_selection": False,
        "orientation_policy": EXPECTED_POLICY,
        "orientation_state": EXPECTED_STATE,
        "builder_sha256": EXPECTED_BUILDER,
        "cache_release_sha256": sha(cache_root / "GLOBAL_RELEASE_RECEIPT.json"),
        "training_sample_ids_sha256": h.semantic(sorted(fit.SampleID.astype(str))),
        "validation_parent_ids_sha256": h.semantic(sorted(val.StudyInstanceUID.astype(str).unique())),
        "gold_ids_sha256": h.semantic(sorted(gold.StudyInstanceUID.astype(str))),
        "code_hashes": code_hashes(),
        "instance_binding": {
            "instance_id": budget["instance_id"],
            "hourly_usd": budget["hourly_usd"],
            "gpu_count": budget["gpu_count"],
            "gpu_name": budget["gpu_name"],
        },
        "model_identity": model_identity,
        "steps_per_epoch": steps,
    }
    atomic_json(out / "IDENTITY.json", identity)
    atomic_json(out / "POS_WEIGHT.json", dict(zip(h.TARGETS + aux_names, pos_weight.cpu().tolist())))
    fit[["SampleID", "StudyInstanceUID", "ParentStudyInstanceUID", "report_group", "fold"]].to_csv(
        out / "fit_ids.csv", index=False
    )
    if not args.full:
        val[["SampleID", "StudyInstanceUID", "ParentStudyInstanceUID", "report_group", "fold"]].to_csv(
            out / "validation_ids.csv", index=False
        )

    start = updates = 0
    history = []
    latest = out / "latest.pt"
    if latest.exists():
        state = h.load_checkpoint(latest)
        if state["identity"] != identity:
            raise ValueError("Checkpoint identity drift")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        start = int(state["epoch"])
        updates = int(state["updates"])
        torch.set_rng_state(state["rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])

    started = time.monotonic()
    for epoch_index in range(start, cfg["stop"]):
        epoch = epoch_index + 1
        train_loader = make_loader(fit, aux_names, True, epoch_index, cfg["batch"], args.workers)
        model.train()
        opt.zero_grad(set_to_none=True)
        t0 = last = time.monotonic()
        wait = loss_sum = main_sum = aux_sum = 0.0
        count = 0
        torch.cuda.reset_peak_memory_stats(device)
        for step, (x, mask, meta, y, label_weight, _index, gain) in enumerate(train_loader):
            wait += time.monotonic() - last
            guard()
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            label_weight = label_weight.to(device, non_blocking=True)
            gain = gain.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, mask, meta, gain)
                main, aux, losses = loss_parts(logits, y, label_weight, pos_weight, aux_weight=.25)
                loss = losses.sum()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            n = len(x)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(n)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
            opt.step()
            scheduler.step()
            opt.zero_grad(set_to_none=True)
            updates += 1
            count += n
            loss_sum += float(loss.detach())
            main_sum += float(main.sum().detach())
            aux_sum += float(aux.sum().detach())
            if step % 25 == 0:
                atomic_json(
                    out / "LIVE.json",
                    {
                        "run": args.run_name,
                        "phase": "train",
                        "epoch": epoch,
                        "epochs": cfg["stop"],
                        "batch": step + 1,
                        "batches": len(train_loader),
                        "samples": count,
                        "train_loss": loss_sum / count,
                        "main_loss": main_sum / count,
                        "aux_loss": aux_sum / count,
                        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                        "utc": time.time(),
                    },
                )
            last = time.monotonic()
        torch.cuda.synchronize(device)
        train_seconds = time.monotonic() - t0
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / count,
            "main_loss": main_sum / count,
            "aux_loss": aux_sum / count,
            "train_seconds": train_seconds,
            "data_wait_seconds": wait,
            "data_wait_fraction": wait / train_seconds,
            "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "updates": updates,
            "lr": [group["lr"] for group in opt.param_groups],
        }
        if not args.full:
            atomic_json(out / "LIVE.json", {"run": args.run_name, "phase": "fold_validation", "epoch": epoch, "utc": time.time()})
            fold_logits, fold_seconds = predict(model, val, aux_names, device, args.workers)
            fold_prob = torch.from_numpy(fold_logits[:, :12]).sigmoid().numpy()
            record["pseudo"] = evaluate_fold(val, fold_prob)
            record["fold_validation_seconds"] = fold_seconds
            save_predictions(out / f"fold_predictions_e{epoch:02d}.csv", val, aux_names, fold_logits)
        atomic_json(out / "LIVE.json", {"run": args.run_name, "phase": "gold_validation", "epoch": epoch, "utc": time.time()})
        gold_logits, gold_seconds = predict(model, gold, aux_names, device, args.workers)
        gold_prob = torch.from_numpy(gold_logits[:, :12]).sigmoid().numpy()
        record["gold"] = gold_auc(gold, gold_prob)
        record["gold_validation_seconds"] = gold_seconds
        save_predictions(out / f"gold_predictions_e{epoch:02d}.csv", gold, aux_names, gold_logits)
        history.append(record)
        cpu_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch in cfg["average"]:
            h.atomic_torch(out / f"weights-e{epoch:02d}.pt", {"model": cpu_state, "identity": identity, "epoch": epoch})
        h.atomic_torch(
            latest,
            {
                "model": cpu_state,
                "identity": identity,
                "epoch": epoch,
                "updates": updates,
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
            },
        )
        atomic_json(out / "history.json", history)
        print(json.dumps(record, allow_nan=False), flush=True)

    states = [h.load_checkpoint(out / f"weights-e{epoch:02d}.pt")["model"] for epoch in cfg["average"]]
    average = h.average_states(states)
    model.load_state_dict(average)
    h.atomic_torch(out / "averaged.pt", {"model": average, "identity": identity, "selected_epochs": cfg["average"]})
    result = {
        "status": "PASS",
        "run": args.run_name,
        "completed_epochs": cfg["stop"],
        "selected_epochs": cfg["average"],
        "selection_rule": "fixed; no held-out or Gold checkpoint selection",
        "averaged_sha256": sha(out / "averaged.pt"),
        "elapsed_seconds": time.monotonic() - started,
    }
    if not args.full:
        logits, seconds = predict(model, val, aux_names, device, args.workers)
        prob = torch.from_numpy(logits[:, :12]).sigmoid().numpy()
        result["pseudo"] = evaluate_fold(val, prob)
        result["fold_validation_seconds"] = seconds
        save_predictions(out / "averaged_fold_predictions.csv", val, aux_names, logits)
    logits, seconds = predict(model, gold, aux_names, device, args.workers)
    prob = torch.from_numpy(logits[:, :12]).sigmoid().numpy()
    result["gold"] = gold_auc(gold, prob)
    result["gold_validation_seconds"] = seconds
    save_predictions(out / "averaged_gold_predictions.csv", gold, aux_names, logits)
    if args.model == "coatnet":
        h.atomic_torch(
            out / "main12.pt",
            {
                "model": {key: value for key, value in average.items() if not key.startswith(("aux_att.", "auxW", "auxb"))},
                "identity": identity,
                "selected_epochs": cfg["average"],
            },
        )
    atomic_json(out / "RESULT.json", result)
    atomic_json(out / "LIVE.json", {"run": args.run_name, "phase": "complete", "utc": time.time(), **result})
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["coatnet", "r5"], required=True)
    parser.add_argument("--fold", type=int, default=3)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--workers", type=int, default=8)
    train(parser.parse_args())
