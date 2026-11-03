#!/usr/bin/env python3
"""Core v2 arm A short20: AUX60 CoAtNet-384 on the frozen global v8.3 cache + E1 augmentation block + short20 schedule.

Reference implementation derived line by line from gpu_package/train_v83.py (Claude 2026-09-18). Everything
that is not listed in prompts/codex/CORE_V2_ARM_A_SPEC_20260918.md section 5 is unchanged from v8.3:
labels, folds, Gold58 as validation only, 24 windows per training sample in this isolated K24 variant, all valid tokens at eval, batch 8,
AdamW 3e-5/1e-3 OneCycle pct 0.15, wd 0.02, aux weight 0.25, clip 3.0, BF16 encoder, FP32 heads.

Run the preflight (`preflight_core_v2_a.py`) and the CPU tests (`tests/run_core_v2_a_tests.py`) first.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = Path(os.environ.get("V83_PACKAGE_ROOT", HERE.parents[2] / "rsna-v83-big-train-20260915" / "gpu_package"))
sys.path[:0] = [str(HERE), str(PKG)]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import h1_base as h  # noqa: E402
from aug336 import BLUR_P, BLUR_SIGMA, BLUR_TAPS, CONTRAST, CROP_AREA, GAMMA, NOISE_P, NOISE_STD, ROTATION_DEG, SHIFT_FRACTION, BIAS_MAX  # noqa: E402
from core_v2_a_model import make  # noqa: E402
from models import loss_parts, optimizer, positive_weights  # noqa: E402
from token_slot_map import TokenSlotMap  # noqa: E402
from train_v83 import EXPECTED_BUILDER, EXPECTED_POLICY, EXPECTED_STATE, SEED, atomic_json, evaluate_fold, guard, load_data, save_predictions, sha  # noqa: E402
from v83_aug_data import AugConfig, V83AugBags, loader  # noqa: E402
from v83_data import gold_auc  # noqa: E402

IDENTITY_EXTRA = {}
ARM = "CORE_V2_A"
SCHEDULE = 24
STOP = 20
SWA_WINDOW = [18, 19, 20]                 # weights kept every epoch here
SPARSE_CHECKPOINTS = [14, 17, 20]    # plus every third epoch from 14
KEEP_WEIGHTS = sorted(set(SPARSE_CHECKPOINTS) | set(SWA_WINDOW))
REFERENCE_WINDOW = [18, 19, 20]                  # v8.3-style three-epoch average, diagnostic only
BATCH = 8
WEIGHT_DECAY = 0.02
AUX_WEIGHT = 0.25


def code_hashes():
    mine = ["train_core_v2_a.py", "aug336.py", "v83_aug_data.py", "core_v2_a_model.py", "token_slot_map.py"]
    theirs = ["train_v83.py", "v83_data.py", "h1_base.py", "models.py"]
    return {**{name: sha(HERE / name) for name in mine}, **{name: sha(PKG / name) for name in theirs}}


def average_states(states):
    if not states or any(set(s) != set(states[0]) for s in states):
        raise ValueError("Matching state schemas required")
    result = {}
    for key, first in states[0].items():
        values = [s[key] for s in states]
        if any(v.shape != first.shape or v.dtype != first.dtype for v in values):
            raise ValueError("State schema drift")
        result[key] = (sum(v.double() for v in values) / len(values)).to(first.dtype) if first.is_floating_point() else first.clone()
    return result


def make_loader(frame, aux_names, train, epoch, batch, workers, slot_map, config):
    dataset = V83AugBags(frame, aux_names, train=train, seed=SEED, epoch=epoch, slot_map=slot_map if train else None, config=config)
    return loader(dataset, batch if train else 1, workers, SEED + epoch, train)


@torch.inference_mode()
def predict(model, frame, aux_names, device, workers):
    model.eval()
    result = np.full((len(frame), 60), np.nan, np.float32)
    data = make_loader(frame, aux_names, False, 0, 1, workers, None, None)
    started = time.monotonic()
    for x, mask, meta, _y, _lw, index, gain, _params in data:
        guard()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x.to(device, non_blocking=True), mask.to(device, non_blocking=True),
                           meta.to(device, non_blocking=True), gain.to(device, non_blocking=True))
        result[index.numpy()] = logits.float().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("Incomplete validation predictions")
    return result, time.monotonic() - started


def fold_macro(frame, logits):
    return evaluate_fold(frame, torch.from_numpy(logits[:, :12]).sigmoid().numpy())["parent"]["macro_auc"]


def epoch_averaging_diagnostics(out, model, val, gold, aux_names, device, workers, identity, full):
    """SWA over 30-40 versus the mean of per-epoch predictions (spec section 6). No selection happens here."""
    report = {"swa_window": SWA_WINDOW, "reference_window": REFERENCE_WINDOW}
    if not full:
        per_epoch = {}
        for epoch in KEEP_WEIGHTS:
            path = out / f"fold_predictions_e{epoch:02d}.csv"
            if path.exists():
                probs = pd.read_csv(path)[h.TARGETS].to_numpy(np.float32)
                per_epoch[epoch] = probs
        report["fold_macro_per_epoch"] = {str(e): evaluate_fold(val, p)["parent"]["macro_auc"] for e, p in per_epoch.items()}
        for name, window in (("swa_window", SWA_WINDOW), ("reference_window", REFERENCE_WINDOW)):
            if all(e in per_epoch for e in window):
                mean = np.mean([per_epoch[e] for e in window], axis=0)
                report[f"prediction_mean_{name}_fold_macro"] = evaluate_fold(val, mean)["parent"]["macro_auc"]
    for name, window in (("swa_window", SWA_WINDOW), ("reference_window", REFERENCE_WINDOW)):
        states = [h.load_checkpoint(out / f"weights-e{epoch:02d}.pt")["model"] for epoch in window]
        model.load_state_dict(average_states(states))
        entry = {}
        if not full:
            logits, _ = predict(model, val, aux_names, device, workers)
            entry["fold_macro"] = fold_macro(val, logits)
            save_predictions(out / f"weight_average_{name}_fold_predictions.csv", val, aux_names, logits)
        logits, _ = predict(model, gold, aux_names, device, workers)
        entry["gold_macro"] = gold_auc(gold, torch.from_numpy(logits[:, :12]).sigmoid().numpy())["macro_auc"]
        report[f"weight_average_{name}"] = entry
    if not full and "prediction_mean_swa_window_fold_macro" in report:
        report["swa_minus_prediction_mean_fold_macro"] = report["weight_average_swa_window"]["fold_macro"] - report["prediction_mean_swa_window_fold_macro"]
        report["equivalence_rule"] = "SWA is accepted for the kernel when swa_minus_prediction_mean_fold_macro >= -0.002; otherwise run BN recalibration (spec 6.3) and repeat"
    atomic_json(out / "EPOCH_AVERAGING.json", report)
    return report


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

    assets = PKG / "assets"
    aux_names = json.loads((assets / "aux_names.json").read_text())
    if len(aux_names) != 48:
        raise ValueError("AUX60 requires exactly 48 auxiliary heads")
    cache_root = Path(args.cache_root)
    frame, gold, cache_receipt = load_data(cache_root, aux_names)
    slot_map = TokenSlotMap(args.slot_map)
    map_receipt = json.loads(Path(args.slot_map).with_name(Path(args.slot_map).stem + "_RECEIPT.json").read_text())
    if not map_receipt.get("status", "").startswith("PASS"):
        raise ValueError("Token slot map receipt is not PASS")
    config = AugConfig(series_drop=args.series_drop, token_drop=args.token_drop, jitter=args.jitter, params=not args.no_params)
    val = frame.loc[frame.fold == args.fold].copy().reset_index(drop=True)
    fit = frame.copy() if args.full else frame.loc[frame.fold != args.fold].copy().reset_index(drop=True)
    if not args.full and set(fit.report_group) & set(val.report_group):
        raise ValueError("Fold leakage through report_group")
    if args.full:
        val = val.iloc[:0].copy()
    unmapped_fit = sorted(set(fit.SampleID.astype(str)) & slot_map.unmapped)

    cfg = {"schedule": SCHEDULE, "stop": STOP, "swa_window": SWA_WINDOW, "kept_weights": KEEP_WEIGHTS, "batch": BATCH, "weight_decay": WEIGHT_DECAY}
    model = make(assets).to(device)
    steps = math.ceil(len(fit) / BATCH)
    opt, scheduler = optimizer(model, SCHEDULE, steps, False)
    pos_weight = positive_weights(fit, aux_names).to(device)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    identity = {
        "run": args.run_name,
        "arm": ARM,
        "control": "AUX60-v8.3 (train_v83.py --model coatnet), same fold",
        "model": "coatnet",
        "full": bool(args.full),
        "fold": None if args.full else args.fold,
        "seed": SEED,
        "recipe": cfg,
        "aux_weight": AUX_WEIGHT,
        "windows_per_train_sample": 24,
        "train_perturbations": config.as_dict(),
        "augmentation": {
            "scope": "train only; eval path is the v8.3 path bit for bit",
            "where": "GPU, on 336 px source tokens before H1.features (336->384 bilinear, gain, clamp, normalise)",
            "random_resized_crop_area": list(CROP_AREA), "aspect": "fixed 1",
            "rotation_deg": [-ROTATION_DEG, ROTATION_DEG], "shift_fraction": [-SHIFT_FRACTION, SHIFT_FRACTION],
            "gamma_log_uniform": list(GAMMA), "contrast": list(CONTRAST),
            "bias_field": f"exp(degree-2 polynomial), 5 coefficients U(-{BIAS_MAX}, {BIAS_MAX})",
            "blur": f"p={BLUR_P}, gaussian sigma U{BLUR_SIGMA} source px, {BLUR_TAPS}-tap separable",
            "noise": f"p={NOISE_P}, gaussian std U{NOISE_STD}",
            "per_study_gain": "inherited U(0.9, 1.1) inside H1.features",
            "parameter_sharing": "spatial and intensity parameters shared within a semantic slot (series); noise per token",
            "flips": "none (canonical_L_v8_3)",
        },
        "validation_tokens": "all valid up to 64",
        "validation_each_epoch": True,
        "selection_rule": "fixed before training: uniform weight average over epochs 18-20; no held-out or Gold selection",
        "gold_fit_or_selection": False,
        "orientation_policy": EXPECTED_POLICY,
        "orientation_state": EXPECTED_STATE,
        "builder_sha256": EXPECTED_BUILDER,
        "cache_release_sha256": sha(cache_root / "GLOBAL_RELEASE_RECEIPT.json"),
        "token_slot_map_sha256": sha(args.slot_map),
        "token_slot_map_receipt": {k: map_receipt.get(k) for k in ("status", "samples", "mapped", "unmapped", "unmapped_reasons")},
        "unmapped_training_samples": unmapped_fit,
        "training_sample_ids_sha256": h.semantic(sorted(fit.SampleID.astype(str))),
        "validation_parent_ids_sha256": h.semantic(sorted(val.StudyInstanceUID.astype(str).unique())),
        "gold_ids_sha256": h.semantic(sorted(gold.StudyInstanceUID.astype(str))),
        "code_hashes": code_hashes(),
        "instance_binding": {k: budget[k] for k in ("instance_id", "hourly_usd", "gpu_count", "gpu_name")},
        "model_identity": {"family": "CoAtNet-rmlp-2-384 AUX60 + GPU augmentation", "initialization": "ImageNet-12k-to-1k",
                           "pretrained_sha256": sha(assets / "coatnet_imagenet.safetensors")},
        "steps_per_epoch": steps,
        **IDENTITY_EXTRA,
    }
    atomic_json(out / "IDENTITY.json", identity)
    atomic_json(out / "POS_WEIGHT.json", dict(zip(h.TARGETS + aux_names, pos_weight.cpu().tolist())))
    fit[["SampleID", "StudyInstanceUID", "ParentStudyInstanceUID", "report_group", "fold"]].to_csv(out / "fit_ids.csv", index=False)
    if not args.full:
        val[["SampleID", "StudyInstanceUID", "ParentStudyInstanceUID", "report_group", "fold"]].to_csv(out / "validation_ids.csv", index=False)

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
        history, start, updates = state["history"], int(state["epoch"]), int(state["updates"])
        torch.set_rng_state(state["rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])

    started = time.monotonic()
    for epoch_index in range(start, STOP):
        epoch = epoch_index + 1
        train_loader = make_loader(fit, aux_names, True, epoch_index, BATCH, args.workers, slot_map, config)
        model.train()
        opt.zero_grad(set_to_none=True)
        t0 = last = time.monotonic()
        wait = loss_sum = main_sum = aux_sum = 0.0
        count = encoded = 0
        torch.cuda.reset_peak_memory_stats(device)
        for step, (x, mask, meta, y, label_weight, _index, gain, params) in enumerate(train_loader):
            wait += time.monotonic() - last
            guard()
            encoded += int(mask.sum())
            x, mask, meta, y, label_weight, gain, params = [v.to(device, non_blocking=True) for v in (x, mask, meta, y, label_weight, gain, params)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, mask, meta, gain, params)
                main, aux, losses = loss_parts(logits, y, label_weight, pos_weight, aux_weight=AUX_WEIGHT)
                loss = losses.sum()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            n = len(x)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(n)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0, error_if_nonfinite=True)
            opt.step()
            scheduler.step()
            opt.zero_grad(set_to_none=True)
            updates += 1
            count += n
            loss_sum += float(loss.detach())
            main_sum += float(main.sum().detach())
            aux_sum += float(aux.sum().detach())
            if step % 25 == 0:
                atomic_json(out / "LIVE.json", {"run": args.run_name, "phase": "train", "epoch": epoch, "epochs": STOP, "batch": step + 1,
                                                "batches": len(train_loader), "samples": count, "train_loss": loss_sum / count,
                                                "main_loss": main_sum / count, "aux_loss": aux_sum / count,
                                                "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30, "utc": time.time()})
            last = time.monotonic()
        torch.cuda.synchronize(device)
        train_seconds = time.monotonic() - t0
        record = {"epoch": epoch, "train_loss": loss_sum / count, "main_loss": main_sum / count, "aux_loss": aux_sum / count,
                  "train_seconds": train_seconds, "data_wait_seconds": wait, "data_wait_fraction": wait / train_seconds,
                  "encoded_images": encoded, "encoded_images_per_sample": encoded / count,
                  "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30, "updates": updates,
                  "lr": [group["lr"] for group in opt.param_groups]}
        if not args.full:
            atomic_json(out / "LIVE.json", {"run": args.run_name, "phase": "fold_validation", "epoch": epoch, "utc": time.time()})
            fold_logits, fold_seconds = predict(model, val, aux_names, device, args.workers)
            record["pseudo"] = evaluate_fold(val, torch.from_numpy(fold_logits[:, :12]).sigmoid().numpy())
            record["fold_validation_seconds"] = fold_seconds
            save_predictions(out / f"fold_predictions_e{epoch:02d}.csv", val, aux_names, fold_logits)
        atomic_json(out / "LIVE.json", {"run": args.run_name, "phase": "gold_validation", "epoch": epoch, "utc": time.time()})
        gold_logits, gold_seconds = predict(model, gold, aux_names, device, args.workers)
        record["gold"] = gold_auc(gold, torch.from_numpy(gold_logits[:, :12]).sigmoid().numpy())
        record["gold_validation_seconds"] = gold_seconds
        save_predictions(out / f"gold_predictions_e{epoch:02d}.csv", gold, aux_names, gold_logits)
        history.append(record)
        cpu_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch in KEEP_WEIGHTS:
            h.atomic_torch(out / f"weights-e{epoch:02d}.pt", {"model": cpu_state, "identity": identity, "epoch": epoch})
        h.atomic_torch(latest, {"model": cpu_state, "identity": identity, "epoch": epoch, "updates": updates,
                                "optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(), "history": history,
                                "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                                "python_rng": random.getstate(), "numpy_rng": np.random.get_state()})
        atomic_json(out / "history.json", history)
        print(json.dumps(record, allow_nan=False), flush=True)

    states = [h.load_checkpoint(out / f"weights-e{epoch:02d}.pt")["model"] for epoch in SWA_WINDOW]
    average = average_states(states)
    model.load_state_dict(average)
    h.atomic_torch(out / "averaged.pt", {"model": average, "identity": identity, "selected_epochs": SWA_WINDOW})
    result = {"status": "PASS", "run": args.run_name, "arm": ARM, "completed_epochs": STOP, "selected_epochs": SWA_WINDOW,
              "selection_rule": identity["selection_rule"], "averaged_sha256": sha(out / "averaged.pt"),
              "elapsed_seconds": time.monotonic() - started}
    if not args.full:
        logits, seconds = predict(model, val, aux_names, device, args.workers)
        result["pseudo"] = evaluate_fold(val, torch.from_numpy(logits[:, :12]).sigmoid().numpy())
        result["fold_validation_seconds"] = seconds
        save_predictions(out / "averaged_fold_predictions.csv", val, aux_names, logits)
    logits, seconds = predict(model, gold, aux_names, device, args.workers)
    result["gold"] = gold_auc(gold, torch.from_numpy(logits[:, :12]).sigmoid().numpy())
    result["gold_validation_seconds"] = seconds
    save_predictions(out / "averaged_gold_predictions.csv", gold, aux_names, logits)
    h.atomic_torch(out / "main12.pt", {"model": {k: v for k, v in average.items() if not k.startswith(("aux_att.", "auxW", "auxb"))},
                                       "identity": identity, "selected_epochs": SWA_WINDOW})
    result["epoch_averaging"] = epoch_averaging_diagnostics(out, model, val, gold, aux_names, device, args.workers, identity, args.full)
    model.load_state_dict(average)  # leave the declared model in memory/on disk, not a diagnostic variant
    atomic_json(out / "RESULT.json", result)
    atomic_json(out / "LIVE.json", {"run": args.run_name, "phase": "complete", "utc": time.time(), **result})
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=3)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--slot-map", required=True, help="TOKEN_SLOT_MAP.csv built by token_slot_map.py (receipt beside it)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--series-drop", type=float, default=0.10)
    parser.add_argument("--token-drop", type=float, default=0.15)
    parser.add_argument("--jitter", type=int, default=1)
    parser.add_argument("--no-params", action="store_true", help="ablation only: identity spatial/intensity parameters")
    train(parser.parse_args())
