"""Synchronous, two-slot optimizer-boundary recovery. No cache or network operations."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

SCHEMA = "knee_optimizer_boundary_v1"
IN_EPOCH = "IN_EPOCH"
BETWEEN_EPOCHS = "BETWEEN_EPOCHS"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def freeze(value):
    """Independent CPU copies; later optimizer/meter updates cannot change a snapshot."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: freeze(v) for k, v in value.items()}
    if isinstance(value, list):
        return [freeze(v) for v in value]
    if isinstance(value, tuple):
        return tuple(freeze(v) for v in value)
    return copy.deepcopy(value)


def rng_state():
    return {
        "torch": torch.get_rng_state().clone(),
        "cuda": freeze(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else [],
        "numpy": freeze(np.random.get_state()),
        "python": random.getstate(),
    }


def restore_rng(state):
    # Call after ALL model/optimizer/scheduler setup and resumed iterator construction.
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        assert torch.cuda.is_available(), "RECOVERY_EXPECTED_CUDA"
        assert len(state["cuda"]) == torch.cuda.device_count(), "RECOVERY_CUDA_COUNT_CHANGED"
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def meter_state(meter):
    return {
        "buffers": {head: {"names": list(names), "values": freeze(values)}
                    for head, (names, values) in meter.buffers.items()},
        "sources": copy.deepcopy(meter.sources),
        "steps": int(meter.steps),
        "source": meter.source,
    }


def restore_meter(meter, state, device):
    assert state["steps"] >= 0
    buffers = {}
    for head, record in state["buffers"].items():
        values = record["values"]
        assert values.dtype == torch.float32 and values.shape == (4, len(record["names"]))
        assert torch.isfinite(values).all(), "NONFINITE_RECOVERY_METER"
        buffers[head] = (list(record["names"]), values.to(device).clone())
    assert all(math.isfinite(float(v)) for v in state["sources"].values())
    meter.buffers = buffers
    meter.sources = copy.deepcopy(state["sources"])
    meter.steps = int(state["steps"])
    meter.source = state["source"]


def cursor(epoch, step, global_step, phase=IN_EPOCH):
    return {"epoch": int(epoch), "step": int(step), "global_step": int(global_step), "phase": phase}


def validate_cursor(value, steps_per_epoch, epochs):
    e, k, g = (value[n] for n in ["epoch", "step", "global_step"])
    assert all(type(v) is int for v in [e, k, g]), "RECOVERY_CURSOR_TYPE"
    assert value["phase"] in [IN_EPOCH, BETWEEN_EPOCHS]
    assert 1 <= e <= epochs + 1 and 0 <= k <= steps_per_epoch
    assert g == (e - 1) * steps_per_epoch + k, "RECOVERY_GLOBAL_STEP_MISMATCH"
    if value["phase"] == BETWEEN_EPOCHS:
        assert k == 0, "RECOVERY_FINALIZED_PHASE_HAS_STEPS"
    else:
        assert e <= epochs and k > 0, "RECOVERY_IN_EPOCH_CURSOR_INVALID"


def snapshot(binding, model, optimizer, scheduler, meter, position, seen, swa, swa_n,
             elapsed_seconds=0.0, sampler=None):
    # No random operation is permitted in capture or serialization.
    result = {
        "schema": SCHEMA, "binding": copy.deepcopy(binding), "cursor": copy.deepcopy(position),
        "model": freeze(model.state_dict()), "optimizer": freeze(optimizer.state_dict()),
        "scheduler": freeze(scheduler.state_dict()), "meter": meter_state(meter),
        "seen": list(map(int, seen)), "swa": freeze(swa), "swa_n": int(swa_n),
        "rng": rng_state(), "elapsed_seconds": float(elapsed_seconds),
        "sampler": copy.deepcopy(sampler),
    }
    return result


def restore_training(state, binding, model, optimizer, scheduler, meter, device):
    assert state["schema"] == SCHEMA and state["binding"] == binding, "RECOVERY_IDENTITY_MISMATCH"
    model.load_state_dict(state["model"], strict=True)
    # Construct BOTH optimizer and scheduler before this call. Loading optimizer last
    # relative to scheduler initialization preserves the checkpoint's parameter-group LRs.
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_meter(meter, state["meter"], device)
    restore_rng(state["rng"])


def make_epoch_iterators(main_loader, fm_loader, position, restored_rng=None):
    """Match original FM-then-main iterator order; discard only extra resume seed draws.

    BETWEEN_EPOCHS RNG is BEFORE the new epoch's loader draws. IN_EPOCH RNG is
    AFTER those draws: reconstruction must not consume them a second time.
    """
    if position["phase"] == IN_EPOCH:
        assert restored_rng is not None
        with torch.random.fork_rng(devices=[]):
            fm_iter, main_iter = iter(fm_loader), iter(main_loader)
        restore_rng(restored_rng)
    else:
        fm_iter, main_iter = iter(fm_loader), iter(main_loader)
    return main_iter, fm_iter


def _fsync_dir(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial." + str(os.getpid()))
    with temporary.open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)
    _fsync_dir(path.parent)


class RecoveryStore:
    """One writer under the trainer's lifetime GPU lease; two independently committed slots."""
    def __init__(self, root, binding):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.binding = copy.deepcopy(binding)
        self.generation = 0
        self.last_errors = []

    def exists(self):
        return any((self.root / ("slot" + str(i) + suffix)).exists()
                   for i in [0, 1] for suffix in [".pt", ".manifest.json"])

    def _verified_records(self):
        records, errors = [], []
        for slot in [0, 1]:
            manifest = self.root / ("slot" + str(slot) + ".manifest.json")
            if not manifest.exists():
                continue
            try:
                entry = json.loads(manifest.read_text())
                assert entry["status"] == "PASS_FULL_SHA_DURABLE_RECOVERY_SLOT"
                assert entry["schema"] == SCHEMA and entry["slot"] == slot
                assert entry["binding"] == self.binding, "RECOVERY_BINDING_MISMATCH"
                assert type(entry["generation"]) is int and entry["generation"] > 0
                path = self.root / ("slot" + str(slot) + ".pt")
                assert path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"]
                records.append(entry)
            except Exception as exc:
                errors.append({"slot": slot, "reason": type(exc).__name__ + ":" + str(exc)[:160]})
        self.last_errors = errors
        return sorted(records, key=lambda x: x["generation"])

    def load(self):
        records = self._verified_records()
        assert records, "NO_VERIFIED_RECOVERY_SLOT"
        entry = records[-1]
        state = torch.load(self.root / ("slot" + str(entry["slot"]) + ".pt"),
                           map_location="cpu", weights_only=False)
        assert state["schema"] == SCHEMA and state["binding"] == self.binding
        assert state["cursor"] == entry["cursor"]
        self.generation = entry["generation"]
        return state, entry

    def save(self, state):
        assert state["schema"] == SCHEMA and state["binding"] == self.binding
        self.generation += 1
        slot = self.generation % 2
        path = self.root / ("slot" + str(slot) + ".pt")
        temporary = path.with_name(path.name + ".partial." + str(os.getpid()))
        with temporary.open("wb") as f:
            torch.save(state, f); f.flush(); os.fsync(f.fileno())
        digest, amount = sha(temporary), temporary.stat().st_size
        os.replace(temporary, path); _fsync_dir(self.root)
        entry = {"status": "PASS_FULL_SHA_DURABLE_RECOVERY_SLOT", "schema": SCHEMA,
                 "slot": slot, "generation": self.generation, "sha256": digest, "bytes": amount,
                 "binding": self.binding, "cursor": state["cursor"], "at_unix": time.time()}
        atomic_json(self.root / ("slot" + str(slot) + ".manifest.json"), entry)
        # The per-slot manifest is the commit record. A crash before this summary
        # replacement can still recover the newest independently verified slot.
        summary = {"status": "PASS_TWO_SLOT_RECOVERY_MANIFEST", "schema": SCHEMA,
                   "binding": self.binding, "latest_generation": self.generation,
                   "slots": self._verified_records(), "at_unix": time.time()}
        atomic_json(self.root / "MANIFEST.json", summary)
        return entry
