"""Ordinary Core V20 teacher-V fold3; only train sampling changes 12 to24."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import matched_core as matched
import train_core_v2_a as core
import v83_aug_data as data
from token_slot_map import TokenSlotMap

ROOT = Path(__file__).resolve().parent
TASK = ROOT.parent


def sha(path):
    return matched.sha(Path(path))


def verify_static():
    qa = json.loads((TASK / "ACTUAL_ENCODER_CPU_QA.json").read_text())
    if qa["status"] != "PASS_ACTUAL_ENCODER_CPU_ORDINARY_EQUIVALENCE":
        raise ValueError("Ordinary actual-encoder CPU equivalence missing")
    for name, digest in qa["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("CPU-qualified ordinary source drift: " + name)
    if data.WINDOWS != 24:
        raise ValueError("Wrong ordinary K24 sampler")
    slots = TokenSlotMap(ROOT / "assets/TOKEN_SLOT_MAP.csv")
    if len(set(slots.counts) | slots.unmapped) != 4352:
        raise ValueError("Wrong training slot map")


def configure_core(cache):
    core.ARM = "CORE_V20_ORDINARY_K24_F3"
    core.V83AugBags = matched.ConfidenceBags
    core.loader = data.loader
    # Leave the ordinary H1AuxAug make() and six-field metadata unchanged.
    core.load_data = lambda cache_root, names: matched.frame_for(cache_root, names, "V")
    core.evaluate_fold = matched.original_reference_eval
    core.positive_weights = lambda fit, names: matched.fixed_weights(3, names)
    original = core.code_hashes
    core.code_hashes = lambda: {**original(), "ordinary_v20_fold3.py": sha(__file__),
                              "v83_aug_data.py": sha(ROOT / "v83_aug_data.py")}
    core.IDENTITY_EXTRA = {
        "recipe_variant": "V20_teacherV_fold3_ordinary_K24", "teacher": "V",
        "gold_in_training": False, "gold_metric_role": "held_out_development_diagnostic",
        "architecture": "ordinary Core V20 H1AuxAug without Transformer",
        "main_targets_sha256": sha(ROOT / "assets/V_MAIN.csv"),
        "fixed_pos_weight_sha256": sha(ROOT / "assets/POS_WEIGHT_f3.json"),
        "cpu_equivalence_sha256": sha(TASK / "ACTUAL_ENCODER_CPU_QA.json"),
        "cache_reference_sha256": sha(cache / "GLOBAL_RELEASE_RECEIPT.json"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--run-name", default="core_v20_ordinary_k24_f3")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Fresh single attempt only; output exists")
    verify_static()
    gate = json.loads((TASK / "START_GATE.json").read_text())
    if gate["status"] != "PASS_START_AFTER_K24_TERMINAL" or gate["instance_id"] != 55518941:
        raise ValueError("Authorized post-K24 start gate missing")
    if gate["package_sha256"] != sha(__file__) or gate["cache_release_sha256"] != sha(args.cache_root / "GLOBAL_RELEASE_RECEIPT.json"):
        raise ValueError("Start gate source/cache drift")
    if gate["cache_npz_verification"] != "PASS_4410_ACTUAL_SHA256":
        raise ValueError("Actual cache bytes not checked")
    guard = json.loads((ROOT / "BUDGET_GUARD.json").read_text())
    if guard["instance_id"] != 55518941 or guard["gpu_count"] != 1 or guard["hard_stop_unix"] <= time.time() + 600:
        raise ValueError("Fresh shared-server guard missing")
    matched.verify_freeze(args.cache_root)
    configure_core(args.cache_root)
    core.train(SimpleNamespace(cache_root=str(args.cache_root), slot_map=str(ROOT / "assets/TOKEN_SLOT_MAP.csv"),
        fold=3, full=False, series_drop=.10, token_drop=.15, jitter=1, no_params=False,
        output_dir=str(args.output_dir), run_name=args.run_name, workers=args.workers))


if __name__ == "__main__":
    main()
