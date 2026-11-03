"""Pre-dispatch semantic gate. Reads proofs/metadata only; never promotes pixels by name."""
from __future__ import annotations

from collections import Counter
from typing import Mapping


class CacheNotAccepted(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise CacheNotAccepted(code)


def require_bundle(bundle: Mapping, target_instance=56967269):
    """Normalized acceptance dossier, to be produced by independent cache QA.

    These fields are evidence requirements, not a reinterpretation of a producer's
    candidate receipt. In particular a byte-exact candidate is not training ready.
    """
    require(bundle.get("target_instance") == target_instance, "WRONG_TARGET")
    datasets = bundle.get("datasets", {})
    require(set(datasets) == {"RSNA", "OAI", "fastMRI"}, "THREE_CORPORA_REQUIRED")
    for name, data in datasets.items():
        require(data.get("training_ready") is True, name + ":NOT_TRAINING_READY")
        require(data.get("semantic_gate") == "PASS", name + ":SEMANTIC_GATE")
        require(data.get("sealed") is True, name + ":INCOMPLETE_MANIFEST")
        expected, verified = data.get("expected_ids"), data.get("verified_ids")
        require(isinstance(expected, list) and bool(expected), name + ":NO_ROSTER")
        require(len(expected) == len(set(expected)), name + ":DUPLICATE_ROSTER")
        require(isinstance(verified, list) and len(verified) == len(set(verified)), name + ":DUPLICATE_PROOF")
        require(set(expected) == set(verified), name + ":INCOMPLETE_COVERAGE")
        require(data.get("destination_full_sha_verified") is True, name + ":NO_DESTINATION_PROOF")
        require(data.get("target_instance") == target_instance, name + ":WRONG_TARGET")
        for field in ("producer_sha256", "manifest_sha256", "loader_qa_sha256", "semantic_qa_sha256", "destination_fullsha_proof_sha256"):
            digest = data.get(field, "")
            require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), name + ":MISSING_" + field)
        require(data.get("legacy_110_fallback") is False, name + ":LEGACY_FALLBACK")
        if name in ("RSNA", "OAI"):
            require(data.get("format_profile") == "full_v4", name + ":NOT_FULL_V4")
            require(data.get("orientation_state") == "canonical_L_v8_3", name + ":ORIENTATION")
            for field in ("fullfield_and_anatomical_views_verified", "all_source_slice_slab_coverage_verified", "physical_geometry_verified"):
                require(data.get(field) is True, name + ":" + field)
        else:
            require(data.get("format_profile") == "fm384_v1", "fastMRI:FORMAT")
            for field in ("anatomical_view_semantics_verified", "box_transform_verified", "class_mapping_verified", "patient_holdout_verified"):
                require(data.get(field) is True, "fastMRI:" + field)
    return True


def require_v4_audit(audit: Mapping, *, recipe_sha256: str):
    """Actual per-case OAI audit schema; no arrays, imports or source rehashing."""
    require(audit.get("recipe_sha256") == recipe_sha256, "RECIPE_BINDING")
    require(audit.get("orientation_state") == "canonical_L_v8_3", "ORIENTATION")
    detail = audit.get("audit", {})
    blocks, lineage = detail.get("blocks", []), detail.get("lineage", [])
    require(bool(blocks) and len(lineage) == len(blocks), "SERIES_LINEAGE")
    require(all(s.get("orientation_applications") == 1 for s in lineage), "ORIENTATION_NOT_ONCE")
    required = {"Sagittal": {0, 1, 4, 5}, "Axial": {0, 1, 4, 5}, "Coronal": {0, 2, 3}}
    spans = []
    for block in blocks:
        require(block.get("plane") in required, "UNKNOWN_PLANE")
        views = block.get("views", [])
        require(Counter(v[0] for v in views) == Counter(required[block["plane"]]), "MISSING_FULL_ANATOMICAL_VIEWS")
        count = block.get("n_slices", 0)
        slabs = block.get("slabs", [])
        require(count > 0 and len(slabs) == count, "SLAB_PROVENANCE")
        require(not block.get("dropped_tail_source_sops"), "SOURCE_SLICES_DROPPED")
        for slab in slabs:
            sops, weights = slab.get("source_sops", []), slab.get("weights", [])
            require(bool(sops) and len(sops) == len(weights), "SLAB_SOURCE_BINDING")
            require(all(w > 0 for w in weights) and abs(sum(weights) - 1) < 1e-6, "SLAB_WEIGHTS")
        for _, start in views:
            require(isinstance(start, int) and start >= 0, "BAD_ROW_SPAN")
            spans.append((start, start + count))
    cursor = 0
    for start, end in sorted(spans):
        require(start == cursor, "ROW_GAP_OR_OVERLAP")
        cursor = end
    require(cursor == audit.get("rows"), "ROW_COUNT")
    geometry = detail.get("view_geometry", [])
    require(len(geometry) == cursor and {g.get("row") for g in geometry} == set(range(cursor)), "PHYSICAL_GEOMETRY_ROWS")
    return True
