"""Evidence-first helpers for auditing report-derived RSNA Knee labels.

The functions in this module deliberately produce conservative audit states.  They do
not replace the frozen report teacher and they never treat an absent mention as a
negative finding.  Normalized evidence clauses are retained so every rule-derived state
can be inspected without copying the complete report into the long analysis panel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

import numpy as np

from .constants import TARGET_COLUMNS
from .reports import dominant_script
from . import report_rules as rr


@dataclass(frozen=True)
class TargetEvidenceAudit:
    """Conservative text evidence attached to one report-target pair."""

    text_state: str
    uncertain_mentions: int
    evidence_clause_normalized: str
    evidence_section_proxy: str
    criteria_state: str
    criteria_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_TOKEN_RE = re.compile(r"[a-z]{2,}")
_SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "impression",
        re.compile(
            r"\b(impression|conclusion|conclusions|impresion|conclusion|sonuc|"
            r"beurteilung|conclusie|zakljucak|заключение|заключение|συμπερασμα)\b"
        ),
    ),
    (
        "findings",
        re.compile(
            r"\b(findings|results|hallazgos|resultados|bulgular|befund|bevindingen|"
            r"nalaz|reperti|резултати|находка|ευρηματα)\b"
        ),
    ),
    (
        "technique",
        re.compile(r"\b(technique|tecnica|technik|techniek|teknik|metoda|техника)\b"),
    ),
    (
        "history",
        re.compile(r"\b(history|clinical|indication|historia|anamnes|клинич|ιστορικο)\b"),
    ),
)


# These are intentionally labelled as proxies.  They are high-precision lexical cues,
# not a claim that a report belongs to a country, institution, or specific reader.
_LANGUAGE_CUES: dict[str, set[str]] = {
    "english_proxy": {
        "findings", "impression", "knee", "tear", "intact", "without",
        "there", "joint", "effusion", "cartilage", "ligament",
    },
    "spanish_proxy": {
        "rodilla", "hallazgos", "impresion", "derrame", "rotura", "menisco",
        "ligamento", "senal", "condropatia", "ausencia", "conservado",
    },
    "turkish_proxy": {
        "diz", "bulgular", "sonuc", "izlenmedi", "izlendi", "menisk", "capraz",
        "eklem", "normaldir", "yirtik", "bag", "saptanmadi",
    },
    "south_slavic_latin_proxy": {
        "koljena", "koleno", "meniskusa", "ligamenta", "nalaz", "zakljucak",
        "ocuvan", "medijalnog", "lateralnog", "intenziteta", "hrskavice",
    },
    "german_proxy": {
        "befund", "beurteilung", "knie", "meniskus", "kreuzband", "innenband",
        "erguss", "unauffallig", "knorpel", "ruptur",
    },
    "dutch_proxy": {
        "bevindingen", "conclusie", "knie", "meniscus", "kruisband", "vocht",
        "kraakbeen", "afwijking", "gewricht", "scheur",
    },
    "french_proxy": {
        "genou", "menisque", "epanchement", "chondropathie", "croise",
        "dechirure", "rotule", "femoropatellaire", "osseux", "conserve",
    },
    "portuguese_proxy": {
        "joelho", "menisco", "ligamento", "derrame", "conclusao", "condropatia",
        "cartilagem", "rotura", "cruzado", "patelar",
    },
    "italian_proxy": {
        "ginocchio", "menisco", "legamento", "versamento", "reperti",
        "conclusioni", "cartilagine", "rotula", "femororotulea", "crociato",
    },
}


def language_proxy(text: object) -> tuple[str, float]:
    """Return a conservative language/script proxy and a margin-like confidence."""

    script = dominant_script(text)
    if script != "latin":
        return f"{script}_script_proxy", 1.0
    normalized = rr.normalize(str(text))
    tokens = set(_TOKEN_RE.findall(normalized))
    scores = {
        language: len(tokens.intersection(cues))
        for language, cues in _LANGUAGE_CUES.items()
    }
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best_name, best_score = ordered[0]
    second_score = ordered[1][1]
    if best_score < 2 or best_score == second_score:
        return "latin_unknown_proxy", 0.0
    confidence = (best_score - second_score) / max(best_score, 1)
    return best_name, float(confidence)


def _target_clause_match(target: str, clause: str) -> bool:
    if target in rr.PAIRED:
        hit = rr.ANAT_MATCH[target].search(clause)
        if hit is None and "Meniscus" in target and rr.PLURAL_MENISCI.search(clause) \
                and not rr.ANY_SIDE.search(clause):
            hit = rr.PLURAL_MENISCI.search(clause)
        return hit is not None

    if target in rr.OA_TARGETS:
        if rr.OA_EVIDENCE.search(clause) is None:
            return False
        hits: list[str] = []
        if rr._near(clause, rr.TF_SITE, rr.SIDE_MEDIAL, 45):
            hits.append("Medial OA")
        if rr._near(clause, rr.TF_SITE, rr.SIDE_LATERAL, 45):
            hits.append("Lateral OA")
        if rr.PF_SITE.search(clause) is not None:
            hits.append("PF OA")
        # The frozen teacher inherits unlocalised whole-joint OA evidence to all
        # compartments.  Keep the clause visible so the inheritance can be audited.
        return target in hits or not hits

    hit = rr.DIRECT_MATCH[target].search(clause)
    if hit is None:
        return False
    decoy = rr.DECOY.get(target)
    return decoy is None or decoy.search(clause) is None


def _section_proxy(clauses: list[str]) -> str:
    found: list[str] = []
    for clause in clauses:
        for name, pattern in _SECTION_PATTERNS:
            if pattern.search(clause) and name not in found:
                found.append(name)
    return "|".join(found) if found else "unknown"


_CHRONIC_FRACTURE = re.compile(
    r"\b(old|healed|healing|chronic|remote|sequela|callus|ancienne|ancien|"
    r"alt|abgeheilt|eski|kronik|зараснал|стар[аио]|παλαι)\b"
)
_ACUTE_FRACTURE = re.compile(
    r"\b(acute|new|fresh|agud[oa]|akut|aigu|frisch|остр[а-я]*|прясн[а-я]*|οξει)\b"
)
_HIGH_GRADE = re.compile(
    r"\b(high grade|high-grade|complete|full thickness|full-thickness|greater than 50|"
    r"more than 50|>\s*50|discontinuity|disrupted|macerated|bucket handle)\b"
)


def _criteria_state(target: str, state: str, evidence_clauses: list[str]) -> tuple[str, str]:
    if state == "not_mentioned":
        return "not_assessable", "no_rule_supported_target_clause"
    if state == "explicit_negative":
        return "not_applicable_negative", "explicit_negative_text"
    if state in {"uncertain", "contradictory", "implicit_or_proxy"}:
        return "unclear", state

    text = " ".join(evidence_clauses)
    low = rr.SEV_LOW.search(text) is not None
    high = rr.SEV_HIGH.search(text) is not None
    tear = rr.TEAR.search(text) is not None
    degeneration = rr.DEGEN.search(text) is not None
    grades = [rr._grade_of(clause) for clause in evidence_clauses]
    grades = [grade for grade in grades if grade is not None]
    grade = max(grades) if grades else None

    if target in {"Medial Meniscus", "Lateral Meniscus"}:
        if tear or (grade is not None and grade >= 3):
            return "meets_threshold", "surface_or_morphologic_tear_proxy"
        if degeneration or (grade is not None and grade <= 2):
            return "below_threshold", "intrasubstance_or_low_grade_meniscal_change"
        return "unclear", "meniscal_abnormality_without_surface_or_grade_evidence"

    if target in {"ACL", "MCL"}:
        if tear and (_HIGH_GRADE.search(text) is not None or (grade is not None and grade >= 2)):
            return "meets_threshold", "high_grade_or_complete_tear_proxy"
        if low or degeneration or ("sprain" in text and not high):
            return "below_threshold", "mild_low_grade_or_chronic_change_proxy"
        return "unclear", "ligament_injury_without_high_grade_evidence"

    if target in rr.OA_TARGETS:
        if (grade is not None and grade >= 3) or high:
            return "meets_threshold", "high_grade_or_moderate_large_cartilage_loss_proxy"
        if low or (grade is not None and grade <= 2):
            return "below_threshold", "mild_or_low_grade_cartilage_change"
        return "unclear", "oa_wording_without_size_or_high_grade_evidence"

    if target in {"Effusion", "Baker's"}:
        if high:
            return "meets_threshold", "moderate_or_large_fluid_proxy"
        if low:
            return "below_threshold", "trace_small_or_mild_fluid_proxy"
        return "unclear", "fluid_finding_without_size_evidence"

    if target == "Synovitis":
        return "meets_threshold", "direct_synovitis_assertion"

    if target == "Contusion":
        if rr.TRAUMA.search(text) is not None:
            return "meets_threshold", "traumatic_contusion_or_bone_bruise_proxy"
        if rr.DEGENERATIVE_MARROW.search(text) is not None:
            return "below_threshold", "degenerative_or_reactive_marrow_change_proxy"
        return "unclear", "marrow_edema_without_traumatic_context"

    if target == "Fracture":
        if _ACUTE_FRACTURE.search(text) is not None:
            return "meets_threshold", "acute_fracture_cue"
        if _CHRONIC_FRACTURE.search(text) is not None:
            return "below_threshold", "old_healed_or_chronic_fracture_cue"
        return "unclear", "fracture_without_acuity_evidence"

    raise ValueError(f"Unknown target: {target}")


def audit_target_evidence(
    report: object,
    target: str,
    *,
    positive_mentions: int,
    negative_mentions: int,
    rule_score: float,
) -> TargetEvidenceAudit:
    """Extract a conservative state, normalized clauses, and rubric-alignment proxy."""

    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unknown target: {target}")
    clauses = [
        clause for clause in rr.clauses(str(report))
        if _target_clause_match(target, clause)
    ]
    # Preserve order while removing duplicated clauses introduced by long-clause splitting.
    evidence_clauses = list(dict.fromkeys(clauses))
    uncertain_mentions = sum(rr.UNCERTAIN.search(clause) is not None for clause in evidence_clauses)

    positive_mentions = int(positive_mentions)
    negative_mentions = int(negative_mentions)
    if positive_mentions and negative_mentions:
        state = "contradictory"
    elif uncertain_mentions and not positive_mentions:
        state = "uncertain"
    elif positive_mentions:
        state = "explicit_positive"
    elif negative_mentions:
        state = "explicit_negative"
    elif not np.isclose(float(rule_score), 0.28, atol=1e-12, rtol=0.0):
        state = "implicit_or_proxy"
    else:
        state = "not_mentioned"

    criteria_state, criteria_reason = _criteria_state(target, state, evidence_clauses)
    return TargetEvidenceAudit(
        text_state=state,
        uncertain_mentions=int(uncertain_mentions),
        evidence_clause_normalized=" || ".join(evidence_clauses[:4]),
        evidence_section_proxy=_section_proxy(evidence_clauses),
        criteria_state=criteria_state,
        criteria_reason=criteria_reason,
    )


def linear_downweight(
    priority: np.ndarray | float,
    *,
    lower: float = 0.75,
    upper: float = 0.95,
    floor: float = 0.10,
) -> np.ndarray:
    """Frozen continuous multiplier used by the abstention/downweight policy."""

    values = np.asarray(priority, dtype=float)
    progress = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    return np.clip(1.0 - progress * (1.0 - floor), floor, 1.0)


def conservative_soft_probability(
    text_probability: np.ndarray | float,
    image_probability: np.ndarray | float,
    priority: np.ndarray | float,
    image_consensus: np.ndarray | float,
    *,
    activation_priority: float = 0.75,
    full_activation_priority: float = 0.95,
    maximum_image_weight: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a low-weight OOF sensitivity blend and its explicit image weight."""

    text = np.asarray(text_probability, dtype=float)
    image = np.asarray(image_probability, dtype=float)
    priority_values = np.asarray(priority, dtype=float)
    consensus = np.asarray(image_consensus, dtype=float)
    activation = np.clip(
        (priority_values - activation_priority)
        / (full_activation_priority - activation_priority),
        0.0,
        1.0,
    )
    image_weight = maximum_image_weight * activation * np.clip(consensus, 0.0, 1.0)
    probability = (1.0 - image_weight) * text + image_weight * image
    return probability, image_weight
