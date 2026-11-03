"""Report-only auxiliary finding pilot; no canonical target or fold mutation."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


# id, stable name, Russian display name, report-semantic rubric (not image truth).
FINDINGS = [
    ("F01", "medial_meniscus_anterior_tear", "Разрыв переднего рога медиального мениска", "Tear specifically in medial meniscus anterior horn; generic meniscal tear does not localize it."),
    ("F02", "medial_meniscus_body_tear", "Разрыв тела медиального мениска", "Tear specifically in medial meniscus body."),
    ("F03", "medial_meniscus_posterior_tear", "Разрыв заднего рога медиального мениска", "Tear specifically in medial meniscus posterior horn."),
    ("F04", "lateral_meniscus_anterior_tear", "Разрыв переднего рога латерального мениска", "Tear specifically in lateral meniscus anterior horn."),
    ("F05", "lateral_meniscus_body_tear", "Разрыв тела латерального мениска", "Tear specifically in lateral meniscus body."),
    ("F06", "lateral_meniscus_posterior_tear", "Разрыв заднего рога латерального мениска", "Tear specifically in lateral meniscus posterior horn."),
    ("F07", "meniscus_root_tear", "Корневой разрыв мениска", "Explicit tear/avulsion of a meniscal root; posterior horn alone does not establish root involvement."),
    ("F08", "meniscus_extrusion", "Экструзия мениска", "Explicit meniscal extrusion or displacement beyond the joint margin; displaced internal tear fragment alone is insufficient."),
    ("F09", "discoid_lateral_meniscus", "Дискоидный латеральный мениск", "Discoid lateral meniscus."),
    ("F10", "meniscus_intrasubstance_degeneration", "Дегенерация мениска без разрыва в той же зоне", "Grade 1/2 or non-surface-reaching intrameniscal degeneration in a site without a stated tear. A tear at another site does not negate this finding. Generic degenerative tear alone is insufficient."),
    ("F11", "parameniscal_cyst", "Параменискальная киста", "Parameniscal/meniscal cyst."),
    ("F12", "acl_complete_tear", "Полный разрыв ACL", "Complete ACL rupture, discontinuity, or unequivocal absent torn ACL. Unqualified tear without completeness: equivocal P, p=0.5, low confidence."),
    ("F13", "acl_partial_tear_sprain", "Частичный разрыв / sprain ACL", "Partial ACL tear or sprain. Pure mucoid degeneration or isolated complete rupture is not a positive."),
    ("F14", "acl_mucoid_degeneration", "Мукоидная дегенерация ACL", "Explicit ACL mucoid degeneration."),
    ("F15", "acl_graft", "Пластика / трансплантат ACL", "ACL reconstruction/graft, regardless of graft integrity; surgical history stated in report counts."),
    ("F16", "pcl_tear_sprain", "Разрыв / sprain PCL", "PCL tear or sprain; mucoid degeneration alone is not enough."),
    ("F17", "lcl_plc_injury", "Повреждение LCL / заднелатерального угла", "LCL or posterolateral corner ligament complex injury, including popliteus tendon injury; isolated nonspecific muscle edema is insufficient."),
    ("F18", "mcl_grade2plus", "Повреждение MCL grade ≥ 2", "MCL grade 2/3, partial/complete tear or structural fiber disruption. Explicit grade 1 only is N (below threshold); generic sprain without grade/disruption is equivocal P p=0.5."),
    ("F19", "cartilage_medial_femoral_grade2plus", "Хрящ медиального мыщелка бедра ≥ 2", "Medial femoral condyle cartilage grade >=2 or explicit thinning, fissure, partial/full thickness defect. Signal heterogeneity/softening alone or OA without cartilage detail is insufficient."),
    ("F20", "cartilage_medial_tibial_grade2plus", "Хрящ медиального плато ≥ 2", "Medial tibial plateau cartilage grade >=2 or explicit thinning/fissure/defect. Unlocalized medial compartment cartilage loss alone does not establish this exact surface."),
    ("F21", "cartilage_lateral_femoral_grade2plus", "Хрящ латерального мыщелка бедра ≥ 2", "Lateral femoral condyle cartilage grade >=2 or explicit thinning/fissure/defect."),
    ("F22", "cartilage_lateral_tibial_grade2plus", "Хрящ латерального плато ≥ 2", "Lateral tibial plateau cartilage grade >=2 or explicit thinning/fissure/defect. Unlocalized lateral compartment cartilage loss alone does not establish this exact surface."),
    ("F23", "cartilage_patella_grade2plus", "Хрящ надколенника ≥ 2", "Patellar cartilage grade >=2 or explicit thinning/fissure/defect. Unspecified chondromalacia is equivocal P p=0.5. Generic PF OA does not localize cartilage damage."),
    ("F24", "cartilage_trochlea_grade2plus", "Хрящ блока бедра ≥ 2", "Trochlear cartilage grade >=2 or explicit thinning/fissure/defect. Generic PF OA does not localize cartilage damage."),
    ("F25", "bme_medial_femoral", "Костномозговой отёк медиального мыщелка бедра", "Bone marrow edema/contusion in medial femoral condyle; includes subchondral edema. Cysts alone are insufficient."),
    ("F26", "bme_lateral_femoral", "Костномозговой отёк латерального мыщелка бедра", "Bone marrow edema/contusion in lateral femoral condyle."),
    ("F27", "bme_medial_tibial", "Костномозговой отёк медиального плато", "Bone marrow edema/contusion in medial tibial plateau."),
    ("F28", "bme_lateral_tibial", "Костномозговой отёк латерального плато", "Bone marrow edema/contusion in lateral tibial plateau."),
    ("F29", "bme_patella", "Костномозговой отёк надколенника", "Bone marrow edema/contusion in patella."),
    ("F30", "bme_other", "Костномозговой отёк другой / неуточнённой зоны", "Bone marrow edema at another or unspecified bone site; do not duplicate explicitly localized F25-F29 here."),
    ("F31", "subchondral_cysts", "Субхондральные кисты", "Subchondral cysts/geodes; unspecified soft-tissue cyst does not count."),
    ("F32", "osteophytes", "Остеофиты", "Explicit marginal osteophytes/bone spurs; generic OA alone is insufficient."),
    ("F33", "subchondral_insufficiency_osteonecrosis", "Субхондральный insufficiency-перелом / остеонекроз", "Subchondral insufficiency fracture, osteonecrosis/SONK. Ordinary traumatic fracture or nonspecific marrow edema alone is insufficient."),
    ("F34", "effusion_size", "Размер выпота (ординально)", "Joint effusion size. v=0 explicitly none/normal physiological amount; 1 small/mild/trace; 2 moderate; 3 large/marked; -1 present or equivocal but size unspecified. P(present) p is separate from ordinal auxiliary v/3. Suprapatellar recess fluid counts as effusion, not automatically bursitis."),
    ("F35", "suprapatellar_bursitis", "Супрапателлярный бурсит", "Explicit suprapatellar bursitis; fluid in the communicating suprapatellar joint recess alone is not sufficient."),
    ("F36", "pre_infrapatellar_bursitis", "Пре- / инфрапателлярный бурсит", "Explicit prepatellar or superficial/deep infrapatellar bursitis; general subcutaneous edema is insufficient."),
    ("F37", "pes_anserine_bursitis", "Бурсит гусиной лапки", "Pes anserine bursitis; semimembranosus-MCL bursitis only if explicitly equated by the report."),
    ("F38", "baker_cyst_rupture", "Разрыв кисты Бейкера", "Explicit ruptured/leaking/dissecting Baker cyst. An unruptured Baker cyst or stated cyst size does not establish rupture. Size is left in evidence, not discretized into an invented threshold."),
    ("F39", "synovial_plica", "Синовиальная plica", "Explicit synovial plica, thickened or not."),
    ("F40", "loose_bodies", "Свободные тела", "Intra-articular loose bodies, osteochondral fragments or synovial chondromatosis nodules; meniscal fragments alone do not count."),
    ("F41", "hoffa_fat_pad_edema", "Отёк жирового тела Гоффа", "Explicit Hoffa/infrapatellar fat-pad edema or inflammatory impingement; generic anterior soft-tissue edema is insufficient."),
    ("F42", "patellar_tendon_pathology", "Тендиноз / разрыв связки надколенника", "Patellar tendon tendinopathy/tendinosis/tear. Isolated normal insertion enthesophyte without tendon abnormality is insufficient."),
    ("F43", "quadriceps_tendinosis", "Тендиноз квадрицепса", "Quadriceps tendon tendinopathy/tendinosis (includes tendinitis). Isolated rupture without tendinosis is not this target."),
    ("F44", "patellar_lateralization_tilt", "Латерализация / наклон надколенника", "Patellar lateral tilt, lateral subluxation or lateralization; patella alta alone is insufficient."),
    ("F45", "trochlear_dysplasia", "Дисплазия блока бедра", "Trochlear dysplasia, hypoplastic/shallow/flat trochlea explicitly described; isolated patellar morphology is insufficient."),
    ("F46", "non_baker_ganglion_cyst", "Ганглий / подколенная киста (не Бейкера)", "Non-Baker ganglion or other popliteal soft-tissue cyst; do not duplicate a parameniscal cyst, subchondral geode or named bursitis."),
    ("F47", "fracture", "Перелом (локализация в evidence)", "Explicit traumatic, stress, insufficiency or avulsion fracture. Keep anatomical location in evidence. Old healed fracture counts only if explicitly described as a persisting structural finding."),
    ("F48", "muscle_edema_strain", "Отёк / растяжение мышцы", "Explicit muscle edema, strain or tear; generic soft-tissue edema does not localize it to muscle."),
    ("F49", "postoperative_changes", "Послеоперационные изменения", "Explicit postoperative state, meniscectomy, reconstruction, surgical hardware or surgical scars. Includes ACL reconstruction."),
    ("F50", "open_physes", "Открытые зоны роста", "Explicit open/unfused growth plates/physes; do not infer from age."),
]
IDS = [f[0] for f in FINDINGS]
BY_ID = {f[0]: f for f in FINDINGS}
STATES = ["P", "N", "U"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def output_schema() -> dict:
    def obj(properties):
        return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
    evidence = obj({"id": {"type": "string"}, "sid": {"type": "string"}, "quote": {"type": "string"}})
    finding = obj({
        "id": {"type": "string", "enum": IDS},
        "s": {"type": "string", "enum": ["P", "N"]},
        "p": {"type": "number"}, "c": {"type": "number"},
        "e": {"type": "array", "items": {"type": "string"}},
        "v": {"type": "integer", "enum": [-1, 0, 1, 2, 3]},
    })
    case = obj({"case_id": {"type": "string"}, "complete": {"type": "boolean"},
                "evidence": {"type": "array", "items": evidence},
                "findings": {"type": "array", "items": finding}})
    return obj({"cases": {"type": "array", "items": case}})


def validate_batch(payload: dict, cases: list[dict]) -> list[dict]:
    from jsonschema import Draft202012Validator
    Draft202012Validator(output_schema()).validate(payload)
    if [r['case_id'] for r in payload['cases']] != [c['case_id'] for c in cases]:
        raise ValueError("Case identity/order mismatch")
    expanded = []
    for record, case in zip(payload['cases'], cases, strict=True):
        if not record['complete']:
            raise ValueError("Incomplete extraction must not become not_mentioned")
        sentences = dict(re.findall(r'\[(S\d+)\] (.*?)(?=\n\[S\d+\]|\Z)', case['sentenced_report'], re.S))
        ev = {}
        for e in record['evidence']:
            if e['id'] in ev or not e['quote'].strip() or e['sid'] not in sentences or e['quote'] not in sentences[e['sid']]:
                raise ValueError(f"Invalid exact evidence {case['case_id']} {e['id']}")
            ev[e['id']] = e
        seen = set()
        for f in record['findings']:
            fid = f['id']
            if fid in seen:
                raise ValueError("Duplicate finding")
            seen.add(fid)
            if not (0 <= f['p'] <= 1 and 0 <= f['c'] <= 1):
                raise ValueError("Out of range probability/confidence")
            if not f['e'] or any(e not in ev for e in f['e']) or len(set(f['e'])) != len(f['e']):
                raise ValueError("Non-silence requires exact evidence")
            if (f['s'] == 'N' and f['p'] > .5) or (f['s'] == 'P' and f['p'] < .5):
                raise ValueError("Inconsistent state/probability")
            if f['s'] == 'P' and f['p'] < .8 and f['c'] > .5:
                raise ValueError("Equivocal positive must have low confidence")
            if (fid != 'F34' and f['v'] != -1) or (fid == 'F34' and ((f['s'] == 'N' and f['v'] != 0) or (f['s'] == 'P' and f['v'] == 0))):
                raise ValueError("Invalid ordinal encoding")
            if fid == 'F34' and f['v'] == -1 and f['c'] > .5:
                raise ValueError("Unknown effusion size must have low confidence")
            expanded.append({"case_id": case['case_id'], "finding_id": fid, "state": f['s'],
                             "p_present": f['p'], "confidence": f['c'], "mask": 1.,
                             "aux_value": f['p'] if fid != 'F34' else (.5 if f['v'] == -1 else f['v'] / 3),
                             "ordinal_level": f['v'], "evidence": [ev[e] for e in f['e']]})
        for fid in IDS:
            if fid not in seen:
                expanded.append({"case_id": case['case_id'], "finding_id": fid, "state": 'U',
                                 "p_present": .5, "confidence": 0., "mask": .5,
                                 "aux_value": .5, "ordinal_level": -1, "evidence": []})
    return expanded


def kappa(a, b, weights=None):
    a, b = np.asarray(a), np.asarray(b)
    if not len(a) or len(set(a) | set(b)) < 2:
        return None
    value = cohen_kappa_score(a, b, labels=STATES, sample_weight=weights)
    return float(value) if np.isfinite(value) else None


def correlation(a, b, weights=None):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3:
        return None
    w = np.ones(len(a)) if weights is None else np.asarray(weights, dtype=float)
    ac, bc = a - np.average(a, weights=w), b - np.average(b, weights=w)
    denominator = np.sqrt(np.average(ac ** 2, weights=w) * np.average(bc ** 2, weights=w))
    return float(np.average(ac * bc, weights=w) / denominator) if denominator > 1e-12 else None


def stratified_sample(frame: pd.DataFrame, per_language=20, seed='aux-pilot-20260909-v1'):
    """Equal allocation, uniform hash-ranked studies per language; no label selection."""
    if not frame.StudyInstanceUID.is_unique:
        raise ValueError("Non-unique study frame")
    if frame[['StudyInstanceUID', 'language_proxy']].isna().any().any():
        raise ValueError("Missing identity/language")
    df = frame.copy()
    df['sample_hash'] = df.StudyInstanceUID.map(lambda u: canonical_hash([seed, u]))
    counts = df.groupby('language_proxy').size()
    if (counts < per_language).any():
        raise ValueError("Stratum too small")
    selected = df.sort_values(['language_proxy', 'sample_hash']).groupby('language_proxy', sort=True).head(per_language).copy()
    selected['population_n_language'] = selected.language_proxy.map(counts)
    selected['sample_n_language'] = per_language
    selected['study_weight'] = selected.population_n_language / per_language
    return selected.reset_index(drop=True)


def combine_reader_values(pair: pd.DataFrame) -> pd.DataFrame:
    """Consensus on the actual training-target scale (ordinal size is not presence)."""
    out=pair.copy()
    out['aux_value']=(out.aux_value_a+out.aux_value_b)/2
    out['confidence']=(out.confidence_a+out.confidence_b)/2*(1-(out.aux_value_a-out.aux_value_b).abs())
    out['mask']=(out.mask_a+out.mask_b)/2
    return out
