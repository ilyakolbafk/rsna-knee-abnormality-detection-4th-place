"""Laterality from relative anatomical landmarks, independent of coordinate origin.

This module does not detect the fibula. Landmark providers must supply reviewed
or independently validated LPS coordinates for the fibular head and tibial
centre from the same image frame. Absolute X-sign is not anatomical evidence.
Existing Native64 inputs remain unchanged unless a caller opts into v2.
"""
from __future__ import annotations
import numpy as np

def side_from_fibula_landmarks(landmarks, *, min_lateral_separation_mm=8., min_frames=2):
    """Require concordant fibula-to-tibia X displacement on distinct source frames."""
    votes=[];seen=set();frame_of_reference=None
    for item in landmarks:
        if item.get('landmark_qc') is not True:
            return {'side':'','status':'unreviewed_landmark'}
        uid=item.get('sop_instance_uid');ref=item.get('frame_of_reference_uid')
        if not uid or not ref or not item.get('source_sha256'):
            return {'side':'','status':'missing_source_identity'}
        if uid in seen:continue
        seen.add(uid)
        if frame_of_reference is not None and ref!=frame_of_reference:
            return {'side':'','status':'mixed_frame_of_reference'}
        frame_of_reference=ref
        f=np.asarray(item.get('fibula_lps'),dtype=float);t=np.asarray(item.get('tibia_lps'),dtype=float)
        if f.shape!=(3,) or t.shape!=(3,) or not np.isfinite(f).all() or not np.isfinite(t).all():
            return {'side':'','status':'invalid_landmark_geometry'}
        dx=float(f[0]-t[0])
        if abs(dx)<min_lateral_separation_mm:
            return {'side':'','status':'insufficient_lateral_separation'}
        votes.append(dx)
    if len(votes)<min_frames:return {'side':'','status':'insufficient_distinct_frames'}
    if not (all(x>0 for x in votes) or all(x<0 for x in votes)):
        return {'side':'','status':'inconsistent_anatomical_votes'}
    return {'side':'L' if votes[0]>0 else 'R','status':'anatomical_geometry_supported',
            'source_frames':sorted(seen),'fibula_minus_tibia_x_mm':votes,
            'frame_of_reference_uid':frame_of_reference}

def resolve_side_v2(base_record, landmarks=()):
    """Validated relative anatomy takes precedence; retain explicit weak fallback."""
    result=dict(base_record);anatomy=side_from_fibula_landmarks(landmarks)
    result.update(side_policy_version=2,anatomical_evidence=anatomy)
    if anatomy['side']:
        result.update(side=anatomy['side'],source='anatomical_geometry',
                      overrides_baseline=bool(anatomy['side']!=base_record.get('side')))
    else:
        result['overrides_baseline']=False
    result['anatomical_side_supported']=bool(anatomy['side'])
    return result


def resolve_side_v3(base_record, metadata, reviews=(), *, study_uid):
    """Apply source-bound anatomical reviews before tags, without inventing evidence.

    Reviews are observations supplied by a person or an explicitly named visual
    reviewer, not predictions from an automatic landmark detector. Each series
    needs two distinct, geometrically valid source frames. Conflicting series
    abstain at study level; neither majority voting nor absolute X resolves them.
    Untouched baseline assignments retain their original evidence strength.
    """
    from .canonical_inputs import transform_spec

    result = dict(base_record, side_policy_version=3,
                  original_side=base_record.get('side', ''),
                  original_source=base_record.get('source', ''),
                  canonical_target='L', anatomical_side_supported=False,
                  overrides_baseline=False)
    if not reviews:
        result['review_status'] = 'no_anatomical_review_baseline_retained'
        return result
    planes = {p['SOPInstanceUID']: p for p in metadata['planes']}
    supported = []

    def abstain(reason):
        return dict(result, side='', source='anatomical_review_abstain',
                    review_status=reason)

    for review in reviews:
        if (review.get('study_uid') != study_uid or
                review.get('side') not in ('L', 'R') or
                review.get('orientation_verified') is not True or
                not review.get('reviewer') or
                review.get('method') != 'fibula_relative_to_tibia_in_patient_axes'):
            return abstain('invalid_review_identity_or_method')
        sid = review.get('series_uid')
        if sid not in metadata['series']:
            return abstain('review_series_missing')
        witnesses = review.get('frames', [])
        seen = set()
        refs = set()
        for witness in witnesses:
            sop = witness.get('sop_instance_uid')
            p = planes.get(sop)
            if (p is None or p['SeriesInstanceUID'] != sid or
                    not witness.get('source_sha256') or
                    p.get('source_sha256') != witness['source_sha256']):
                return abstain('source_identity_mismatch')
            g = p.get('geometry_tags', {})
            ref = g.get('FrameOfReferenceUID')
            if not ref or transform_spec(g.get('ImageOrientationPatient'),
                                        review.get('plane'), 'L') is None:
                return abstain('review_geometry_not_supported')
            seen.add(sop)
            refs.add(ref)
        if len(seen) < 2 or len(refs) != 1:
            return abstain('insufficient_distinct_concordant_frames')
        supported.append(dict(series_uid=sid, side=review['side'],
                              reviewer=review['reviewer'], frames=sorted(seen)))
    sides = {r['side'] for r in supported}
    if len(sides) != 1:
        return abstain('cross_series_anatomy_disagreement')
    side = next(iter(sides))
    result.update(side=side, source='source_bound_anatomical_review',
                  anatomical_side_supported=True,
                  review_status='supported_by_reviewed_source_frames',
                  anatomical_reviews=supported,
                  overrides_baseline=side != base_record.get('side'))
    return result
