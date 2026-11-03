"""Diagnostic scope of image-side evidence; never canonicalize or rewrite labels.

Opposite votes can reflect two knees or classifier errors. Opposite absolute
patient-X centers are advisory because acquisitions can use different origins.
The existing predictor is unchanged; callers must opt into this diagnostic.
"""
from __future__ import annotations
import math
from .laterality_inference import reconcile_series_sides


def summarize_series_scope(series_results, *, high_score=.9,
                           geometry_centers=None, min_abs_x_mm=20.):
    """Keep L/R evidence per evaluated series instead of taking a majority.

    The caller binds source frames and series to exactly one study upstream.
    Geometry may include additional, unevaluated series, so even a unanimous
    image result is not proof that the complete study contains one knee.
    """
    if not math.isfinite(high_score) or not .5 < high_score < 1:
        raise ValueError('invalid_scope_score_threshold')
    if not math.isfinite(min_abs_x_mm) or min_abs_x_mm <= 0:
        raise ValueError('invalid_scope_geometry_threshold')
    rows = list(series_results)
    ids = [r.get('series_uid') for r in rows]
    if any(not isinstance(sid, str) or not sid for sid in ids) or len(set(ids)) != len(ids):
        raise ValueError('unique_nonempty_series_ids_required')
    groups = {'L': [], 'R': [], 'unresolved': []}
    details = []
    for row in sorted(rows, key=lambda r: r['series_uid']):
        vote = reconcile_series_sides([row], enable_image_estimate=True, high_score=high_score)
        key = vote['side'] if vote['side'] in ('L', 'R') else 'unresolved'
        groups[key].append(row['series_uid'])
        details.append(dict(series_uid=row['series_uid'], side=vote['side'], issues=vote['issues']))
    if groups['L'] and groups['R']:
        status, hint = 'opposite_image_sides', ''
    elif groups['unresolved'] or not rows:
        status, hint = 'incomplete_image_evidence', ''
    else:
        status, hint = 'single_image_side', ('L' if groups['L'] else 'R')
    centers = geometry_centers or {}
    if any(not isinstance(k, str) or not k or isinstance(v, bool)
           or not isinstance(v, (int, float)) or not math.isfinite(v) for k, v in centers.items()):
        raise ValueError('invalid_series_geometry_centers')
    positive = sorted(k for k, v in centers.items() if v >= min_abs_x_mm)
    negative = sorted(k for k, v in centers.items() if v <= -min_abs_x_mm)
    return dict(status=status, image_side_hint=hint, series_groups=groups, series=details,
                opposite_coordinate_groups=bool(positive and negative),
                coordinate_groups=dict(positive=positive, negative=negative),
                geometry_only_series=sorted(set(centers)-set(ids)),
                scope_basis='image_evaluated_series_only',
                whole_study_single_knee_verified=False,
                bilateral_anatomy_verified=False,
                canonical_input_plan=[], labels_changed=False)
