"""Select central centers; preserve neighbors in the original source series."""
from __future__ import annotations

import math
import numpy as np

STATUS = {'missing': 0, 'center': 2, 'native_neighbor': 1,
          'series_end_replicated': 3, 'read_error_replicated': 4,
          'geometry_unverified_missing': 5}


def center_window(source_count, trim_each_side):
    if not math.isfinite(trim_each_side) or not 0 <= trim_each_side < .5 or source_count < 1:
        raise ValueError('Trim must be finite in [0,0.5); source count must be positive')
    cut = int(math.floor(source_count * trim_each_side))
    return cut, source_count-cut  # half-open; trim source positions before validity filtering


def allocate_centers(capacities, budget, weights=None):
    """Capped weighted fairness prevents one thin-slice series taking the budget."""
    cap = np.asarray(capacities, dtype=int)
    w = np.ones(len(cap)) if weights is None else np.asarray(weights, dtype=float)
    if cap.ndim != 1 or np.any(cap < 0) or w.shape != cap.shape or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('Invalid slot capacities/weights')
    if budget < int((cap > 0).sum()):
        raise ValueError('Token budget must cover every available semantic slot')
    result = (cap > 0).astype(int)
    while result.sum() < min(budget, cap.sum()):
        cost = (result + 1) / w
        cost[result >= cap] = np.inf
        result[int(np.argmin(cost))] += 1
    return result


def select_centers(arrays, metadata, *, budget, trim_each_side=.1, slot_weights=None,
                   policy='central_uniform', rng=None, grid_phase=None):
    if budget < 1:
        raise ValueError('Positive token budget required')
    slot_count = len(metadata['legacy_slots'])
    centers = np.full(budget, -1, np.int64)
    slots = np.zeros(budget, np.int64)
    if policy == 'legacy_centers':
        old = arrays['legacy_center_refs']
        if budget != len(old):
            raise ValueError('Legacy control requires the original token count')
        return old.astype(np.int64).copy(), arrays['legacy_slot_ids'].astype(np.int64).copy(), []
    if policy not in ('central_uniform', 'central_stratified', 'central_grid'):
        raise ValueError('Unknown full-source center policy')
    if policy == 'central_grid' and (grid_phase is None or not math.isfinite(grid_phase) or not 0 <= grid_phase < 1):
        raise ValueError('Fixed grid requires a finite phase in [0,1)')
    if policy == 'central_stratified' and not isinstance(rng, np.random.Generator):
        raise ValueError('Stratified centers require an explicit independent numpy Generator')
    trims = ([float(trim_each_side)] * slot_count if np.isscalar(trim_each_side)
             else [float(x) for x in trim_each_side])
    if len(trims) != slot_count:
        raise ValueError('One trim fraction per semantic slot required')
    candidates = [np.array([], dtype=np.int64) for _ in range(slot_count)]
    records = [None] * slot_count
    for sid, series in metadata['series'].items():
        slot = series['legacy_slot']
        if slot is None:
            continue  # keep exactly the frozen five-series selection; ignore extra sequences
        start, count = int(series['offset']), int(series['source_count'])
        lo, hi = center_window(count, trims[slot])
        raw = np.arange(start+lo, start+hi)
        candidates[slot] = raw[arrays['decode_valid'][raw]]
        records[slot] = {'slot': slot, 'series_uid': sid, 'source_count': count,
                         'source_center_start': lo, 'source_center_stop': hi,
                         'trim_each_side': trims[slot], 'available_centers': len(candidates[slot])}
    counts = allocate_centers([len(x) for x in candidates], budget, slot_weights)
    at = 0
    for slot, (available, count) in enumerate(zip(candidates, counts)):
        if not count:
            continue
        pool = available
        # At a real source endpoint one channel would only repeat the center.
        # Moving the endpoint center one plane inward still exposes the edge
        # through its native side channel. Do not shrink a trimmed window or
        # move across a decode gap; retain every center when the budget fits all.
        record = records[slot]
        series = metadata['series'][record['series_uid']]
        source_start = int(series['offset'])
        source_stop = source_start + int(series['source_count'])
        lower = int(len(available)>1 and available[0]==source_start and available[1]==source_start+1)
        upper = len(available)-int(len(available)>1 and available[-1]==source_stop-1 and available[-2]==source_stop-2)
        if count > 1 and upper-lower >= count:
            pool = available[lower:upper]
        if policy in ('central_stratified', 'central_grid'):
            # Disjoint nonempty rank strata preserve slot coverage and source order.
            # The candidate pool, quotas and endpoint policy match the fixed control.
            edges = np.arange(count + 1, dtype=np.int64) * len(pool) // count
            ranks = (rng.integers(edges[:-1], edges[1:]) if policy == 'central_stratified'
                     else edges[:-1] + np.floor((edges[1:] - edges[:-1]) * grid_phase).astype(np.int64))
        else:
            # Preserve baseline inference exactly, including endpoint placement.
            ranks = ([len(pool)//2] if count == 1 else
                     np.rint(np.linspace(0, len(pool)-1, count)).astype(int))
        selected = pool[ranks]
        centers[at:at+count] = selected; slots[at:at+count] = slot; at += count
        records[slot]['selected_centers'] = int(count)
    if at == 0:
        raise ValueError('No valid central source centers; do not train an empty study')
    return centers, slots, [r for r in records if r is not None]


def source_triplets(arrays, metadata, centers, slot_ids):
    centers, slot_ids = np.asarray(centers), np.asarray(slot_ids)
    n = len(arrays['images'])
    if centers.ndim != 1 or centers.dtype.kind not in 'iu' or np.any((centers < -1) | (centers >= n)):
        raise ValueError('Invalid source center references')
    if slot_ids.shape != centers.shape or slot_ids.dtype.kind not in 'iu' or np.any((slot_ids < 0) | (slot_ids >= len(metadata['legacy_slots']))):
        raise ValueError('Invalid semantic slot IDs')
    refs = np.full((len(centers), 3), -1, np.int64)
    status = np.zeros_like(refs, dtype=np.uint8)
    valid = np.zeros(len(centers), bool)
    by_index = {s['series_index']: s for s in metadata['series'].values()}
    for t, center in enumerate(centers):
        if center < 0 or not arrays['decode_valid'][center]:
            continue
        series = by_index[int(arrays['series_index'][center])]
        if series['legacy_slot'] != int(slot_ids[t]):
            raise ValueError('Source center belongs to a different semantic slot')
        valid[t] = True
        lo, hi = series['offset'], series['offset']+series['source_count']
        for channel, wanted in enumerate((center-1, center, center+1)):
            if channel == 1:
                refs[t, channel] = center; status[t, channel] = STATUS['center']
            elif not lo <= wanted < hi:
                refs[t, channel] = center; status[t, channel] = STATUS['series_end_replicated']
            elif not arrays['decode_valid'][wanted]:
                refs[t, channel] = center; status[t, channel] = STATUS['read_error_replicated']
            elif not series['geometry_qc']['within_series_valid'] or not (arrays['geometry_valid'][center] and arrays['geometry_valid'][wanted]):
                status[t, channel] = STATUS['geometry_unverified_missing']
            else:
                refs[t, channel] = wanted; status[t, channel] = STATUS['native_neighbor']
    if not valid.any():
        raise ValueError('No valid tokens in source study')
    return {'bank_refs': refs, 'token_mask': valid, 'slot_ids': slot_ids.astype(np.int64),
            'neighbor_status': status,
            'channel_valid': (status == STATUS['center']) | (status == STATUS['native_neighbor'])}
