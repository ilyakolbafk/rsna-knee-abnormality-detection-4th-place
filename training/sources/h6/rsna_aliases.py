"""Declared exact native aliases preserve every source SOP in the audit."""
from rsna_public_npz import need
RENDER_KEYS=('positions','spacing_mm','native_shape','plane','slot','fat_suppression','fluid_sensitive','side','intensity')

def validate(side,native):
    plan=side['exact_duplicate_plan'];series=native['series'];n=len(series)
    canonical=plan['canonical_index'];reps=plan['representative_indices']
    need(plan['policy']=='collapse_exact' and len(canonical)==n and len(reps)==len(set(reps)),'ALIAS_PLAN')
    need(all(isinstance(v,int) and 0<=v<=j and canonical[v]==v for j,v in enumerate(canonical)),'ALIAS_CANONICAL_INDEX')
    need(reps==sorted(set(canonical)) and plan['removed_series']==n-len(reps),'ALIAS_REPRESENTATIVES')
    need([b['real_series_idx'] for b in side['blocks']]==reps,'ALIAS_RENDERED_COVERAGE')
    need(all(b['series']==series[j]['SeriesInstanceUID'] for b,j in zip(side['blocks'],reps)),'ALIAS_RENDERED_IDENTITY')
    removed=[j for j in range(n) if canonical[j]!=j]
    for j in removed:
        k=canonical[j];a,b=series[k],series[j]
        need(all(a.get(f)==b.get(f) for f in RENDER_KEYS),'ALIAS_CONSUMED_METADATA')
        need(a['native_array_sha256']==b['native_array_sha256'] and len(a['native_array_sha256'])==64,'ALIAS_NATIVE_PAYLOAD_BINDING')
        need(a['affines_col_row_origin_lps_mm']==b['affines_col_row_origin_lps_mm'],'ALIAS_NATIVE_GEOMETRY')
        matched=[p for p in plan['pairs'] if p.get('collapsed') is True and p['first_index']==k and p['second_index']==j]
        need(len(matched)==1 and matched[0]['pixels_exactly_equal'] is True and matched[0]['consumer_differences']==[],'ALIAS_EXACT_PAIR_PROOF')
        need(matched[0]['first_series']==a['SeriesInstanceUID'] and matched[0]['second_series']==b['SeriesInstanceUID'],'ALIAS_PAIR_IDENTITY')
    # Pair hash has dtype/shape-prefix domain in public exact_duplicates.py;
    # it is deliberately NOT compared to native_array_sha256's raw domain.
    alias_sops=set()
    for j in removed:
        frames=series[j]['source_frames']
        need(len(frames)==series[j]['native_shape'][0] and all(f['final_index']==i and f['crc_verified'] is True and f['decode_valid'] is True for i,f in enumerate(frames)),'ALIAS_SOURCE_FRAME_BINDING')
        alias_sops.update(f['SOPInstanceUID'] for f in frames)
    return alias_sops
