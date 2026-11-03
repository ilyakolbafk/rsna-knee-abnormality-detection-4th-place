"""The exact frozen14child/7parent v8.3 routing, no general error waiver."""
from pathlib import Path
import json
from rsna_public_npz import need
POLICY=json.loads((Path(__file__).parent/'RSNA_BILATERAL_POLICY.frozen.json').read_text())['samples']

def ordered(records):
    need(isinstance(records,list) and all(isinstance(x,dict) and set(x)=={'series','assigned_side','reason'} for x in records),'ROUTE_ERROR_SCHEMA')
    return sorted(records,key=lambda x:x['series'])

def validate(part,side,native,context,sops):
    key=part['SampleID'];p=POLICY.get(key);errors=side['source_errors'];coverage=part['actual_SOP_coverage']
    if p is None:
        need(not errors and not native.get('errors',[]) and not coverage['rejected_series_and_routes'] and context is None,'UNAPPROVED_SOURCE_ERRORS')
        return len(sops)
    need(context is not None and context['sample_id']==key and context['sibling_id']==p['sibling'],'ROUTE_CONTEXT')
    own_sids={v['SeriesInstanceUID'] for v in native['series']}
    need(len(own_sids)==len(native['series']) and own_sids==set(p['own_series']),'ROUTE_OWN_SERIES')
    need(part['parent_study_uid']==p['parent'] and part['report_group']==p['report_group'],'ROUTE_FROZEN_IDENTITY')
    need(native['orientation']['route']==p['side'] and side['orientation']['route']==p['side'],'ROUTE_EXPLICIT_SIDE')
    expected=p['expected_errors']
    for records in [errors,native['errors'],coverage['rejected_series_and_routes']]:need(ordered(records)==expected,'ROUTE_EXACT_OPPOSITE_RECORDS')
    sibling=context['sibling'];sp,sn,ss,sc=[sibling[k] for k in ['part','native','side','config']];q=POLICY[p['sibling']]
    need(q['sibling']==key and q['parent']==p['parent'] and q['report_group']==p['report_group'] and q['side']!=p['side'],'ROUTE_RECIPROCAL_POLICY')
    need(sp['SampleID']==ss['SampleID']==sc['SampleID']==p['sibling'],'ROUTE_SIBLING_SAMPLE')
    need(sp['parent_study_uid']==sn['parent_study_uid']==sc['parent_study_uid']==p['parent'] and sp['report_group']==sc['report_group']==p['report_group'],'ROUTE_SIBLING_GROUP')
    need(sp['partition']==sc['partition']==ss['partition']==part['partition']=='fit','ROUTE_PARTITION')
    need(sn['orientation']==ss['orientation'] and sn['orientation']['route']==q['side'] and sn['orientation']['laterality_resolver_calls']==1,'ROUTE_SIBLING_ORIENTATION')
    need({v['SeriesInstanceUID'] for v in sn['series']}==set(q['own_series'])==set(p['opposite_series']) and set(q['opposite_series'])==own_sids,'ROUTE_SIBLING_SERIES')
    for records in [sn['errors'],ss['source_errors'],sp['actual_SOP_coverage']['rejected_series_and_routes']]:need(ordered(records)==q['expected_errors'],'ROUTE_SIBLING_ERRORS')
    # The sourceframes here are canonical native rows, not slab/downsampled rows.
    own={f['SOPInstanceUID'] for s in native['series'] for f in s['source_frames']}
    other={f['SOPInstanceUID'] for s in sn['series'] for f in s['source_frames']}
    need(own==sops and len(own)==p['own_expected_SOPs'] and len(other)==q['own_expected_SOPs'],'ROUTE_NATIVE_SOP_COUNT')
    need(not own&other and len(own|other)==p['parent_expected_SOPs']==q['parent_expected_SOPs'],'ROUTE_PARENT_SOP_CLOSURE')
    need(coverage['excluded_frame_count']==sp['actual_SOP_coverage']['excluded_frame_count']==0 and coverage['native_retained_SOP_count']==len(own) and sp['actual_SOP_coverage']['native_retained_SOP_count']==len(other),'ROUTE_CHILD_COVERAGE')
    need(coverage['source_member_count']==sp['actual_SOP_coverage']['source_member_count']==p['parent_expected_SOPs'],'ROUTE_PARENT_MEMBER_COUNT')
    return p['parent_expected_SOPs']
