"""Versioned registration/per-frame-group adaptation; frozen v1 stays unchanged."""
from common import *
import ast,copy,gc,importlib,zipfile,resource,sys
import numpy as np
import build_case as base

FORMAT='oai-v4-v83-lps-fr-adapt-v2'
ROOT=RUN/'geometry_v2'
CONFIG=RUN/'GEOMETRY_BUILD_CONFIG.frozen.json'

def decoder():
    node=next(n for n in ast.parse((RUN/'code/build_case.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='decode')
    hits=[]
    for parent in ast.walk(node):
        for field,value in ast.iter_fields(parent):
            if not isinstance(value,list):continue
            for i,n in enumerate(value):
                if isinstance(n,ast.Raise) and isinstance(n.exc,ast.Call) and n.exc.args and isinstance(n.exc.args[0],ast.Constant) and n.exc.args[0].value=='CROSS_SERIES_FRAME_OF_REFERENCE_REQUIRES_REGISTRATION_EVIDENCE':
                    hits.append({'line':n.lineno,'before':ast.unparse(n),'after':'pass; mandatory registration or independent same-frame SKM anchor below'})
                    value[i]=ast.copy_location(ast.Pass(),n)
    assert len(hits)==1
    scope=dict(base.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),scope)
    return scope['decode'],hits

def originals():
    import training
    recovery=RUN/'registration_recovery';folder=recovery/'source/cloud_code/training'
    if str(folder) not in training.__path__:training.__path__.append(str(folder))
    state=json.loads((recovery/'RECOVERY_STATE.json').read_text())
    pins={Path(x['file']).name:x['sha256'] for x in state['files']+state['copied_original_dependencies']}
    registration=importlib.import_module('training.series_registration_rigid')
    runtime=importlib.import_module('training.registration_runtime')
    for name,m in list(sys.modules.items()):
        path=getattr(m,'__file__',None)
        if name.startswith('training.') and path and Path(path).name in pins:
            assert sha(path)==pins[Path(path).name],'ORIGINAL_REGISTRATION_COMPONENT_CONFLICT'
    align=base.exact_function(folder/'geometry_bank.py','alignment_quality',{'np':np})
    return registration,runtime,align

def composite_registration_adapter(registration,row):
    """Separately hashed effective module; original numeric AST is preserved."""
    group=json.loads((HANDOFF/'reference/oai/SOURCE_GROUPS.frozen.json').read_text())['groups'].get(row['case_id'])
    assert group and group['composite_allowed'],'UNAUTHORIZED_CROSS_STUDY_REGISTRATION'
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    adapted=importlib.import_module('geometry_composite_registration.series_registration_rigid')
    runtime=importlib.import_module('geometry_composite_registration.registration_runtime')
    policy=ROOT/'geometry_composite_registration/ELIGIBLE_PAIRS.frozen.json'
    assert adapted.v1.ELIGIBILITY_SHA256==sha(policy)
    return adapted,runtime,{'kind':'explicit_frozen_composite_eligibility_adapter','case_id':row['case_id'],
        'allowlist_sha256':sha(policy),'effective_registration_sha256':sha(adapted.v1.__file__),
        'builder_sha256':sha(__file__),'image_acceptance_thresholds_changed':False,'actual_uids_modified':False}

def build(case):
    config=json.loads(CONFIG.read_text());start=time.time()
    for name,digest in config['adaptation_component_sha256'].items():assert sha(RUN/name)==digest,'STALE_GEOMETRY_COMPONENT:'+name
    base.resources();base.torch.set_num_threads(1)
    row=next(r for r in roster() if r['case_id']==case);work=ROOT/'work'/case;work.mkdir(parents=True,exist_ok=True)
    decode,diff=decoder();series=decode(row);decision=base.decide(series,row)
    series=[base.orient(s,decision) for s in series]
    registration,runtime,aligned=originals();groups={};native={};geometry={};original_affines={}
    for s in series:
        key=(s['source_study_uid'],s['frames'][0]['frame_uid']);groups.setdefault(key,[]).append(s)
        p=work/f"{s['family']}.npy";np.save(p,s['volume'],allow_pickle=False)
        original_affines[s['sid']]=s['aff'].copy()
        native[s['sid']]={'path':str(p),'SeriesInstanceUID':s['sid'],'native_shape':list(s['volume'].shape),'plane':s['plane'],'fluid_sensitive':s['fluid'],
            'spacing_mm':[[float(np.linalg.norm(a[:,1])),float(np.linalg.norm(a[:,0]))] for a in s['aff']]}
        geometry[s['sid']]={'StudyInstanceUID':s['source_study_uid'],'SeriesInstanceUID':s['sid'],'FrameOfReferenceUID':key[1],'archive_sha256':s['archive_sha256'],'archive_bytes':s['archive_bytes'],'frozen_case_id':case,'frozen_key':[row['subjectkey'],row['side'],row['visit']],
            'affines_col_row_origin_lps_mm':s['aff'].tolist(),'quality':aligned(s['aff'])}
        assert key[0] and key[1] and geometry[s['sid']]['quality']['aligned'],'INVALID_REAL_FRAME_GROUP_GEOMETRY'
        for f in s['frames']:f.pop('image',None)
    assert len(groups)>1,'GEOMETRY_ADAPTATION_ONLY_FOR_ACTUAL_MULTI_FRAME_CASE'
    reference=next(s for s in series if s['family']=='sagiw');refkey=(reference['source_study_uid'],reference['frames'][0]['frame_uid'])
    anchor_by_sid={};group_records=[]
    def independent(members):
        geo={s['sid']:geometry[s['sid']] for s in members}
        rec=base.MemoryAnchorBank(base.MODEL,geo).record(members[0]['source_study_uid'],{'series':[native[s['sid']] for s in members]})
        if not str(rec.get('method','')).startswith('skm_tea_joint') or rec['joint_lps'] is None:
            from frame_localizer import localize
            member=min(members,key=lambda s:base.FAMILIES.index(s['family']))
            try:rec,evidence=localize(member,native[member['sid']],geometry[member['sid']],work,registration,runtime,aligned)
            except AssertionError as error:
                if str(error)!='NO_SKM_JOINT_ON_OWN_FRAME_SAGITTAL_REFORMAT':raise
                assert rec.get('method')=='slab_prior' and rec['joint_lps'] is not None,'NO_ORIGINAL_SAME_FRAME_ANCHOR_FALLBACK'
                evidence={'policy':'explicit_original_AnchorBank_same_frame_slab_prior','reason':'native and declared reformat SKM gates did not produce an accepted candidate',
                    'original_record_method':rec['method'],'original_record_why':rec['why'],'source_frame_group_families':[s['family'] for s in members],
                    'anchor_producer_sha256':sha(base.PUBLIC/'knee_anchor_bank_v2.py'),'geometrical_prior_not_skm_localization':True,
                    'cross_frame_anchor_averaging':False,'silent_field_center_substitution':False,'mandatory_all_view_numeric_and_visual_QA':True}
            rec['declared_localization_adaptation']=evidence
        return rec
    refanchor=independent(groups[refkey])
    for key,members in sorted(groups.items(),key=lambda item:item[0]!=refkey):
        reg=None;adapter=None;policy='independent_same_frame_skm';anchor=None
        if key==refkey:anchor=refanchor
        else:
            # Pair selection uses source metadata only, before image correspondence.
            target=min(members,key=lambda s:base.FAMILIES.index(s['family']))
            source=native[reference['sid']];target_native=native[target['sid']]
            active_registration,active_runtime=registration,runtime
            if key[0]!=refkey[0]:active_registration,active_runtime,adapter=composite_registration_adapter(registration,row)
            with active_runtime.sampling_session(active_runtime.NativeSamplingCache(ram_budget_bytes=0)):
                    reg=active_registration.register_pair(source,target_native,geometry[reference['sid']],geometry[target['sid']],refanchor['joint_lps'],
                        max_translation_mm=12.,max_rotation_deg=15.,grid_mm=3.,extent_mm=100.,cache_dir=work/('registration_composite_'+sha(__file__)[:12] if adapter else 'registration'),
                        source_sha256=sha(source['path']),target_sha256=sha(target_native['path']))
            if reg['accepted']:
                for s in members:
                    s['aff']=registration.transformed_affines(original_affines[s['sid']],reg)
                    assert aligned(s['aff'])['aligned'],'REGISTERED_WITHIN_SERIES_GEOMETRY'
                anchor=copy.deepcopy(refanchor);policy='accepted_image_registration_with_frozen_composite_adapter' if adapter else 'original_accepted_target_frame_to_reference'
        if anchor is None:
            anchor=independent(members)
            if anchor.get('declared_localization_adaptation'):
                policy='independent_same_frame_sagittal_reformat_skm' if str(anchor['method']).startswith('skm_tea_joint') else 'explicit_original_same_frame_slab_prior'
        if anchor['method']=='slab_prior':
            policy='accepted_registration_of_explicit_original_slab_prior' if reg and reg['accepted'] else 'explicit_original_same_frame_slab_prior'
        projections={}
        for s in members:
            z,c,r,dist=base.KB.project(s['aff'],np.asarray(anchor['joint_lps'],float));h,w=s['volume'].shape[-2:]
            assert 0<=r<h and 0<=c<w and dist<=max(5.,float(np.linalg.norm(s['aff'][1,:,2]-s['aff'][0,:,2]))*1.6),'GROUP_JOINT_OUTSIDE_SOURCE_FIELD'
            projections[s['sid']]={'slice':z,'column':c,'row':r,'distance_mm':dist,'in_field':True}
            anchor_by_sid[s['sid']]=anchor
        group_records.append({'actual_study_uid':key[0],'actual_frame_uid':key[1],'families':[s['family'] for s in members],
            'series_uids':[s['sid'] for s in members],'policy':policy,'anchor':anchor,'registration':reg,'registration_execution_adapter':adapter,'joint_projections':projections,
            'original_oriented_affines_lps':{s['sid']:original_affines[s['sid']].tolist() for s in members},
            'render_affines_lps':{s['sid']:s['aff'].tolist() for s in members}})
    pxparts=[];mparts=[];audit={'blocks':[],'lineage':[],'view_geometry':[]};offset=0
    for real,s in enumerate(series):
        px,meta,a=base.render([s],anchor_by_sid[s['sid']],row);meta[:,0]=real;meta[:,11]=real
        for block in a['blocks']:block['views']=[[vid,start+offset] for vid,start in block['views']]
        for g in a['view_geometry']:g['row']+=offset
        for name in audit:audit[name].extend(a[name])
        pxparts.append(px);mparts.append(meta);offset+=len(px)
    px=np.concatenate(pxparts);meta=np.concatenate(mparts)
    anchor={'method':'frame_group_registration_or_independent_skm_v2','groups':group_records,
            'per_series':{sid:{k:value.get(k) for k in ['method','joint_lps','patella_lps','why']} for sid,value in anchor_by_sid.items()},
            'unregistered_frames_averaged':False,'slab_prior_fallback_allowed':True,
            'slab_prior_fallback_count':sum(g['anchor']['method']=='slab_prior' for g in group_records)}
    out=RUN/'cache'/f'{case}.npz';out.parent.mkdir(exist_ok=True);tmp=out.with_suffix('.npz.partial')
    payload={'px':px,'meta':meta,'orientation_state':np.asarray('canonical_L_v8_3'),'case_id':np.asarray(case),'format':np.asarray(FORMAT),
        'roster_sha256':np.asarray(sha(ROSTER)),'recipe_sha256':np.asarray(config['semantic_sha256']),
        'orientation_policy_json':np.asarray(json.dumps(config['orientation_policy'],sort_keys=True))}
    with tmp.open('wb') as raw:
        with zipfile.ZipFile(raw,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
            for name,value in payload.items():
                with archive.open(name+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,value,allow_pickle=False)
        raw.flush();os.fsync(raw.fileno())
    with np.load(tmp,allow_pickle=False) as z:assert np.array_equal(z['px'],px) and np.array_equal(z['meta'],meta),'GEOMETRY_CACHE_ARRAY_PARITY'
    os.replace(tmp,out)
    result={'case_id':case,'status':'GEOMETRY_V2_CANDIDATE_PENDING_QA','bytes':out.stat().st_size,'sha256':sha(out),
        'pixel_sha256':hashlib.sha256(px.tobytes()).hexdigest(),'metadata_sha256':hashlib.sha256(meta.tobytes()).hexdigest(),
        'rows':len(px),'metadata_columns':base.META_COLS,'roster_sha256':sha(ROSTER),'source_pool_sha256':'ec7a85668e3c0ae2caf445be7756f247e9970c6078008065e91dfff88e2527a8',
        'recipe_sha256':config['semantic_sha256'],'previous_recipe_sha256':config['previous_semantic_sha256'],'config_sha256':sha(CONFIG),'code_sha256':sha(__file__),
        'orientation_policy':config['orientation_policy'],'orientation_state':'canonical_L_v8_3','decision':decision,'anchor':anchor,'audit':audit,
        'adaptation':FORMAT,'decode_ast_diff':diff,'reader_bitexact':True,'private_hrepro_parity':False,'labels_read':False,
        'seconds':time.time()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    dump(RUN/'cache'/f'{case}.audit.json',result);dump(ROOT/'cases'/f'{case}.build.json',{'case_id':case,'status':result['status'],'seconds':result['seconds'],'recipe_sha256':result['recipe_sha256'],'unix':time.time()})
    print(json.dumps({'case_id':case,'status':result['status'],'seconds':result['seconds'],'groups':[{k:g[k] for k in ['families','policy']} for g in group_records]}),flush=True)
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('case');args=p.parse_args();build(args.case)
