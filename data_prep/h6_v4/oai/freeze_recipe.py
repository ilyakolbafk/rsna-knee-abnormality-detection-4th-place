from common import *
import importlib.metadata,platform,cv2

files={}
for directory in [RUN/'component_discovery/source',HANDOFF/'orientation_v8_3',HANDOFF/'reference/public_v4']:
    for p in directory.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':files[str(p.relative_to(RUN))]=sha(p)
for name in ['build_case.py','reader.py']:
    p=RUN/'code'/name;files[str(p.relative_to(RUN))]=sha(p)
files[str((HANDOFF/'reference/oai/SOURCE_GROUPS.frozen.json').relative_to(RUN))]=sha(HANDOFF/'reference/oai/SOURCE_GROUPS.frozen.json')
config={'version':1,'format':'oai-v4-v83-lps-adapt-v1','orientation_policy':json.loads((HANDOFF/'orientation_v8_3/ORIENTATION_RECEIPT.json').read_text())['orientation_policy'],
 'roster_sha256':sha(ROSTER),'component_sha256':files,'metadata_columns':['series_idx','slot','plane','view_id','rel_pos','log_spacing','extent_130','fluid','fatsat','slice_order','n_slices','real_series_idx'],
 'size':384,'local_mm':100.,'thin_mm':1.75,'slab_mm':3.5,'policy':'series_companded','compression':'NPZ_DEFLATE_1_lossless',
 'slots':{'sagiw':1,'sagds':1,'cormpr':3,'coriw':4,'axmpr':5},'decode':'source exact pixels_from_dataset and quantize_series functions',
 'orientation':'source v8.3 resolver + h1_adapter applied once to full native16 stack; geometry composed from identical D4 pull map; side locator preserved',
 'anchors':'source AnchorBank v2 + original SKM-TEA weights; original slab_prior fallback explicitly counted; no field-centre fallback',
 'offsets':'source v4 offsets expressed in LPS and projected through actual affine; MedLat keyed to actual side',
 'adaptations':['private hrepro orchestration unavailable; new explicit case builder','v8.3 fullfield orientation with affine rather than legacy side_cls/rerender','world-LPS offsets required by v8.3 differing sagittal column axis','lossless per-case NPZ and bounded reader rather than corpus mmap','AST extraction of exact quantize_series and load_segmenter bodies avoids unrelated missing imports','MONAI0.6 official wheel fills runtime; author module.py 80-byte patch unavailable after Kaggle429'],
 'private_hrepro_bitexact_claim':False,'environment':{'python':platform.python_version(),'platform':platform.platform(),
    'opencv':cv2.__version__,**{m:importlib.metadata.version(m) for m in ['numpy','torch','pydicom','Pillow']}},'torch_cpu_threads_per_worker':2}
config['semantic_sha256']=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
p=RUN/'BUILD_CONFIG.frozen.json'
if p.exists() and json.loads(p.read_text())['semantic_sha256']!=config['semantic_sha256']:raise ValueError('RECIPE_ALREADY_FROZEN_NEW_VERSION_REQUIRED')
dump(p,config)
dump(RUN/'ADAPTATION_DIFF.json',{'format':config['format'],'adaptations':config['adaptations'],'private_hrepro_bitexact_claim':False})
print(json.dumps({'recipe_sha256':config['semantic_sha256'],'files':len(files),'format':config['format']}))
