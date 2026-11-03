"""Real mixed-recipe two-worker reader smoke, plus fail-closed identity guards."""
from common import *
from recipe_registry import BASE,config_for_case,validate_audit,recipe_counts
from collection_reader import CollectionStudiesNPZ,CollectionReader
import torch

if __name__=='__main__':
    fixture=RUN/'geometry_v2/loader_smoke';fixture.mkdir(parents=True,exist_ok=True)
    selected=[]
    for p in sorted((RUN/'cache').glob('*.audit.json')):
        case=p.name.removesuffix('.audit.json')
        try:
            a=json.loads(p.read_text());recipe=a['recipe_sha256']
            if recipe in [r for _,r in selected]:continue
            os.link(RUN/'cache'/f'{case}.npz',fixture/f'{case}.npz');os.link(p,fixture/p.name)
            assert sha(fixture/f'{case}.npz')==a['sha256'];selected.append((case,recipe))
        except FileNotFoundError:continue
        if len(selected)==len(recipe_counts()):break
    assert {r for _,r in selected}==set(recipe_counts()),'MISSING_RECIPE_LOADER_SAMPLE'
    data=CollectionStudiesNPZ(fixture,[c for c,_ in selected],n_win=176,mode='train')
    loader=torch.utils.data.DataLoader(data,batch_size=1,num_workers=2,multiprocessing_context='spawn',prefetch_factor=1)
    seen=[]
    for px,meta,index in loader:
        assert px.dtype==torch.uint8 and px.shape[-3:]==(3,384,384) and meta.shape[-1]==6
        seen.append(int(index[0]))
    assert set(seen)==set(range(len(selected)))
    bad=json.loads((fixture/f'{selected[0][0]}.audit.json').read_text());bad['code_sha256']='0'*64
    try:validate_audit(selected[0][0],bad)
    except AssertionError:pass
    else:raise AssertionError('ALTERED_BUILDER_MUST_FAIL')
    try:config_for_case('UNKNOWN_NOT_IN_FROZEN_ROSTER')
    except AssertionError:pass
    else:raise AssertionError('UNKNOWN_CASE_MUST_FAIL')
    result={'status':'PASS_MIXED_RECIPES_TWO_WORKER_READER_AND_IDENTITY_GUARDS','covered_recipes':[r for _,r in selected],'samples':[c for c,_ in selected],
        'workers':2,'dtype':'uint8','shape':[3,384,384],'metadata_columns':6,'frozen_reader_sha256':sha(RUN/'code/reader.py'),
        'collection_reader_sha256':sha(RUN/'code/collection_reader.py'),'recipe_registry_sha256':sha(RUN/'code/recipe_registry.py'),'unix':time.time()}
    dump(RUN/'COLLECTION_READER_QA.json',result);print(json.dumps(result))
