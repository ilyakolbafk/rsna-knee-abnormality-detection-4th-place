"""Read back a local bank, or all 16 train shards plus Gold; fail closed."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def verify(root):
    root=Path(root)
    c=json.loads((root/'CACHE_COMPLETE.json').read_text())
    assert c['status'] in ['PASS_LOCAL_EXT_BANK','PASS_EXT_SHARD']
    assert c['manifest_sha256']==sha(root/'EXT_MANIFEST.csv')
    frame=pd.read_csv(root/'EXT_MANIFEST.csv',dtype={'SampleID':str,'StudyInstanceUID':str})
    assert frame.SampleID.is_unique and len(frame)==c['samples']
    total=0
    for r in frame.itertuples():
        p=root/r.cache_file
        assert sha(p)==r.sha256
        with np.load(p,allow_pickle=False) as z:
            assert str(z['format'])=='k24-allseries-ext-v1'
            assert str(z['sample_id'])==r.SampleID
            assert str(z['orientation_state'])=='canonical_L_v8_3'
            assert int(z['source_image_size'])==336 and int(z['crop_mm'])==140
            bank,refs,mask,meta=z['bank'],z['refs'],z['mask'],z['meta6']
            n=len(refs)
            assert n==r.windows and n<=108
            assert bank.dtype==np.uint8 and bank.shape[1:]==(336,336)
            assert refs.dtype.kind=='i' and refs.shape==(n,3)
            if n:assert refs.min()>=0 and refs.max()<len(bank)
            assert mask.shape==(n,) and mask.all()
            assert meta.shape==(n,6) and meta.dtype==np.float32 and np.isfinite(meta).all()
            assert np.isin(meta[:,3:6],[0,1]).all() and (meta[:,3:6].sum(1)==1).all()
            for k in ['series_uid','plane','fs','is_oblique','dcls','orient','native_orientation','center_source_index']:
                assert z[k].shape==(n,), k
            assert not z['native_orientation'].any()
            assert np.isin(z['orient'],['legacy','v3']).all()
            assert z['neighbor_status'].shape==(n,3)
            assert all(p in ['Sagittal','Coronal','Axial'] for p in z['plane'])
            assert len(set(z['series_uid']))==r.series
            px=bank[refs]
            assert hashlib.sha256(px.tobytes()).hexdigest()==r.pixels_sha256
            assert hashlib.sha256(meta.tobytes()).hexdigest()==r.meta6_sha256
            assert (pd.Series(z['series_uid']).value_counts()<=12).all()
            total+=n
    assert total==c['windows']
    pr=json.loads((root/'PARITY.json').read_text())
    assert len(pr)==c['parity_parents'] and all(r['pass'] for r in pr)
    assert c['parity_pass']
    return c,frame,pr

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bank',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--train-shards',type=Path,nargs='*')
    ap.add_argument('--supplemental-parity',type=Path)
    ap.add_argument('--inventory',type=Path,required=True)
    ap.add_argument('--overlay',type=Path,required=True)
    a=ap.parse_args()
    gc,gf,gp=verify(a.bank)
    inv=pd.read_csv(a.inventory,dtype=str).fillna('')
    gold=inv[inv.source_cohort.eq('gold58')]
    assert set(gf.StudyInstanceUID)==set(gold.uid) and len(gf)==58
    gs=pd.read_csv(a.bank/'SERIES_MANIFEST.csv',dtype={'series_uid':str})
    expected=set(gold.loc[gold.sel.eq('False'),'sid'])
    assert set(gs.series_uid)==expected and len(gs)==len(expected)
    assert len(gp)==58
    nonexact=sum(r['nonexact_frames'] for r in gp)
    total_frames=sum(r['total_frames'] for r in gp)
    assert 1-nonexact/total_frames>=.999
    assert max(r['max_lsb'] for r in gp)<=1
    result=dict(status='PASS_GOLD_EXT_QA',gold_samples=58,extra_series=len(gs),
        extra_parents=int((gf.series>0).sum()),windows=int(gf.windows.sum()),
        bytes=int(gf.bytes.sum()),native_windows=int(gf.native_windows.sum()),
        extension_geometry_policy=gc['extension_geometry_policy'],
        v3_windows=int(gc['v3_windows']),excluded_orientation_windows=int(gc['excluded_orientation_windows']),
        excluded_orientation_series=int(gc['excluded_orientation_series']),
        parity_exact_frame_fraction=1-nonexact/total_frames,
        parity_nonexact_frames=nonexact,parity_total_frames=total_frames,
        parity_max_lsb=max(r['max_lsb'] for r in gp),
        parity_max_meta_error=max(r['max_meta_error'] for r in gp),
        cross_path_pixels_exact=all(r['cross_path_pixels_exact'] for r in gp),
        series_by_class=gs.groupby('dcls').size().to_dict(),
        windows_by_class=gs.groupby('dcls').windows.sum().astype(int).to_dict(),
        train_verified=False,training_started=False)
    if a.train_shards:
        assert len(a.train_shards)==16
        out=[verify(p) for p in a.train_shards]
        assert {c['shard'] for c,_,_ in out}==set(range(16))
        frames=pd.concat([f for _,f,_ in out],ignore_index=True)
        assert frames.SampleID.is_unique
        ex=[r for p in a.train_shards for r in json.loads((p/'EXCLUDED_BILATERAL.json').read_text())]
        overlay=pd.read_csv(a.overlay,dtype=str)
        assert set(frames.SampleID)|{r['SampleID'] for r in ex}==set(overlay.SampleID)
        assert not(set(frames.SampleID)&{r['SampleID'] for r in ex})
        assert len(ex)==14 and len(frames)==4338
        ss=pd.concat([pd.read_csv(p/'SERIES_MANIFEST.csv',dtype={'series_uid':str}) for p in a.train_shards])
        excluded={r['StudyInstanceUID'] for r in ex}
        expected=set(inv.loc[inv.source_cohort.eq('train')&inv.sel.eq('False')&~inv.uid.isin(excluded),'sid'])
        assert ss.series_uid.is_unique and set(ss.series_uid)==expected
        prs=[r for _,_,pr in out for r in pr]
        original_parity_count=len(prs)
        if a.supplemental_parity:
            sup=json.loads(a.supplemental_parity.read_text())
            assert sup['status']=='PASS_SUPPLEMENTAL_PARITY'
            assert sup['builder_sha256']==gc['builder_sha256']
            roster_path=a.supplemental_parity.parent/'ROSTER.json'
            assert sha(roster_path)==sup['roster_sha256']
            assert sha(a.supplemental_parity.parent/'RAW_READBACK.json')==sup['source_manifest_sha256']
            roster=json.loads(roster_path.read_text())
            sr=sup['parity']
            assert len(sr)==4 and all(r['pass'] and r['exact_hash_match'] and r['cross_path_pixels_exact'] for r in sr)
            assert {r['StudyInstanceUID'] for r in sr}=={r['StudyInstanceUID'] for r in roster['rows']}
            assert not({r['StudyInstanceUID'] for r in sr}&{r['StudyInstanceUID'] for r in prs})
            assert not({r['StudyInstanceUID'] for r in sr}&excluded)
            assert {r['StudyInstanceUID'] for r in sr}<=set(overlay.ParentStudyInstanceUID)
            prs+=sr
        assert len(prs)==200 and len({r['StudyInstanceUID'] for r in prs})==200
        assert all(c['builder_sha256']==gc['builder_sha256'] for c,_,_ in out)
        contacts=[p for root in a.train_shards for p in (root/'contacts').glob('*.png')]
        assert len(contacts)==30 and sum(p.name.startswith('R__') for p in contacts)==20
        assert sum(p.name.startswith('L__') for p in contacts)==10
        result.update(status='PASS_EXT_NUMERIC_CONTACT_REVIEW_REQUIRED',train_verified=True,train_samples=len(frames),
            excluded_bilateral_samples=len(ex),train_series=len(ss),train_windows=int(frames.windows.sum()),
            train_parity_parents=len(prs),train_original_parity_parents=original_parity_count,
            train_supplemental_parity_parents=len(prs)-original_parity_count,
            train_retained_series=int((ss.windows>0).sum()),
            train_bytes=int(frames.bytes.sum()),train_v3_windows=int(ss.v3_windows.sum()),
            train_excluded_orientation_windows=int(ss.excluded_orientation_windows.sum()),
            train_excluded_orientation_series=int(ss.excluded_series.sum()),
            train_native_windows=int(ss.native_windows.sum()),
            train_source_series_by_class=ss.groupby('dcls').size().astype(int).to_dict(),
            train_retained_series_by_class=ss[ss.windows>0].groupby('dcls').size().astype(int).to_dict(),
            train_windows_by_class=ss.groupby('dcls').windows.sum().astype(int).to_dict(),
            train_source_series_per_parent=frames.source_series.value_counts().sort_index().astype(int).to_dict(),
            train_retained_series_per_parent=frames.series.value_counts().sort_index().astype(int).to_dict())
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__':main()
