"""Full-field, native-matrix uint16 Series cache. No spatial crop or resize.

One linear intensity encoding is shared by an entire Series. Source ZIP and
rescale metadata remain available; runtime rendering chooses the actual view.
"""
from pathlib import Path
import hashlib
import io
import json
import time

import numpy as np
import pandas as pd
import pydicom

from .archive_cache import open_archive
from . import dicom_geometry as geom
from .common import UID,SLOTS,file_hash,save_json

# Only reader/cleanup changes followed this encoder; its pixel contract is unchanged.
COMPATIBLE_BUILDER_HASHES={'7541dd9026488533a1b1faaf38dd14b9ecb143a7c60db3598801aeaa07a4fd2b'}


class DirectorySource:
    def __init__(self,root,split):self.root=Path(root);self.split=split
    def read(self,item):return (self.root/(item.filename if hasattr(item,'filename') else item)).read_bytes()
    def infolist(self):
        from types import SimpleNamespace
        return [SimpleNamespace(filename=p.relative_to(self.root).as_posix())
                for p in (self.root/f'{self.split}_series').glob('*/*/*.dcm')]
    def close(self):pass


def quantize_series(images):
    volume=np.stack(images).astype(np.float32)
    lo=float(volume.min());hi=float(volume.max());scale=max(hi-lo,1e-6)
    encoded=np.rint((volume-lo)/scale*65535).clip(0,65535).astype(np.uint16)
    # Shared display window; encoding itself retains the full observed range.
    window=np.percentile(encoded[:,::4,::4],[1,99]).astype(float).tolist()
    return encoded,{'minimum':lo,'maximum':hi,'max_absolute_quantization_error_bound':scale/131070,
                    'display_window_uint16':window}


def prepare(root='F:/rsnaknee',output=None,*,split='train',study_ids=None,source_directory=False):
    root=Path(root);output=Path(output or root/'cache_native');output.mkdir(parents=True,exist_ok=True)
    if source_directory:
        archive=reader=DirectorySource(root,split)
        meta={'etag':'directory:'+file_hash(root/f'{split}_series.csv')+':'+str(root.resolve())}
    else:archive,reader,meta=open_archive(root)
    try:
        frame=pd.read_csv(io.BytesIO(archive.read(f'{split}_series.csv')),dtype={UID:str,'SeriesInstanceUID':str})
        expected=pd.read_csv(io.BytesIO(archive.read(f'{split}.csv')),dtype={UID:str})[UID].tolist()
        if study_ids is not None:
            chosen=set(study_ids);expected=[u for u in expected if u in chosen]
            if chosen-set(expected):raise ValueError('requested study absent from source')
        config={'format':'native_series_uint16_v1','size':0,'archive_etag':meta['etag'],
                'geometry_hash':file_hash(geom.__file__),'builder_hash':file_hash(__file__)}
        signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        compatible={signature}|{hashlib.sha256(json.dumps({**config,'builder_hash':h},sort_keys=True).encode()).hexdigest()
                                for h in COMPATIBLE_BUILDER_HASHES}
        rows={u:x.to_dict('records') for u,x in frame.groupby(UID,sort=False)}
        entries={}
        for info in archive.infolist():
            parts=info.filename.split('/')
            if len(parts)==4 and parts[0]==f'{split}_series' and parts[-1].endswith('.dcm'):
                entries.setdefault((parts[1],parts[2]),[]).append(info)
        result=[];started=time.perf_counter()
        for number,uid in enumerate(expected):
            study_path=output/'studies'/(hashlib.sha256(uid.encode()).hexdigest()[:24]+'.json')
            if study_path.exists():
                saved=json.loads(study_path.read_text(encoding='utf-8'))
                if saved['config_hash'] not in compatible:raise ValueError('native cache builder changed; use another output')
                if saved['series'] and all(Path(s['path']).exists() for s in saved['series']):
                    result.append(saved);continue
            objects={};orders={};tags=set();sagittal=[];pids=set();errors=[]
            for row in rows.get(uid,[]):
                sid=row['SeriesInstanceUID'];plane=row['Anatomical_Plane']
                if plane not in geom.CARDINAL_NORMAL:continue
                try:
                    datasets={i.filename:pydicom.dcmread(io.BytesIO(archive.read(i))) for i in entries.get((uid,sid),[])}
                    records=[geom.header_from_dataset(ds,name) for name,ds in datasets.items()]
                    order=geom.order_series(Path(sid),plane,provided_records=records)
                    objects[sid]=datasets;orders[sid]=order;tags.update(order.laterality_values)
                    if plane=='Sagittal' and order.ipp_x_median is not None:sagittal.append(order.ipp_x_median)
                    pids.update(str(getattr(ds,'PatientID','')).strip() for ds in datasets.values())
                except (FileNotFoundError,EOFError):raise  # incomplete archive is not a bad scan
                except Exception as exc:errors.append({'series':sid,'error':str(exc)[:250]})
            pids.discard('')
            if len(pids)>1:raise ValueError('conflicting patients within one Study')
            try:side=geom.decide_study_side(uid,tags,sagittal).side
            except Exception as exc:side=None;errors.append({'series':'study','error':str(exc)[:250]})
            prepared=[]
            for row in rows.get(uid,[]):
                sid=row['SeriesInstanceUID'];plane=row['Anatomical_Plane']
                if sid not in orders:continue
                order=orders[sid];images=[];positions=[];spacing=[];names=[];bad=[]
                for name,pos in zip(order.files,order.positions_mm):
                    try:
                        payload=geom.canonicalize_payload(geom.pixels_from_dataset(objects[sid][name],path=name),plane)
                        image=payload.image
                        if geom.flip_for_plane(plane,side):image=image[:,::-1]
                        if images and (image.shape!=images[0].shape or not np.allclose(payload.spacing,spacing[0],rtol=1e-4)):
                            raise ValueError('native matrix or spacing changes within Series')
                        images.append(image);positions.append(float(pos));spacing.append(list(payload.spacing));names.append(Path(name).name)
                    except Exception as exc:bad.append({'file':Path(name).name,'error':str(exc)[:250]})
                if not images:errors.append({'series':sid,'error':'no usable native slices','details':bad});continue
                encoded,intensity=quantize_series(images)
                target=output/'series'/(hashlib.sha256((uid+'/'+sid).encode()).hexdigest()[:24]+'.npy')
                target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.tmp')
                with tmp.open('wb',buffering=8*2**20) as handle:np.save(handle,encoded,allow_pickle=False)
                tmp.replace(target)
                fs=int(row['Fat_Suppression']) if pd.notna(row.get('Fat_Suppression')) else -1
                fs=fs if fs in (0,1) else -1
                prepared.append({'SeriesInstanceUID':sid,'path':str(target.resolve()),'positions':positions,
                    'spacing_mm':spacing,'native_shape':list(encoded.shape),'intensity':intensity,
                    'slot':SLOTS[(plane,fs)],'plane':plane,'fat_suppression':fs,
                    'fluid_sensitive':int(row['Fluid_Sensitive']) if pd.notna(row.get('Fluid_Sensitive')) else -1,
                    'files':names,'errors':bad,'side':side})
            study={UID:uid,'PatientID':next(iter(pids),uid),'series':prepared,'errors':errors,'config_hash':signature}
            save_json(study_path,study);result.append(study)
            progress={'processed':number+1,'expected':len(expected),'seconds':time.perf_counter()-started,
                      'failed_studies':sum(not s['series'] for s in result),'scope':'pilot' if study_ids else 'complete_split'}
            save_json(output/'prepare_progress.json',progress)
            if number%10==0 or number+1==len(expected):print(json.dumps(progress),flush=True)
            del objects
        manifest={'config':config,'studies':result,'failed_studies':[s[UID] for s in result if not s['series']],
                  'scope':'explicit_subset' if study_ids is not None else 'complete_split'}
        save_json(output/'manifest.json',manifest)
        pd.DataFrame([{UID:s[UID],'PatientID':s['PatientID']} for s in result]).to_csv(output/'patients.csv',index=False)
        return manifest
    finally:archive.close();reader.close()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default='F:/rsnaknee');p.add_argument('--output')
    p.add_argument('--selection');p.add_argument('--split',default='train',choices=['train','test'])
    p.add_argument('--source-directory',action='store_true')
    a=p.parse_args();ids=json.loads(Path(a.selection).read_text(encoding='utf-8'))['studies'] if a.selection else None
    prepare(a.root,a.output,split=a.split,study_ids=ids,source_directory=a.source_directory)
