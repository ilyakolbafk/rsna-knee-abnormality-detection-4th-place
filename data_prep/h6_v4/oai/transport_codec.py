"""Reversible transport only; receiver must reconstruct the original NPZ SHA."""
from pathlib import Path
import io,json,hashlib,zipfile,os,time,subprocess,uuid
import numpy as np

FORMAT='oai-npz-delta-transport-v1'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def transform(raw,method,decode=False):
    stream=io.BytesIO(raw);version=np.lib.format.read_magic(stream)
    assert version in [(1,0),(2,0)]
    reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
    shape,fortran,dtype=reader(stream);offset=stream.tell()
    assert dtype==np.uint8 and len(shape)==3 and shape[1:]==(384,384) and not fortran
    assert 0<shape[0]<=2700 and int(np.prod(shape))*dtype.itemsize==len(raw)-offset
    value=np.frombuffer(raw,dtype=dtype,offset=offset).reshape(shape)
    header=raw[:offset];axis={'left':-1,'up':-2,'two_dimensional':-2}[method]
    if decode:
        result=np.cumsum(value,axis=axis,dtype=np.uint8)
        if method=='two_dimensional':result=np.cumsum(result,axis=-1,dtype=np.uint8)
    else:
        result=value.copy()
        if axis==-1:result[...,1:]-=value[...,:-1]
        else:result[:,1:,:]-=value[:,:-1,:]
        if method=='two_dimensional':
            prior=result.copy();result[...,1:]-=prior[...,:-1]
    return header+result.tobytes(order='C')

def pack(source,destination,method='left',level=1):
    started=time.monotonic();entries=[]
    with zipfile.ZipFile(source) as original,zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=level,allowZip64=True) as out:
        comment=original.comment.hex()
        for info in original.infolist():
            assert info.compress_type==zipfile.ZIP_DEFLATED and not info.is_dir()
            raw=original.read(info.filename)
            meta={k:getattr(info,k) for k in ['filename','date_time','create_version','create_system','extract_version','reserved','flag_bits','volume','internal_attr','external_attr']}
            meta.update(extra=info.extra.hex(),comment=info.comment.hex(),raw_sha256=hashlib.sha256(raw).hexdigest())
            entries.append(meta)
            encoded=transform(raw,method) if info.filename=='px.npy' else raw
            out.writestr('entries/'+info.filename,encoded)
        manifest={'format':FORMAT,'method':method,'original_sha256':digest(source),'original_bytes':Path(source).stat().st_size,
                  'zip_comment':comment,'entries':entries,'source_compresslevel':1}
        out.writestr('manifest.json',json.dumps(manifest,separators=(',',':')))
    return {'bytes':Path(destination).stat().st_size,'sha256':digest(destination),'seconds':time.monotonic()-started,'manifest':manifest}

def pack_zstd(source,destination,method='two_dimensional',level=3):
    start=time.monotonic();destination=Path(destination)
    plain=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.plain')
    staged=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.writing')
    try:
        original=pack(source,plain,method,0)
        p=subprocess.run(['zstd','-q','-f',f'-{level}','-T1',str(plain),'-o',str(staged)],capture_output=True,text=True,timeout=60)
        if p.returncode:raise ValueError('TRANSPORT_ZSTD_ENCODE_FAILED')
        with staged.open('rb') as f:os.fsync(f.fileno())
        os.replace(staged,destination)
        return {'format':FORMAT,'codec':'zstd','method':method,'level':level,'bytes':destination.stat().st_size,
                'sha256':digest(destination),'original_sha256':original['manifest']['original_sha256'],'seconds':time.monotonic()-start}
    finally:
        plain.unlink(missing_ok=True);staged.unlink(missing_ok=True)

def unpack(source,destination):
    with Path(source).open('rb') as f:magic=f.read(4)
    if magic==b'\x28\xb5\x2f\xfd':
        plain=Path(destination).with_name(Path(destination).name+'.'+uuid.uuid4().hex+'.plain')
        process=subprocess.Popen(['zstd','-q','-d','-c',str(source)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            total=0
            with plain.open('wb') as out:
                for block in iter(lambda:process.stdout.read(8<<20),b''):
                    total+=len(block)
                    if total>400_000_000:raise ValueError('TRANSPORT_DECOMPRESSED_SIZE_LIMIT')
                    out.write(block)
            if process.wait(timeout=30)!=0:raise ValueError('TRANSPORT_ZSTD_DECODE_FAILED')
            result=unpack(plain,destination);result['transport_codec']='zstd';return result
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=5)
            process.stdout.close();process.stderr.close();plain.unlink(missing_ok=True)
    started=time.monotonic()
    with zipfile.ZipFile(source) as package:
        manifest=json.loads(package.read('manifest.json'));assert manifest['format']==FORMAT
        assert manifest['method'] in ['left','up','two_dimensional'] and len(manifest['entries'])<=16
        assert sum(x.file_size for x in package.infolist())<400_000_000
        with Path(destination).open('wb') as rawout:
            with zipfile.ZipFile(rawout,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as out:
                out.comment=bytes.fromhex(manifest['zip_comment'])
                for meta in manifest['entries']:
                    name=meta['filename'];assert '/' not in name and name.endswith('.npy')
                    raw=package.read('entries/'+name)
                    if name=='px.npy':raw=transform(raw,manifest['method'],decode=True)
                    assert hashlib.sha256(raw).hexdigest()==meta['raw_sha256'],'RAW_ENTRY_SHA_MISMATCH'
                    info=zipfile.ZipInfo(name,tuple(meta['date_time']))
                    for key in ['create_version','create_system','extract_version','reserved','flag_bits','volume','internal_attr','external_attr']:setattr(info,key,meta[key])
                    info.extra=bytes.fromhex(meta['extra']);info.comment=bytes.fromhex(meta['comment'])
                    info.compress_type=zipfile.ZIP_DEFLATED;info._compresslevel=manifest['source_compresslevel']
                    with out.open(info,'w',force_zip64=True) as entry:entry.write(raw)
            rawout.flush();os.fsync(rawout.fileno())
    actual=digest(destination)
    assert actual==manifest['original_sha256'],'RECONSTRUCTED_ORIGINAL_NPZ_SHA_MISMATCH'
    return {'status':'ORIGINAL_NPZ_FULLSHA_RECONSTRUCTED','sha256':actual,'bytes':Path(destination).stat().st_size,'seconds':time.monotonic()-started}
