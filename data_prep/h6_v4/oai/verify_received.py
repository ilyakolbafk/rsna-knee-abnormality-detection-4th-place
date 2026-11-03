from common import *
import numpy as np
from recipe_registry import validate_audit

def verify(case,request):
    rows={r['case_id']:r for r in roster()}
    if case not in rows:raise ValueError('UNKNOWN_CASE')
    incoming=RUN/'incoming'/case;final=RUN/'cache';final.mkdir(exist_ok=True)
    ap=incoming/f'{case}.audit.json';npz=incoming/f'{case}.npz'
    if not ap.exists():ap=final/f'{case}.audit.json'
    a=json.loads(ap.read_text())
    config=validate_audit(case,a)
    if request['case_id']!=case or request['sha256']!=a['sha256'] or request['audit_sha256']!=sha(ap):raise ValueError('REQUEST_SOURCE_CONFLICT')
    if a['case_id']!=case or a['recipe_sha256']!=config['semantic_sha256'] or a['roster_sha256']!=sha(ROSTER):raise ValueError('RECIPE_OR_CASE_CONFLICT')
    transport=request.get('transport');packet=None
    if transport:
        from transport_codec import FORMAT,unpack
        assert transport['format']==FORMAT and transport['file']==f"{case}.{request['request_nonce']}.transport.zst",'TRANSPORT_FORMAT_OR_FILENAME'
        if transport['codec_sha256']!=sha(RUN/'code/transport_codec.py'):raise ValueError('TRANSPORT_CODEC_VERSION_CONFLICT')
        packet=incoming/transport['file']
        if packet.stat().st_size!=transport['bytes'] or sha(packet)!=transport['sha256']:raise ValueError('TRANSPORT_WIRE_SHA_CONFLICT')
        staged=incoming/f"{case}.{request['request_nonce']}.reconstructed"
        try:
            try:result=unpack(packet,staged)
            except Exception as error:
                raise RuntimeError('TRANSPORT_REMOTE_DECODE_FAILED:'+type(error).__name__+':'+str(error)) from error
            if result['sha256']!=a['sha256'] or result['bytes']!=a['bytes']:raise ValueError('TRANSPORT_ORIGINAL_REQUEST_CONFLICT')
            os.replace(staged,npz)
        finally:staged.unlink(missing_ok=True)
    if not npz.exists():npz=final/f'{case}.npz'
    if a['bytes']!=npz.stat().st_size or a['sha256']!=sha(npz):raise ValueError('RECEIVER_SHA_CONFLICT')
    expected={x['object']['family']:x for x in rows[case]['objects']}
    for s in a['audit']['lineage']:
        e=expected[s['family']]
        if (s['archive_sha256'],s['source_study_uid'],s['series_uid'])!=(e['archive_sha256'],e['source_study_uid'],e['source_series_uid']):raise ValueError('SOURCE_BINDING_CONFLICT')
        if s['orientation_applications']!=1:raise ValueError('ORIENTATION_APPLICATION_COUNT')
    if len(a['audit']['lineage'])!=5 or {s['family'] for s in a['audit']['lineage']}!=set(expected):raise ValueError('FAMILY_COVERAGE_CONFLICT')
    with np.load(npz,allow_pickle=False) as z:
        if str(z['recipe_sha256'])!=config['semantic_sha256'] or str(z['case_id'])!=case or str(z['orientation_state'])!='canonical_L_v8_3':raise ValueError('NPZ_IDENTITY_CONFLICT')
        px=z['px'];meta=z['meta']
        if px.dtype!=np.uint8 or px.shape!=(a['rows'],384,384) or meta.dtype!=np.float32 or meta.shape!=(a['rows'],12):raise ValueError('NPZ_SHAPE_CONFLICT')
        if not np.isfinite(meta).all():raise ValueError('NONFINITE_METADATA')
        if hashlib.sha256(px.tobytes()).hexdigest()!=a['pixel_sha256'] or hashlib.sha256(meta.tobytes()).hexdigest()!=a['metadata_sha256']:raise ValueError('DECOMPRESSED_ARRAY_SHA_CONFLICT')
    for src in [npz,ap]:
        dst=final/src.name
        if src!=dst:
            with src.open('rb') as f:os.fsync(f.fileno())
            os.replace(src,dst)
    fd=os.open(final,os.O_RDONLY);os.fsync(fd);os.close(fd)
    st=(final/f'{case}.npz').stat();auditfile=final/f'{case}.audit.json'
    receipt={'case_id':case,'status':'RECEIVER_FULLSHA_AND_ARRAYS_VERIFIED','recipe_sha256':config['semantic_sha256'],
             'request_nonce':request['request_nonce'],
             'sha256':a['sha256'],'audit_sha256':sha(auditfile),'bytes':st.st_size,'rows':a['rows'],
             'inode':st.st_ino,'mtime_ns':st.st_mtime_ns,'ctime_ns':st.st_ctime_ns,
             'audit_inode':auditfile.stat().st_ino,'audit_mtime_ns':auditfile.stat().st_mtime_ns,
             'source_objects':5,'anchor_method':a['anchor']['method'],'unix':time.time(),
             'orientation_state':'canonical_L_v8_3','private_hrepro_parity':False}
    if transport:receipt.update(transport_format=transport['format'],wire_bytes=transport['bytes'],wire_sha256=transport['sha256'],original_npz_reconstructed_fullsha=True)
    dump(RUN/'cache_receipts'/f'{case}.json',receipt)
    dump(RUN/'source_receipts'/f'{case}.json',{'case_id':case,'status':'SOURCE_FULLSHA_VERIFIED_ON_MAC',
        'subjectkey':rows[case]['subjectkey'],'side':rows[case]['side'],'visit':rows[case]['visit'],
        'recipe_sha256':config['semantic_sha256'],'cache_sha256':a['sha256'],'audit_sha256':receipt['audit_sha256'],
        'raw_bytes_present_on_target':False,'raw_source_deleted_on_origin':False,
        'objects':[{'family':s['family'],'source_study_uid':s['source_study_uid'],'source_series_uid':s['series_uid'],
                    'archive_bytes':s['archive_bytes'],'archive_sha256':s['archive_sha256']} for s in a['audit']['lineage']],
        'unix':time.time()})
    if packet:packet.unlink(missing_ok=True)
    for failed in (RUN/'incoming').glob(f'{case}.*.failed.json'):
        old=json.loads(failed.read_text());wire=old.get('transport')
        if wire and wire.get('file')==f"{case}.{old['request_nonce']}.transport.zst":
            (incoming/wire['file']).unlink(missing_ok=True)
    return receipt
