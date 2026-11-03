"""Exact final RSNA collection, original per-batch transport proof chains.

The explicit author-v4 basis is preserved. This integration candidate keeps the
production semantic gate closed until exact producer source and geometry release
are available. protocol_probe is an explicit CPU-QA purpose, never a fit default.
"""
from pathlib import Path
import copy, hashlib, json, os, stat
from rsna_public_npz import need, STATE
from rsna_public_reader import TransportStudyReader, fingerprint

BACKEND='rsna_public_npz_collection'

def canonical(entry):
    return {k:entry[k] for k in ['sample_id','parent_id','report_group','source_id','diagnostic_Gold58']}

class CollectionReader:
    def __init__(self,descriptor,cache_bytes=128*2**20,*,diagnostic=False,purpose='training',verify_target=True):
        self.d=copy.deepcopy(descriptor);self.diagnostic=diagnostic;self.purpose=purpose
        need(descriptor.get('backend')==BACKEND and descriptor.get('profile')=='full_v4','COLLECTION_BACKEND')
        need(descriptor.get('orientation_state')==STATE,'COLLECTION_EXPLICIT_BASIS')
        need(purpose in ['training','protocol_probe'],'COLLECTION_PURPOSE')
        # Never reinterpret an unsigned boolean, transport PASS, or probe as a
        # production admission. Final semantic release is still a dependency.
        if purpose=='training':
            from actual_acceptance import require_rsna_release
            self.acceptance=require_rsna_release(descriptor)
        path=Path(descriptor['registry_path']);need(path.is_file() and not path.is_symlink(),'REGISTRY_PATH')
        wire=path.read_bytes();need(len(wire)<=32*2**20 and hashlib.sha256(wire).hexdigest()==descriptor['registry_sha256'],'REGISTRY_SHA')
        self.registry=json.loads(wire);r=self.registry
        need(r['schema']=='h6_rsna_public_final_registry_v1' and r['orientation_state']==STATE,'REGISTRY_SCHEMA')
        self.root=Path(descriptor['root']);need(self.root.is_absolute() and str(self.root)==r['root'],'COLLECTION_ROOT')
        self.verify_target=verify_target
        need(r['target']['instance_id']==56967269 and r['target']['machine_id']==1569 and r['target']['hostname']=='56dc3869206e','COLLECTION_TARGET')
        need(r['target']['boot_id']=='d7c9a413-650b-4f6e-b872-2fa572010446','COLLECTION_BOOT')
        if verify_target:
            need(os.uname().nodename=='56dc3869206e' and Path('/root/.vast_containerlabel').read_text().strip()=='C.56967269','LIVE_TARGET')
            need(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==r['target']['boot_id'],'LIVE_BOOT')
        need(descriptor['destination_fullsha_proof_sha256']==r['destination_fullsha_proof_sha256'],'COLLECTION_FULLSHA_BINDING')
        fit,gold=set(r['fit_ids']),set(r['Gold_ids'])
        need(len(r['fit_ids'])==len(fit)==4355 and len(r['Gold_ids'])==len(gold)==58 and not fit&gold and set(r['studies'])==fit|gold,'COLLECTION_EXACT_ROSTER')
        need({k:canonical(v) for k,v in r['studies'].items()}==descriptor['studies'],'COLLECTION_LABEL_IDENTITY')
        need(set(r['test3'])<=fit and len(set(r['test3']))==3,'COLLECTION_TEST3')
        for k,v in r['studies'].items():
            need(k==v['sample_id'] and v['source_id']==0 and bool(v['parent_id']) and bool(v['report_group']),'COLLECTION_STUDY_IDENTITY')
            need(v['diagnostic_Gold58'] is (k in gold) and v['partition']==('Gold' if k in gold else 'fit'),'COLLECTION_PARTITION')
        for f in ['parent_id','report_group']:
            need(not {r['studies'][k][f] for k in fit}&{r['studies'][k][f] for k in gold},'COLLECTION_GOLD_GROUP_LEAKAGE')
        self.files={**r['files'],**r['root_metadata']};self.pid=None;self.unit=None;self.unit_key=None
        for rel in r['root_metadata']:self._json(rel)

    def __getstate__(self):
        state=self.__dict__.copy();state.update(pid=None,unit=None,unit_key=None);return state

    def _path(self,rel):
        p=Path(rel);need(not p.is_absolute() and '..' not in p.parts and rel in self.files,'COLLECTION_PATH_MEMBERSHIP')
        need(not any(x.is_symlink() for x in [self.root,*self.root.parents]),'COLLECTION_ROOT_SYMLINK')
        full=self.root
        for part in p.parts:full=full/part;need(not full.is_symlink(),'COLLECTION_SYMLINK')
        s=full.stat();need(stat.S_ISREG(s.st_mode) and fingerprint(s)==self.files[rel]['stat'],'COLLECTION_STAT')
        return full

    def _json(self,rel):
        p=self._path(rel);expected=self.files[rel]
        with p.open('rb') as stream:
            need(fingerprint(os.fstat(stream.fileno()))==expected['stat'],'COLLECTION_OPEN_STAT')
            wire=stream.read();need(fingerprint(os.fstat(stream.fileno()))==expected['stat'],'COLLECTION_READ_STAT')
        self._path(rel);need(hashlib.sha256(wire).hexdigest()==expected['sha256'],'COLLECTION_METADATA_SHA')
        return json.loads(wire)

    def _reader(self,key):
        need(key in self.d['studies'],'COLLECTION_UNKNOWN_SAMPLE')
        entry=self.registry['studies'][key]
        need(entry['diagnostic_Gold58'] is self.diagnostic,'COLLECTION_GOLD_MODE')
        if self.pid!=os.getpid():self.pid=os.getpid();self.unit=None;self.unit_key=None
        if self.unit_key==key:return self.unit
        self.unit=None;self.unit_key=None
        batch=entry['batch'];b=self.registry['batches'][batch]
        metadata={}
        for rel in [b['proof'],b['manifest'],*entry['metadata_files']]:
            info=self.files[rel];name=str(Path(rel).relative_to(batch))
            metadata[name]={'document':self._json(rel),'sha256':info['sha256'],'stat':info['stat'][:3],'inode':info['stat'][4]}
        # Only seven JSON documents enter the old unit protocol. NPZ payloads
        # are NEVER put in bundle.files or SHA-read by metadata verification.
        need(len(metadata)==7 and all(k.endswith('.json') for k in metadata),'COLLECTION_METADATA_ONLY_BUNDLE')
        bundle={'root':str(self.root/batch),'target':{'instance':56967269,'hostname':'56dc3869206e','boot_id':self.registry['target']['boot_id']},'files':metadata}
        from rsna_routes import POLICY as route_policy
        if key in route_policy:
            sibling_id=route_policy[key]['sibling'];se=self.registry['studies'][sibling_id]
            need(se['parent_id']==entry['parent_id'] and se['report_group']==entry['report_group'] and se['partition']==entry['partition']=='fit','ROUTE_REGISTRY_SIBLING')
            sibling={}
            for rel in se['metadata_files']:
                name=Path(rel).name
                field={'PART_MANIFEST.json':'part','CONFIG_BINDING.json':'config','NATIVE_PROVENANCE.json':'native'}.get(name)
                if field:sibling[field]=self._json(rel)
                elif '/studies/' in rel and rel.endswith('.json'):sibling['side']=self._json(rel)
            need(set(sibling)=={'part','config','native','side'},'ROUTE_SIBLING_METADATA')
            bundle['routing_context']={'sample_id':key,'sibling_id':sibling_id,'sibling':sibling}
        unit=TransportStudyReader(bundle,purpose=self.purpose,verify_target=self.verify_target)
        a=unit.assessment
        need(a['sample_id']==key and a['parent_id']==entry['parent_id'] and a['report_group']==entry['report_group'] and a['rows']==entry['rows'],'COLLECTION_UNIT_IDENTITY')
        actual_payload=str(Path(batch)/a['payload_path'])
        need(actual_payload==entry['payload'],'COLLECTION_PAYLOAD_PATH')
        expected=self.files[actual_payload];bound=a['payload_destination_binding'];ps=bound['stat']
        need(bound['sha256']==expected['sha256'] and [ps[k] for k in ['size','mtime_ns','ctime_ns','device','inode']]==expected['stat'],'COLLECTION_PAYLOAD_FINAL_BINDING')
        expected_partition='Gold' if self.diagnostic else 'fit'
        need(a['partition']==expected_partition,'COLLECTION_UNIT_PARTITION')
        self.unit=unit;self.unit_key=key;return unit

    def study(self,key):return self._reader(key).study(key)

    def windows(self,key,**kwargs):
        from full_cache_loader import Reader
        return Reader.windows(self,key,**kwargs)
