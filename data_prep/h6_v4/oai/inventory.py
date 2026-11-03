from common import *
import collections

rows=roster()
workspace=RUN.parents[2]
candidates=[]
for root in sorted((workspace/'.codex/runs').glob('oai*')):
    candidates.extend(root.rglob('*.tar.gz'))
candidates.extend((workspace/'data').rglob('*.tar.gz'))
by_size=collections.defaultdict(list)
for row in rows:
    for item in row['objects']:
        by_size[int(item['object']['archive_bytes'])].append(item['archive_sha256'])
matches=collections.defaultdict(list)
for p in candidates:
    if p.is_symlink(): continue
    st=p.stat()
    if st.st_size in by_size:
        h=sha(p)
        if h in by_size[st.st_size]:
            matches[h].append({'path':str(p.resolve()),'bytes':st.st_size,
                              'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'sha256':h})
inventory=[]
for row in rows:
    audit=Path(row['source_audit_path'])
    historical='MISSING'
    if audit.is_file():
        historical='SHA_VERIFIED' if sha(audit)==row['source_audit_sha256'] else 'SHA_CONFLICT'
    objects=[]
    for item in row['objects']:
        o=item['object'];found=matches.get(item['archive_sha256'],[])
        objects.append({'family':o['family'],'image_file':o['image_file'],
                        'download_alias':o['download_alias'],'expected_bytes':int(o['archive_bytes']),
                        'expected_sha256':item['archive_sha256'],
                        'expected_study_uid':item['source_study_uid'],
                        'expected_series_uid':item['source_series_uid'],
                        'local_verified_candidates':found,'remote_received':False})
    inventory.append({'case_id':row['case_id'],'subjectkey':row['subjectkey'],'side':row['side'],
                      'visit':row['visit'],'historical_audit':historical,'objects':objects,
                      'local_raw_complete':all(o['local_verified_candidates'] for o in objects),
                      'received':False,'built':False,'verified':False})
result={'schema':'oai-v4-source-coverage-v1','observed_unix':time.time(),
        'roster_sha256':sha(ROSTER),'source_pool_sha256':'ec7a85668e3c0ae2caf445be7756f247e9970c6078008065e91dfff88e2527a8',
        'expected_cases':len(rows),'expected_objects':sum(len(x['objects']) for x in rows),
        'declared_archive_bytes':sum(int(x['object']['archive_bytes']) for r in rows for x in r['objects']),
        'local_raw_complete':sum(r['local_raw_complete'] for r in inventory),
        'local_verified_objects':len(matches),'historical_audit_counts':dict(collections.Counter(r['historical_audit'] for r in inventory)),
        'candidate_files_scanned':len(candidates),'rows':inventory}
dump(RUN/'SOURCE_COVERAGE.local.json',result)
dump(RUN/'LOCAL_RAW_LOOKUP.json',dict(matches))
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
