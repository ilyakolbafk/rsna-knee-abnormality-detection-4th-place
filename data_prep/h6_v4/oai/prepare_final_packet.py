"""Prepare compact, reproducible delivery documentation from verified artifacts."""
from common import *
import datetime,shutil

q=json.loads((RUN/'FINAL_INTEGRITY_QA.json').read_bytes())
geo=json.loads((RUN/'geometry_v2/DELIVERY_QA.json').read_bytes())
assert q['verified']==3254 and len(geo['checks'])==26
assert all(v['status']=='PASS_FRESH_FULLSHA_NONCE_STAT_AND_CASE_RECIPE_QA' for v in geo['checks'].values())
assert not q['missing'] and not q['rejected'] and not q['stale']
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
out=RUN/'final_handoff';history=RUN/'repairs/final-handoff';history.mkdir(parents=True,exist_ok=True)
for n in ['ledger.md','ACTIVE_OPERATION.json','MONITOR_LAST.json']:
    if not (history/(n+'.before')).exists():shutil.copyfile(RUN/n,history/(n+'.before'))
status={'status':'COMPLETE_ADAPTED_PUBLIC_V4','unix':time.time(),'source_cutoff_utc':now,
 'expected':3254,'received':3254,'built':3254,'verified':3254,'missing':0,'rejected':0,'stale':0,
 'geometry_cases':26,'cache_npz_bytes':q['cache_npz_bytes'],'collection_semantic_sha256':q['collection_semantic_sha256'],
 'final_manifest_sha256':sha(RUN/'FINAL_MANIFEST.json'),'final_integrity_qa':'FINAL_INTEGRITY_QA.json',
 'runtime_retired':True,'own_scratch_bytes':0,'automation_id':'oai3254',
 'automation_state':'ACTIVE_PENDING_FINAL_HANDOFF_MIRROR','private_hrepro_parity':False}
dump(RUN/'ACTIVE_OPERATION.json',status);dump(RUN/'MONITOR_LAST.json',status)
report=f'''# OAI3254 — завершено

Срез UTC: {now}. На существующем Vast **56967269 / machine1569 / datacenter18**, в `/workspace/knee_v4_cache_20261021/oai_v4`, находятся **3254/3254** NPZ, audit, cache receipts и source receipts. Пропущенных, rejected и stale случаев — **0**. Размер NPZ — **228 251 085 480 байт (228,25 GB; 212,57 GiB)**. Исторический накопительный received_bytes pipeline не является итоговым размером.

Источник истины: frozen participant+side+visit roster SHA256 `0862f5a4ecd077422db8f7d4cb6dcf2b8e593c8eee5b1bbf8888cc15e4d86961`.
FINAL_MANIFEST SHA256: `{sha(RUN/'FINAL_MANIFEST.json')}`.
Collection: `{q['collection_semantic_sha256']}`; рецепты base3228 + geometry48:11 + geometry0d49:15. **26/26** geometry cases прошли numeric, visual всех18 видов, exact recipe QA и receiver proof. Original333 components и оба immutable34-component snapshots побайтно сохранены. Реальные Study/Series/SOP/FoR, sourcefullSHA и метки сохранены; v8.3 применён один раз.

Итоговая проверка: receiver full NPZ SHA256 и SHA распакованных массивов, source5fullSHA lineage, auditSHA, nonce и свежая привязка каждого receipt к inode/size/mtime/ctime. `RECEIPT_IDENTITY_MANIFEST.json` содержит SHA всех3254 пар receipts, NPZ/auditSHA и nonce. Это сверка receiver proofs с текущими файлами, без нового чтения всех228GB NPZ. Шесть ранних receipts повторно прошли текущий verifier с новым nonce; before/after сохранены в repairs/legacy-nonce-1792561146, NPZ/audit bytes и recipe не менялись. После ремонта production finalizer снова завершился успешно.

Это **публичная versioned adaptation v4**, `private_hrepro_parity=false`: приватный hrepro builder отсутствует. Геометрическая coverage не подтверждает клиническую точность anchors. Шесть explicit original same-frame slab_prior cases и ограничения M6000358/M6000383/M6000580 сохранены в `geometry_v2/RECIPE_QA.json` и per-case QA.

Reader: `code/collection_reader.py`, классы `CollectionReader` / `CollectionStudiesNPZ`, обёртка над неизменным frozen reader. Использовать per-case registry, не latest config для всех cases. Mixed-recipe loader с двумя workers и identity/corruption guards — PASS. Пример на target:

```python
import sys
from pathlib import Path
root = Path('/workspace/knee_v4_cache_20261021/oai_v4')
sys.path.insert(0, str(root / 'code'))
from collection_reader import CollectionReader
reader = CollectionReader(root / 'cache', max_bytes=256 * 2**20)
px, meta, audit = reader.read('OAI0001')
```

Измеренный максимум RSS одного renderer — **3 119 300 608 байт (2,91 GiB)**, не суммарный пик всего pipeline. Работали6NDA×2 →4renderer×2CPU →16TCP uploads;30GBdiskbuffer/6GBrawprefetch/liveRAMfloor1GiB. Delta/zstd сохранял exact original NPZfullSHA.

Pipeline и supervisor завершились штатно. После полного proof свои idle watcher и receiver остановлены;6CPU/6GiBRAM и резерв будущей работы освобождены под shared flock. Все8 чужих task records/limits/target сохранены. Готовые cache/audit остаются на target. Все6 рабочих каталогов Mac —0байт: scratch, build_scratch, cache, transport_cache, geometry_v2/work, geometry_v2/restores. Исторические source archives, production/orientation пакеты и QA сохранены. Новых аренд, обучений и сабмитов не было.

Автоматизацияoai3254 пока ACTIVE до SHA-подтверждения зеркала этого пакета; последним шагом будет PAUSED с readback. Другие автоматизации не меняются.

Доказательства: `FINAL_INTEGRITY_QA.json`, `geometry_v2/DELIVERY_QA.json`, `COLLECTION_READER_QA.json`, `CANARY_QA.frozen.json`, `COMPONENT_RECEIVER_QA.json`, `RUNTIME_RETIREMENT_QA.json`, `ledger.md`. Компактный пакет содержит manifest/QA/code/ledger без raw MRI и credentials. Полные данные и frozen snapshots остаются на target и в run.

Будущее восстановление требует сначала сверить live identity/singletons/FINAL_MANIFEST и причину. Команды ниже **не запускать для уже завершённого кэша**; это сохранённый маршрут возобновления на Mac:

```sh
python3 '{RUN}/code/deploy_receiver.py'
python3 '{RUN}/code/supervise.py'
```
'''
(out/'README.md').write_text(report)
ledger=f'''# OAI v4 ledger — COMPLETE

- Fresh source-of-truth cutoff UTC: {now}.
- Destination: existing56967269/machine1569/DC18; pinned hostname/boot/hostkey. Namespace `/workspace/knee_v4_cache_20261021/oai_v4`.
- Exact sets:3254NPZ +3254audit +3254cache_receipts +3254source_receipts. Missing/rejected/stale0; source object entries16270(case×5).
- Final NPZ bytes228251085480. FINAL_MANIFEST SHA256 `{sha(RUN/'FINAL_MANIFEST.json')}`; cumulative historical pipeline counter is not retained-file size.
- Collection `{q['collection_semantic_sha256']}`, recipes3228+11+15. All26geometry numeric/visual/reader/freshreceiver QA PASS. Original333 and frozen snapshots unchanged.
- FullSHA/arrays/source/audit proofs bound to nonce+freshstat for all3254. Legacy6 nonce reproof PASS; before/after preserved, NPZ/audit unchanged; subsequent locked production finalizer exit0.
- Proof `FINAL_INTEGRITY_QA.json`;3254receipt SHA identity map `final_handoff/RECEIPT_IDENTITY_MANIFEST.json`.
- Processing Mac, orientationcanonical_L_v8_3 once. Own6scratchdirs0B after receiver proof; historical source/prod/orientation retained.
- Pipeline/supervisor naturally exited. Idle own watcher/receiver retired after PID+ctime+argv+singleton checks. Shared flock released only own6CPU/6GiB/future reservations;8neighbor task records preserved.
- Authorized external writes: cache/metadata delivery only in existing namespace. Compact final mirror pending. No training/submission/new rental/cohort/label changes.
- Public v4 adaptation, private_hrepro_parity=false. Clinical anchor accuracy not claimed;6explicit slab_prior cases recorded in QA.
- Automationoai3254 ACTIVE until verified finalcompactmirror, then PAUSED; other automations untouched.
- Open blockers: none. Next: verify exact-byte final mirror and pause onlyoai3254, record readback. No pipeline restart.
- Repair history: README.md,NEXT_REPAIR_PLAN.md,repairs/monitor_snapshots; previousledger preserved in repairs/final-handoff/ledger.md.before.
'''
(RUN/'ledger.md').write_text(ledger)
finish=f'\n## Final exact-set verification {now}\nAll3254 delivered; missing/rejected/stale0; geometry26/26PASS. Fresh FINAL_INTEGRITY_QA.json; ownscratch0B, runtime retired, neighbors preserved. Sixlegacy nonce upgrades and subsequent locked finalizer PASS; NPZ/pixel/recipe unchanged. Compact final_handoff/README.md; next finalmirror then pause onlyoai3254. No new pipeline needed.\n'
for n in ['README.md','NEXT_REPAIR_PLAN.md']:
    with (RUN/n).open('a') as fp:fp.write(finish)
files=['ledger.md','FINAL_INTEGRITY_QA.json','RUNTIME_RETIREMENT_QA.json','AUTONOMY_APPROVAL_AUDIT.json','RESOURCE_PLAN.json','CONTRACT.json','ACTIVE_OPERATION.json','MONITOR_LAST.json','geometry_v2/RECIPE_QA.json','geometry_v2/DELIVERY_QA.json','repairs/LEGACY_NONCE_REPAIR.json','repairs/ACTIVE_FINALIZATION_DEADLINE_REPAIR.json','code/final_handoff_check.py','code/prepare_final_packet.py']
files += ['transport_probe/'+n for n in ['FINAL_REMOTE_CODEC_QA.json','FALLBACK_QA.json','RECEIVER_NONCE_QA.json']]
data=json.loads((RUN/json.loads((RUN/'COLLECTION_ACTIVE.json').read_bytes())['path']).read_bytes())
files += [v['path'] for v in data['case_qa'].values()]+[v['config'] for v in data['recipes'].values()]
legacy=json.loads((RUN/'repairs/LEGACY_NONCE_REPAIR.json').read_bytes())['path']
files += [str(p.relative_to(RUN)) for p in (RUN/legacy).rglob('*.json')]
files += [str(p.relative_to(RUN)) for p in (RUN/'repairs/finalization-deadline-1792558966').glob('*.json')]
for rel in set(files):
    dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(RUN/rel,dest)
manifest={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256_MANIFEST.json'}
dump(out/'SHA256_MANIFEST.json',{'status':'COMPLETE_COMPACT_PACKET_PRE_PAUSE','unix':time.time(),'files':manifest})
print(json.dumps({'status':'PACKET_READY_FOR_BINARY_SHA_MIRROR','files':len(manifest),'bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}))
