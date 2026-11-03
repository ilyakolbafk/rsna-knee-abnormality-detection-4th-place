"""Render operational findings from verified server receipts, without inference."""
import json,datetime,hashlib,tarfile
from pathlib import Path
R=Path(__file__).resolve().parent;A=R/'server_artifacts'
def read(n):return json.loads((A/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
delivered=json.loads((R/'MAC_METADATA_RECEIPT.json').read_text());bundle=R/'server_metadata.tar.gz'
assert delivered['all_selected_receiver_SHA_verified'] and not delivered['medical_arrays_and_images_transferred']
assert bundle.stat().st_size==delivered['bytes'] and sha(bundle)==delivered['sha256']
with tarfile.open(bundle,'r:gz') as t:assert t.extractfile('COMPACT_METADATA_MANIFEST.json').read()==(A/'COMPACT_METADATA_MANIFEST.json').read_bytes()
delivery_manifest=read('COMPACT_METADATA_MANIFEST.json')
for n,v in delivery_manifest['files'].items():
 path=A/n;assert path.resolve().is_relative_to(A.resolve()) and path.stat().st_size==v['bytes'] and sha(path)==v['sha256'],n
s=read('receipts/SOURCE_VERIFICATION.json');q=read('qa/LOADER_TARGET_QA.json');m=read('candidate_pixel_store/manifest.json');p=read('candidate_pixel_store/COMPLETION_RECEIPT.json');c=read('qa/REAL_CANARY_QA.json');final=read('FINAL_STATUS.json')
assert s['state']=='PASS_SOURCE_STAGE_ONLY' and q['state']=='PASS_STRUCTURAL_AND_TARGET_QA_ONLY'
terminal=json.loads((R/'TERMINAL_READBACK.json').read_text());assert terminal['state']=='PASS_TERMINAL_RECEIPT_AND_UNCHANGED_STATS' and terminal['target']==56967269 and terminal['source_and_finisher_terminal'] and terminal['own_admission_released'] and terminal['all_source_receipt_SHA_and_file_stats_unchanged']
assert not final['training_ready'] and not final['goal_complete'] and final['source']==s and final['candidate_QA']==q
assert read('receipts/SOURCE_FULLSHA_MANIFEST.json')['report']==s
assert terminal['source_manifest_sha256']==final['source_manifest_sha256']==sha(A/'receipts/SOURCE_FULLSHA_MANIFEST.json')
assert terminal['candidate_receipt_sha256']==final['candidate_receipt_sha256']==sha(A/'candidate_pixel_store/COMPLETION_RECEIPT.json')
assert terminal['candidate_files']==p['files']
assert final['adaptation_spec_sha256']==sha(A/'inputs/FM_ADAPTATION_SPEC.json')
verification=read('receipts/SOURCE_VERIFICATION_COMPLETE.json');assert verification['source_terminal_receipt_sha256']==sha(A/'receipts/SOURCE_TERMINAL_RECEIPT.json') and verification['source_manifest_sha256']==final['source_manifest_sha256'] and verification['source_verification_sha256']==sha(A/'receipts/SOURCE_VERIFICATION.json')
source_terminal_binding=read('receipts/SOURCE_TERMINAL_RECEIPT.json')['binding'];assert source_terminal_binding['source_launch_sha256']==sha(A/'LAUNCH_SOURCE_CORPUS.json') and source_terminal_binding['source_code_sha256']==s['source_code_sha256']
for sp in ['train','val']:
 proof=read('receipts/ARCHIVE_COMPLETE_'+sp+'.json');assert sha(A/'receipts'/('ARCHIVE_COMPLETE_'+sp+'.json'))==source_terminal_binding['archive_proof_sha256'][sp]
 assert {k:v for k,v in proof.items() if k!='stems'}==s['archive_proofs'][sp]
assert m['source_manifest_sha256']==p['binding']['source_manifest_sha256']==final['source_manifest_sha256']
assert p['binding']['adaptation_spec_sha256']==final['adaptation_spec_sha256']
for n in ['index.json','boxes.json','manifest.json']:assert sha(A/'candidate_pixel_store'/n)==p['files'][n]['sha256']
assert c['binding']['source_manifest_sha256']==final['source_manifest_sha256'] and c['binding']['adaptation_spec_sha256']==final['adaptation_spec_sha256'] and c['records_sha256']==sha(A/'qa/real_canary/records.json')
remote='/workspace/knee_v4_cache_20261021/fastmri_v1'
lines=[
'# fastMRI: загрузка и геометрическая adaptation на 56967269',
'',
'Официальные train+val из письма NYU скачаны и проверены на сервере. Собран отдельный кандидат трёхвидового fm384_v1. Точная приватная реализация и анатомическое соответствие Medial/Lateral не подтверждены; training_ready=false.',
'',
f"- Источник: {s['actual_volumes']} / 1172 reviewed томов, train={s['source_split_volumes']['train']}, val={s['source_split_volumes']['val']}; {s['physical_slices']} физических RSS-срезов.",
f"- Candidate: {q['n_rows']} строк [384,384] uint8, три вида на срез: fullfield(0), screen_left100mm(2), screen_right100mm(3).",
f"- Используемые тома: {m['usable_volumes']}; spatial={m['spatial_volumes']}, reviewed no findings={m['reviewed_no_findings_volumes']}; исключены {m['excluded_study_level_only']} study-artifact-only, их пиксели сохранены.",
f"- Аннотации: {s['source_spatial_rows']} valid spatial + {s['study_level_rows']} study-level. Единственная zero-height исходная рамка остаётся в карантине, без придуманной высоты.",
f"- Реальные H5 patient_id: {s['official_H5_patient_ids']} exam-групп. Stem SHA1 holdout seed: {s['stem_rule_holdout']}; конфликтующих exam-групп: {s['stem_rule_patient_conflict_groups']}; после group closure holdout: {s['patient_safe_closure_holdout']} томов. Это идентичность обследования, без утверждения о стабильной идентичности человека между обследованиями.",
f"- Фактический store четырёх файлов: {q['actual_store_bytes']:,} bytes; uint8 payload: {q['payload_bytes']:,} bytes; allocated pixel file: {q['allocated_pixel_bytes']:,} bytes. Хранилище NPY без сжатия.",
'',
'Оба архива дочитаны до конца: проверены tar boundaries/padding, XZ footer/CRC и полные compressed bytes. Полные SHA256 получателя:',
'']
for sp in ['train','val']:
 a=read('receipts/ARCHIVE_COMPLETE_'+sp+'.json');lines.append(f"- {sp}: {a['full_compressed_bytes_read']:,} bytes; `{a['local_full_archive_sha256']}`. Официальный full-archive SHA не предоставлен.")
lines += [
'',
'Receiver SHA проверены для всех source-файлов и четырёх файлов candidate. Проверены один flipud, bbox/spacing, transforms и a0 до clipping, все 11 dense targets, отсутствие перехода windows между томами/видами, crop352/RGB3 на лету и exam-group holdout. Синтетические проверки включают interruption/resume и отказ при изменении source/config/code или повреждённых строках.',
f"Canary: {c['samples']} реальных примеров, оба split, boxed/no-findings, все категории и виды, рамка вне старого110 и карантин. Числовой QA прошёл; montage созданы на сервере, завершённый визуальный обзор не заявлен.",
'',
'Нормализация/геометрия и mapping объявлены отдельной adaptation: volume p99.9 → один flipud → точная геометрия recovered author screen-fields → INTER_AREA → uint8. Собственная 21→11 таблица не называется авторской 12→11. Private pixel parity не заявлена.',
'',
f'Серверный candidate: `{remote}/candidate_pixel_store/`.',
f'Серверные montage: `{remote}/qa/real_canary/`; изображения и массивы на Mac не передавались.',
f'Возобновление после проверки отсутствия живого finisher: `{remote}/render_venv/bin/python {remote}/code/finish_pipeline.py`.',
'',
'Сборка выполнялась с общим admission OAI-соседа. Обучение, старые очереди и новая аренда не запускались. Candidate reader по умолчанию отказывает в использовании до разрешения семантического gate; stock public FmStore игнорирует такие flags, поэтому его QA выполнялся только в явном review-режиме.',
'',
'Блокер для запрошенного анатомического M/L cache: отсутствуют точный приватный producer и проверенная ориентация/anchors для standalone coronal RSS. Следующий шаг — получить этот producer либо валидированный RSS M/L mapping; исходный float RSS сохранён для пересборки.',
'',
'Проверенные manifests, SHA receipts, QA и ledger скопированы на Mac в server_artifacts/. Полный SHA доставленного metadata bundle подтверждён MAC_METADATA_RECEIPT.json.',
'',
'Сформировано: '+datetime.datetime.now(datetime.timezone.utc).isoformat(),
]
(R/'RESULT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'result':str(R/'RESULT.md'),'source_volumes':s['actual_volumes'],'physical_slices':s['physical_slices'],'candidate_rows':q['n_rows'],'training_ready':final['training_ready'],'goal_complete':final['goal_complete']}))
