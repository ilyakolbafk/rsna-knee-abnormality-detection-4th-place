"""Create a hash-bound local release only after numeric, meta8 and visual gates."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json
import numpy as np
import pandas as pd
import yaml

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads((RUN/p).read_text())
def write(p,x):
    p=RUN/p;tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');tmp.replace(p)
stamp=datetime.now(ZoneInfo('Europe/Moscow')).isoformat()
n=read('EXT_NUMERIC_QA.json');v=read('contact_review/VISUAL_QA.json');m=read('METADATA8_FULL_QA.json')
assert n['status']=='PASS_EXT_NUMERIC_CONTACT_REVIEW_REQUIRED'
assert n['train_verified'] and n['train_parity_parents']==200 and n['gold_samples']==58
assert n['train_native_windows']==n['native_windows']==0
assert v['status']=='PASS_VISUAL_SIDE_CONCORDANCE' and v['reviewed_cases']==30
assert v['right_cases']==20 and v['left_cases']==10 and v['side_mismatches_observed']==0
assert m['status']=='PASS_METADATA8_FULL_CORPUS'
assert m['cohorts']['train']==dict(samples=4338,windows=n['train_windows'])
assert m['cohorts']['gold58']==dict(samples=58,windows=n['windows'])
assert v['roster_sha256']==sha(RUN/'package/contact_parents30.csv')
assert len(v['cases'])==30 and len({x['path'] for x in v['cases']})==30
for x in v['cases']:assert sha(RUN/x['path'])==x['sha256']
for x in v['pages']:assert sha(RUN/x['path'])==x['sha256']
frames=[];inputs={}
for p in sorted((RUN/'train_outputs').glob('s*/EXT_MANIFEST.csv')):
    f=pd.read_csv(p,dtype={'SampleID':str,'StudyInstanceUID':str})
    f['cache_file']=[str((p.parent/name).relative_to(RUN)) for name in f.cache_file]
    f['ext_cache_path']=[str((RUN/name).resolve()) for name in f.cache_file]
    frames.append(f)
    for name in ['EXT_MANIFEST.csv','SERIES_MANIFEST.csv','CACHE_COMPLETE.json','OUTPUT_RECEIPT.json','PARITY.json']:
        inputs[str((p.parent/name).relative_to(RUN))]=sha(p.parent/name)
train=pd.concat(frames,ignore_index=True).sort_values('SampleID')
assert len(train)==4338 and train.SampleID.is_unique
train.to_csv(RUN/'TRAIN_EXT_MANIFEST.csv',index=False)
overlay=pd.read_csv(ROOT/'artifacts/production/laterality_v8_3/training_overlay_soft_v2/training_manifest4352.csv',dtype=str)
samplemap=overlay[['SampleID','ParentStudyInstanceUID','fold','report_group','side']].merge(
    train[['SampleID','cache_file','ext_cache_path','windows']],on='SampleID',how='left',validate='one_to_one')
bilateral=samplemap.ParentStudyInstanceUID.duplicated(False)
assert int(bilateral.sum())==14 and samplemap.windows.isna().equals(bilateral)
samplemap['extension_status']=np.where(bilateral,'excluded_bilateral','verified_cache')
samplemap['windows']=samplemap.windows.fillna(0).astype(int)
samplemap.to_csv(RUN/'TRAIN_EXT_SAMPLE_MAP.csv',index=False)
base=pd.read_csv(ROOT/'.codex/runs/core-v20-ordinary-k24-f3-20261013/package/assets/V_MAIN.csv',dtype={'SampleID':str})
fit=base[base.fold.ne(3)]
available=fit.SampleID.map(train.set_index('SampleID').windows).fillna(0).astype(int)
draw=24+available.map(lambda count:min(24,round(.375*count)))
contract=yaml.safe_load((RUN/'ANALYSIS_CONTRACT.yaml').read_text())
for name in ['EXT_NUMERIC_QA.json','contact_review/VISUAL_QA.json','METADATA8_FULL_QA.json',
    'supplemental_parity/ROSTER.json','supplemental_parity/SUPPLEMENTAL_PARITY.json',
    'supplemental_parity/RAW_READBACK.json','TRAIN_EXT_MANIFEST.csv','TRAIN_EXT_SAMPLE_MAP.csv',
    'package/build_ext_bank.py','package/allseries_inputs.py','verify_ext_bank.py']:
    inputs[name]=sha(RUN/name)
facts=dict(status='PASS_ENGINEERING_FACT_PACKET',factual_cutoff=stamp,
    contract_semantic_hash=contract['semantic_hash'],train=dict(samples=4338,sample_map_rows=len(samplemap),
        bilateral_empty_samples=14,parents_with_retained_extension=int((train.windows>0).sum()),
        source_series=n['train_series'],retained_series=n['train_retained_series'],windows=n['train_windows'],
        v3_windows=n['train_v3_windows'],excluded_orientation_series=n['train_excluded_orientation_series'],
        excluded_orientation_windows=n['train_excluded_orientation_windows'],bytes=n['train_bytes'],
        parity_parents=200,original_parity_parents=196,supplemental_parity_parents=4),
    gold=dict(samples=58,series=n['extra_series'],windows=n['windows'],v3_windows=n['v3_windows'],
        bytes=n['bytes'],parity_exact_frame_fraction=n['parity_exact_frame_fraction'],
        parity_nonexact_frames=n['parity_nonexact_frames'],parity_max_lsb=n['parity_max_lsb'],meta_error=0),
    visual=dict(cases=30,right=20,left=10,v3_cases=v['new_v3_cases'],observed_side_mismatches=0),
    sampler=dict(fit_samples=len(fit),base_draw=24,extra_fraction=.375,extra_cap=24,
        nominal_mean_before_series_drop=float(draw.mean()),nominal_p95_before_series_drop=float(draw.quantile(.95)),
        nominal_max=int(draw.max()),batch_recommendation=8,CUDAMemory='UNTESTED'),
    inputs=inputs,no_training_or_quality_claim=True)
write('FACT_PACKET.json',facts)
report=f'''# K24-ALL: дополнительный кэш готов — PASS_EXT_BANK

Проверено {stamp}. Все 16 приватных CPU-кернелов завершились, outputs точной v1 получены и проверены. Готов отдельный банк crop 140 мм → 336 px с последующим штатным resize 336 → 384 в K24. Базовые 64 окна сохранены.

| Результат | Train | Gold58 |
|---|---:|---:|
| Файлов расширения | {facts['train']['samples']} | 58 |
| Добавленных серий до / после v3-исключений | {facts['train']['source_series']} / {facts['train']['retained_series']} | 64 / 64 |
| Сохранённых окон | {facts['train']['windows']} | {facts['gold']['windows']} |
| Окон, дополнительно развёрнутых v3 | {facts['train']['v3_windows']} | {facts['gold']['v3_windows']} |
| Исключено из-за неопределённой ориентации | {facts['train']['excluded_orientation_series']} серия / {facts['train']['excluded_orientation_windows']} окна | 0 |
| Размер NPZ | {facts['train']['bytes']/2**30:.3f} GiB | {facts['gold']['bytes']/2**20:.1f} MiB |

Расширение осталось у {facts['train']['parents_with_retained_extension']} train-родителей. Для 14 двусторонних SampleID расширение отсутствует по контракту; карта всех 4352 сэмплов явно отмечает их. Остальные пустые банки существуют как проверенные файлы. Нативных неканонизированных окон нет.

**Проверки.** Все SHA и coverage прошли; metadata8 проверены на всех 4396 файлах / {facts['train']['windows']+facts['gold']['windows']} окнах. Train-паритет 200 родителей побитно точен по пикселям, mask и metadata, включая cross-path сборщика. Первоначальный smoke 200 содержал 4 bilateral-родителя: после их исключения осталось 196; порог 200 сохранён и дополнен 4 первыми оставшимися unilateral SampleID, зафиксированными до чтения. Их 612 исходных DICOM проверены по SHA/размеру и дали точный паритет. Новые внешние кернелы для этого не создавались.

Gold58: {facts['gold']['parity_exact_frame_fraction']*100:.6f}% кадров точны, остальные 3 кадра отличаются максимум на 1 LSB, metadata точны. Это проходит допуск 99.9% / ≤1 LSB. Кэш не заявляется 100% побайтовой копией исторического Gold.

Контактный лист 20R + 10L просмотрен целиком, включая 4 новых v3-случая: относительного несоответствия стороны основной и косой корональной не обнаружено. Это контроль этих 30 пар, не оценка 100% точности классификатора стороны.

**Передача в обучение.** Потребитель — будущий K24-ALL fold3 trainer. Ключ SampleID, окна объединяются с исходными 64 из cache336; кэш статический. `TRAIN_EXT_MANIFEST.csv` содержит 4338 относительных и абсолютных путей. `TRAIN_EXT_SAMPLE_MAP.csv` содержит 4352 строки с fold/report_group/side и явным статусом bilateral; любое другое отсутствие файла — ошибка. Загрузчик `package/allseries_inputs.py` формирует 8 признаков: старые 6 + is_extra/is_oblique. Полные распределения по dcls, v3-исключения и число серий на родителя находятся в EXT_NUMERIC_QA.json; train и Gold там разделены префиксом train_.

Для обучения рекомендую B8 и 24 базовых окна плюс `min(24,round(.375*n_extra))`. По фактическому кэшу fit-fold3 это {facts['sampler']['nominal_mean_before_series_drop']:.2f} окна в среднем до series-drop, p95={facts['sampler']['nominal_p95_before_series_drop']:.0f}, максимум 48. Общий лимит 24 вытеснял бы привычные базовые окна. B8 остаётся основным; CUDA-проверка обычной и тяжёлой восьмёрки со состояниями AdamW ещё не выполнена. Эффективный batch 8 и один optimizer/scheduler step на 8 исследований сохраняются при возможном разбиении тяжёлых групп.

Обучение, аренда, full-fit, сабмит не запускались. Следующий шаг — отдельное ОК на fold3-пилот и его memory gate. N0/N1/N2/Z, VM/Evidence/Gold-гейт и стадия 2 остаются по обновлённой спеке; текущая тренировочная очередь не менялась. Качество модели этим кэшем ещё не измерено.

Артефакты: FACT_PACKET.json, EXT_NUMERIC_QA.json, METADATA8_FULL_QA.json, supplemental_parity/SUPPLEMENTAL_PARITY.json, contact_review/VISUAL_QA.json и contact_review/CONTACT_SHEETS.md. Исходная спека: prompts/codex/K24_ALLSERIES_SPEC_20261015.md.
'''
(RUN/'REPORT.md').write_text(report)
release=dict(status='PASS_EXT_BANK',factual_cutoff=stamp,format='k24-allseries-ext-v1',
    crop_mm=140,source_image_size=336,model_image_size=384,extension_geometry_policy='transform_spec_v3',
    orientation_state='canonical_L_v8_3',builder_sha256=sha(RUN/'package/build_ext_bank.py'),
    contract_semantic_hash=contract['semantic_hash'],fact_packet_sha256=sha(RUN/'FACT_PACKET.json'),
    report_sha256=sha(RUN/'REPORT.md'),numeric_qa_sha256=sha(RUN/'EXT_NUMERIC_QA.json'),
    visual_qa_sha256=sha(RUN/'contact_review/VISUAL_QA.json'),metadata8_qa_sha256=sha(RUN/'METADATA8_FULL_QA.json'),
    train_manifest_sha256=sha(RUN/'TRAIN_EXT_MANIFEST.csv'),train_sample_map_sha256=sha(RUN/'TRAIN_EXT_SAMPLE_MAP.csv'),
    source_files=inputs,training_started=False,paid_compute_started=False,submission_created=False)
write('PASS_EXT_BANK.json',release)
print(json.dumps(dict(status=release['status'],train=facts['train'],gold=facts['gold'],sampler=facts['sampler']),indent=2))
