# Amendment: v8.3 для большого цикла 2026-09-15

Этот документ имеет приоритет над ссылками на старую ориентацию в:

- `COMPARTMENT_MODEL_SPEC_20260913.md`;
- `ANTERIOR_MODEL_SPEC_20260913.md`;
- `SEVERITY_LABEL_ARM_SPEC_20260914.md`;
- `NATIVE64_V2_INPUT_SPEC_20260915.md`;
- `LIGAMENT_MODEL_SPEC_20260915.md`.
- `ANTERIOR_V2_RECESS_V83_AMENDMENT_20260915.md`;
- `R5_V83_MATCHED_PROBE_20260915.md`.

## Замороженный общий вход

- policy: `configs/knee_orientation_policy.json`, id `knee-laterality-v8.3`;
- training manifest: `artifacts/production/laterality_v8_3/TRAINING_MANIFEST_SOFT_V2.json`;
- grain: `SampleID = ParentStudyInstanceUID × knee`;
- fold/group: parent и `report_group` неделимы;
- canonical side: L, policy применяется ровно один раз до augmentation;
- bilateral labels: soft-v2; 75 explicit P/N сохраняются, 69 слабых клеток confidence 0.25, 24 S06 masked;
- Gold58: excluded from training;
- new cache namespace required; старый canonical cache не переиспользуется как новый.

## Matched baseline

До сравнения новых веток обучить `AUX60_V83_F3`. Native64-v2 меняет только input относительно этой базы. Severity меняет только label policy относительно этой базы. Старый `AUX60_f3_REFERENCE` остаётся историческим контролем, но не является парной базой для этих двух решений.

## Epoch и evaluation policy

Расписание 24, стоп 20, среднее весов 18–20. После каждой эпохи считать fold pseudo и Gold58; на full-fit — Gold58. Финально считать average. Метрики не выбирают checkpoint. Held-out не выбирает эпоху. Gold не выбирает модель.

## Локальные модели

Compartment, anterior и ligament строят high-resolution crop непосредственно из DICOM после разрешения knee/series operations v8.3. Нельзя вырезать их из уже сжатого Native64 кэша.

Anterior A/B обязаны читать один superset cache. Ligament использует notch-profile с FS-полярностью и MCL ROI от notch, как зафиксировано в v2-спецификации. R5-v8.3 является отдельным matched global probe и сохраняет собственное расписание 48/40/38–40.

## Обязательная идентичность

Каждый run identity содержит SHA:

- orientation release manifest;
- soft-v2 training manifest;
- cache manifest;
- labels and mask;
- fold assignment;
- code bundle;
- initialization checkpoint;
- fixed epoch rule.

Любое несовпадение — FAIL до GPU fit.
