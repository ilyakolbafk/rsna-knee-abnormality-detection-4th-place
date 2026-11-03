# Native64-v2: core-модель на полном банке

## Контроль

Контроль — `AUX60_V83_F3`, описанный в `V83_BIG_TRAIN_AMENDMENT_20260915.md`. Относительно него v2 меняет только представление входа. Backbone, 60 голов, soft-v2 labels/masks, folds, seed, optimizer, augmentation, k=12 train, 24/20/18–20 остаются одинаковыми.

## Банк

Все декодируемые срезы всех разрешённых v8.3 серий в 336 u8. На кадр сохраняются series UID, plane/slot, physical coordinate along normal, spacing, PixelSpacing, FS и причина выбора/исключения. Оценка 85–100 ГБ.

Восемь слотов: пять AUX60 плюс `axial_structural`, `coronal_oblique`, `sagittal_fluid_2`. Серия внутри слота выбирается минимальным PixelSpacing; UID используется только как детерминированный tie-break при равном качестве.

## Sampling

- ≤5 непустых слотов: 64 eval tokens;
- каждый следующий слот: +8, cap 80;
- train всё равно выбирает 12 окон;
- train centers: deterministic `central_stratified(seed, epoch, uid)`;
- eval: central grid phase 0.5 только если phase probe прошёл, иначе phase 0.

## Slab

Целевой физический шаг 3.4 мм. Для series step `g`: `k=max(1, round(3.4/g))`. Каждый канал — среднее соседних срезов на толщине около 3.4 мм; три канала имеют центры `c-k, c, c+k`. При `g>=2.5` поведение эквивалентно обычному ±1. Missing step: IPP, затем SliceThickness, иначе статус missing и обычный ±1.

## Metadata

К прежним шести признакам добавить one-hot восьми slot types и `g/5`. Новая часть projection инициализируется нулём. Точная размерность и state dict фиксируются smoke-тестом до main fit.

## Gate fold 3

- solo ≥ `AUX60_V83_F3 - 0.001`;
- rank50 blend delta ≥ +0.003;
- paired bootstrap CI95 lower > 0 по report_group;
- mean target Spearman ≤0.93;
- within-site не хуже;
- читается только average 18–20.

PASS ведёт к full-fit. FAIL ведёт к eval-only attribution: mask новых слотов, отключение slab, phase 0. Второй GPU fit без диагноза не запускается.

