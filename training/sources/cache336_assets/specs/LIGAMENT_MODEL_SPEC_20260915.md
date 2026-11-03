# Локальный связочный arm v2: ACL и MCL

Версия: 2, frozen before GPU. Общий input/label/epoch-контракт задаёт `V83_BIG_TRAIN_AMENDMENT_20260915.md`.

## 1. Проверенная геометрия

Прототип и QA: `artifacts/ligament_locator_v1/` и `scripts/ligament_locator_v1.py`.

- `joint_z` берётся из действующего joint-line locator.
- `notch_x` определяется по профилю мыщелковой полосы: средняя яркость в `z ∈ [joint_z+5, joint_z+22]` по центру A-P ткани как функция координаты sagittal slice.
- Для non-FS вырезка является провалом, для FS — пиком. Полярность выбирается только по FS-тегу; правило «брать больший контраст» запрещено.
- Итоговый `notch_x` — среднее профильной оценки и середины sagittal stack. Coronal x-profile сохраняется как диагностика, но не голосует.
- На Gold58 голубая линия оказалась внутри межмыщелковой ямки 58/58: примерно 50 случаев с ошибкой <=5 мм, примерно 6 с 5–10 мм, ни одного >10 мм после усреднения.
- Типичная ширина вырезки — 22 мм; sagittal окно срезов `notch_x ±15 мм` даёт 6–11 slice.

Перед fit на population строится распределение `abs(profile_x - midpoint_x)`. Если доля >15 мм превышает 5%, добавляется третий голос `joint_gap` по глубине суставной щели и повторяется Gold/population QA. Граница не подбирается по fold3.

## 2. MCL ROI

Кожа не является якорем. MCL ROI привязан к вырезке:

- медиальная граница: `notch_x − 80 мм`;
- латеральная граница: `notch_x + 30 мм`;
- вертикаль: `joint_z − 65 мм … joint_z + 45 мм`;
- итоговый физический размер 110×110 мм, ресемпл напрямую из DICOM в 384×384.

Такой кроп покрывает медиальный надмыщелок, всю видимую MCL до метафиза, мягкие ткани и вырезку. Выход 15–20% окна за coronal FOV допускается только со стороны воздуха; причина записывается в sidecar.

## 3. Окна и головы

Water-filling без пустых токенов:

| Группа | Квота |
|---|---:|
| sagittal notch FS | 10 |
| sagittal notch non-FS | 8 |
| coronal notch | 6 |
| coronal oblique ACL, если есть | 4 |
| coronal MCL FS | 8 |
| coronal MCL non-FS | 6 |
| sagittal MCL edge | 6 |
| **Итого** | **48** |

Main heads: `ACL`, `MCL`.

Десять aux-heads из AUX60:

1. `aux__acl_complete_tear`
2. `aux__acl_partial_tear_sprain`
3. `aux__acl_mucoid_degeneration`
4. `aux__pcl_tear_sprain`
5. `aux__lcl_plc_injury`
6. `aux__mcl_grade2plus`
7. `aux__bme_medial_femoral`
8. `aux__bme_lateral_femoral`
9. `aux__bme_medial_tibial`
10. `aux__bme_lateral_tibial`

Отсутствующие source-label maskируются и не становятся нулями. Contusion считается описательно и не входит в гейт.

## 4. Модель и обучение

- `coatnet_rmlp_2_rw_384`, ImageNet initialization;
- k=12 окон train, все 48 окон eval;
- aux weight 0.25;
- schedule 24, stop 20, average 18–20;
- validation after every epoch on fold and Gold58, plus final average;
- side/pose/depth operations только `knee-laterality-v8.3`, применяются один раз до augmentation.

## 5. CPU QA

До GPU обязательны:

1. Gold58 + 40 случайных train-parent montage по всем семи группам.
2. Notch внутри sagittal/coronal ROI и длинная MCL внутри coronal ROI минимум 55/58; все отказы имеют reason code.
3. Population coverage, разрезы vendor/site, bilateral routing.
4. 200-case train smoke без fold3 и Gold, прогноз размера и peak RAM.
5. Cache identity: `orientation_state=canonical_L_v8_3`, runtime/manifest/operations SHA, новый namespace.

## 6. Fold3 gate

Rank-blend 50/50 с `AUX60_V83_F3`:

- MCL delta >= +0.006 и CI95 lower >0;
- ACL delta >= +0.003 и CI95 lower >0;
- mean ACL/MCL delta >= +0.005 и CI95 lower >0;
- mean Spearman <=0.90;
- within-site не хуже.

PASS ведёт к full-fit. FAIL закрывает input-arm без второго GPU-рецепта.
