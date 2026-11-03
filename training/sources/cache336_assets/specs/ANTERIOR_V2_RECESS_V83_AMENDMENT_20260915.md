# Поправка v8.3 к anterior-v2 / synovitis

Этот документ имеет приоритет над `ANTERIOR_V2_RECESS_20260915.md` там, где старая спецификация ссылается на прежнее правило стороны или один anterior-arm.

## Изолируемые армы

Оба арма используют `knee-laterality-v8.3`, soft-v2, один общий superset cache, CoAtNet-384 и расписание AUX60 24/20/18–20.

- **A — контроль разрешения.** Геометрия anterior-v1 без новых карманов, но правильный CoAtNet-384, 48 окон, k=12. Он отделяет эффект класса модели/разрешения от эффекта новых окон.
- **B — карманы.** Всё из A плюс axial recess 50×50 мм при 7.7 px/mm и peripheral sagittal 90×90 мм. Квоты 18/10/8/12, metadata 11.

Серия выбирается по минимальному PixelSpacing. При равном spacing tie-break детерминирован, но не по UID до качества. Тонкая 3D серия с шагом <1.5 мм подаётся нативным triplet без slab.

## Выполненный нулевой тест

Готовый anterior-v1 R5 проверен только как исторический PF-OA specialist:

- rank blend: 70% AUX60 + 30% anterior-v1;
- PF OA delta: +0.00858275;
- report-group bootstrap CI95: [+0.00202334; +0.01533798];
- 841 row / 815 group.

Это PASS для PF-OA-only fallback с весом 30%. Результат не доказывает качество crop-входа для Synovitis, потому что v1 был R5/RadImageNet-224.

## Гейты

1. Диагностический A: solo macro по PF OA/Effusion/Synovitis не ниже AUX60-v8.3 больше чем на 0.010. FAIL закрывает отдельный crop-input.
2. Членство A: rank50 mean delta main3 >=+0.004 с CI95 >0, Synovitis delta >=+0.004, within-site не хуже.
3. B над A: Synovitis delta >=+0.004 с CI95 >0 и AUC Synovitis внутри Effusion+ у B выше A.
4. Обязательно публикуются Synovitis AUC отдельно внутри Effusion+ и Effusion− на population и описательно на Gold58.
5. Gold58 не выбирает epoch/arm. Противоречие Gold и pseudo/LB открывает аудит label/site-PU, а не ручную замену checkpoint.

LB-порог ветки синовита задан заранее: >=0.86 — после crop-теста label-ветку заморозить; 0.78–0.86 — продолжить site/PU; <0.78 — site/PU становится приоритетнее wave 2.

## Cache contract

Один superset cache хранит все типы окон A и B, а manifest каждого арма выбирает подмножество. Запрещено создавать две полные копии пикселей. Namespace новый, `orientation_state=canonical_L_v8_3`, runtime/soft-v2/series-operations SHA обязательны.
