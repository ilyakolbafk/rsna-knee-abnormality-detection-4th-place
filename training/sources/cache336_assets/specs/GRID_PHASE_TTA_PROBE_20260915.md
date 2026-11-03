# Grid phase TTA probe

## Вопрос

Даёт ли второй набор центров на половине страты разнообразие для готового `AUX60_V83_F3` без нового обучения, и безопасна ли phase 0.5 для Native64-v2?

## Протокол

Fold 3, те же 870 parent groups. Один checkpoint average 18–20. Два inference: P0 и P0.5. Слоты, квоты, триплеты, trim и v8.3 одинаковы.

Сначала P0 должен совпасть с saved `AUX60_V83_F3` predictions в пределах заранее записанного precision tolerance. Если parity не проходит, probe останавливается.

Считать pseudo AUC P0, P0.5, probability mean и per-target rank mean; paired bootstrap 1 000 по report_group; target Spearman; Gold58 описательно.

## Решение

- rank mean delta ≥ +0.001 и CI95 lower >0: две фазы разрешены в AUX60 kernel;
- P0.5 solo ≥ P0−0.001: v2 eval использует phase 0.5;
- иначе v2 использует phase 0, TTA закрывается.

Третьи фазы не проверяются до PASS этой пары.

