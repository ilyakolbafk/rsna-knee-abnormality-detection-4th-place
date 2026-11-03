# Matched R5 probe на v8.3

## Цель

Проверить, сохраняет ли RadImageNet ResNet50-224 полезное разнообразие после перехода глобального входа и manifest на v8.3. Это matched input-probe, а не замена CoAtNet локальными R5-моделями.

## Замороженный рецепт

- backbone: исторический R5 RadImageNet ResNet50, 224 px;
- schedule 48, stop 40, average epochs 38–40;
- labels: AUX60 soft-v2;
- fold: 3, группировка parent/report_group;
- input: `knee-laterality-v8.3`, `canonical_L_v8_3`, тот же глобальный банк/слоты, который соответствует R5-рецепту;
- Gold58 excluded from fit. Fold pseudo и Gold58 считаются после каждой эпохи 1–40; никаких checkpoint-решений по held-out/Gold.

Меняется только v8.3 input/manifest. Loss, optimizer, regularization, initialization и pooling остаются историческими R5.

## Историческая точка

- solo pseudo AUC 0.89240524;
- mean Spearman с legacy AUX60 0.85294220;
- rank30 delta +0.002097 с CI95 [+0.000211; +0.003878];
- five-fold OOF 4 345×60 уже собран и используется как legacy-input comparator;
- full R5 30% на other8 участвовал в подтверждённом public 0.948.

## Gate

На общем fold3 parent-grain против AUX60-v8.3:

- rank30 delta >=+0.0015;
- report-group CI95 lower >0;
- Spearman <=0.93;
- weak4 или main12 не показывает направленной материальной просадки.

PASS открывает решение о full-fit и f0/f1 для C2 с новой идентичностью. FAIL оставляет готовый legacy R5 OOF только как фиксированный comparator.

## Ограничение

Не запускать R5 на каждом 384-пиксельном compartment/anterior/ligament crop до PASS соответствующего CoAtNet input-arm: иначе одновременно меняются crop, resolution и backbone. Для label-only severity/Core5+ matched R5 допустим после PASS CoAtNet.
