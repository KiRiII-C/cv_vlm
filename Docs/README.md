# Docs — отчёты по экспериментам

Здесь хранятся воспроизводимые отчёты по экспериментам проекта RGB-D → RTAB-Map → YOLOE → динамическая 3D-карта.

## Правила ведения отчётов

- Один эксперимент — отдельный Markdown-файл в `Docs/experiments/`.
- Каждый отчёт содержит цель, гипотезы, конфигурацию среды и входные данные, код/коммит, методику, измерения, проверенные критерии, ограничения и следующие действия.
- Статусы `PASS`, `FAIL`, `RUNNING`, `NOT_TESTED` и `UNVALIDATED` отражают **фактическую** проверку, а не ожидаемые результаты.
- Не смешивать синтетический микробенчмарк с производительностью реального RTAB-Map, а вычисление ATE по GT — с использованием GT для SLAM.
- Отчёт обновляется по данным GitHub Actions (логи, JSON и артефакты); до получения результата новые показатели не придумываются.
- Артефакты CI хранятся ограниченное время, поэтому важные численные результаты и ссылки должны оставаться в отчётах.
- Изменения по исследованиям ведутся в экспериментальных ветках; перенос в основную ветку — отдельно после проверки.

## Текущие эксперименты

| Код | Эксперимент | Состояние | Отчёт |
|---|---|---|---|
| A0 | Единое организованное RGB-D-облако против повторной проекции | PASS — синтетический CPU-тест; не бенчмарк RTAB-Map | [A0](experiments/A0_shared_geometry.md) |
| A1 | Реальная RGB-D-одометрия и SLAM RTAB-Map | FUNCTIONAL_SMOKE_PASS: 98/150 валидных поз, ATE 5.58 см на подмножестве, 49 узлов и 1 глобальная петля; непрерывность не подтверждена | [A1](experiments/A1_real_tum_rtabmap.md) |
| A2 | Геометрия по измеренным RTAB-Map-позам (не GT) | CANDIDATE_UNVALIDATED: 98 кадров, 10610 вершин TSDF; два разрозненных участка и отсутствие независимой проверки точности | [A2](experiments/A2_rtabmap_geometry.md) |
| A3 | Оптимизация графа RTAB-Map против исходного | 3 REAL RUNS PASS — при loop closure ATE улучшалась на 16% и 57%, без closure улучшения нет; VO нестабильна | [A3](experiments/A3_optimized_graph.md) |
| A3-R | Повторяемость A3 и влияние замыкания циклов | 3 RUNS COMPLETE / FRONTEND REPRODUCIBILITY LIMITED / 3D ACCURACY NOT_VALIDATED | [Отчёт](experiments/A3_repeatability.md) |
| A4 | Полный TUM RGB-D без прореживания: OdomInfoLite и подтверждение каждого кадра | PASS: 595/595 уникальных RGB-D пар, RTAB-Map lost=0, технически валидных 594, ATE 6.75 см; 18 RGB без уникальной пары | [A4](experiments/A4_full_tum_odometry.md) |
| A5 | Информативность OdomInfo на полной TUM: покадровая ATE/RPE и достоверность регистрации | PASS: 595/595 обработано, 594 оценено, ATE 6.36 см, RPE 9.2 мм; OdomInfo lost не выявляет 28 ATE >10 см; переносимость не подтверждена | [Результаты A5](experiments/A5_results.md) |
| A6 | Полный RGB-D оптимизированный граф и три TSDF | FULL_DATASET_PASS: 595 пар, 594 измеренные позы, ATE RMSE 5.88 → 2.19 см; точность независимой 3D-геометрии не проверена | [A6](experiments/A6_full_graph_geometry_relocalization.md) |
| A6-L | Релокализация после рестарта со смещённым началом полной последовательности | SAME_SEQUENCE_SMOKE_PASS: 38 глобальных совпадений, 39 поз, восстановление после 47 lost-кадров; независимая hint-free релокализация NOT_VALIDATED | [A6-L](experiments/A6_full_graph_geometry_relocalization.md) |
| A7 | Независимая RGB-D релокализация: карта TUM desk, новая сессия TUM desk2 | CROSS-SEQUENCE FUNCTIONAL PASS: 595+639 пар, первое совпадение на кадре 42, 24 события с ранее сохранёнными узлами; prior-free и точность глобальной позы NOT_VALIDATED | [A7](experiments/A7_cross_sequence_relocalization.md) |
| A8 | Независимая релокализация TUM desk→desk2 при трёх начальных позах, включая +8 м/180° и отсутствие сохранённой позы | **FULL DATASET PASS**: 595-карты + 3×639 query, 25–26 успешных событий, 1.9–2.3 см расхождение между прогонами; GT-relative RPE 6–7 см; глобальная абсолютная точность NOT_VALIDATED | [A8](experiments/A8_prior_sensitivity.md) |
| B0 | Shared organized RGB-D depth / camera cloud и reprojection без двойной проекции | **FULL_REAL_TUM_TECHNICAL_PASS**: 595/595 обработано, 594 позы A6, 593 reprojection, parity errors=0 | [B0–B2](experiments/B0_B1_B2_perception_and_short_tracks.md) |
| B1 | YOLOE-26s masks → реальные 3D наблюдения → short-term IDs на полных TUM depth кадрах | **FULL_DEPTH_PIPELINE_PASS**: 595 depth, 119 YOLO keyframes, 908 масок, 863 3D наблюдения, 94 tracklets; physical ID NOT_VALIDATED | [B0–B2](experiments/B0_B1_B2_perception_and_short_tracks.md) |
| B1-X | Сравнение YOLOE-26x / 26s на одинаковых 595 depth + 119 YOLO keyframes | RUNNING — официальный YOLOE-26x CPU, отдельное измерение скорости и масок | [B0–B2](experiments/B0_B1_B2_perception_and_short_tracks.md) |
| B2 | Короткий 3D tracking по физическим GT ID на 3×100 Hypersim кадрах с настоящими YOLOE-26x предсказаниями | **GT_PHYSICAL_ID_FAIL**: pooled IDF1 13.59% 3D, 14.50% + HSV; геометрия Hypersim GT, не RTAB-Map | [B0–B2](experiments/B0_B1_B2_perception_and_short_tracks.md) |


**Полнота данных:** начиная с A4 все новые эксперименты должны обрабатывать все доступные синхронизированные RGB-D-пары выбранного датасета; лимиты вида 150 кадров и каждый третий кадр запрещены. Исторические A1–A3 не соответствовали этому требованию и не являются полными прогонами.

## Область действия

Каталог создан по требованию пользователя в точном регистре `Docs`. Уже существующий в репозитории каталог `docs` оставлен без изменений. На файловых системах без учёта регистра одновременное наличие `Docs` и `docs` может создавать проблемы при checkout; при унификации структуры потребуется отдельная миграция.
