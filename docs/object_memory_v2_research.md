# ObjectMemoryV2: долговременная семантическая память физических объектов

**Дата:** 28.10.2026. **Исходный код:** src/semantic_map/object_memory_v2.py;
src/semantic_map/object_surface_memory_v2.py; src/semantic_map/relocation_memory_v2.py;
src/semantic_map/visibility_evidence_v2.py.

**Статус:** разработанная и функционально протестированная подсистема. Полная
приёмка физического Re-ID = **FAIL**: на YOLOE-26 предсказанных масках точность
пока недостаточна; независимая сцена не подтвердила устойчивый выигрыш.

## Основная архитектура

RGB-D (маска YOLOE-26, глубина, калибровка, T_map_cam) -> immutable Observation ->
short-term RGB-D tracker -> ObjectMemoryV2 (3D hard gate, stable identity proposals,
Hungarian с dummy unmatched, bidirectional ambiguity/review, quality-gated prototypes)
-> SQLite с UUID, наблюдениями, семантикой, временем, ревизиями и alias audit.
Объектная поверхность сохраняется отдельно в ObjectSurfaceMemoryV2:
1.5 см цветные UUID-воксели с реальными RGB-D измерениями, число подтверждений,
счётчик опровергающих наблюдений. Удаление устаревших поверхностей допускается
только после явной проверки свободного объёма, не от отсутствия обнаружения.

Изменение графа поз: ObjectMemoryV2.correct_pose_graph() атомарно требует все
сохранённые 3D camera-frame наблюдения и все исправленные T_map_cam;
перестраивает AABB/центры без изменения исходных физических UUID и маркирует
**NEEDS_REASSOCIATION**. 3D-воксельный слой при новой траектории необходимо
перестроить из сохранённых RGB-D кадров отдельно: вызов этого метода сам по
себе НЕ перестраивает SQLite object_surface_v2.

Постоянная физическая идентичность отделена от YOLOE класса, VLM-описания
и short-term track ID. Жёсткие ограничения геометрии/класса/версии карты
никогда не преодолеваются одним лишь сходством эмбеддингов.
Неразрешённые наблюдения сохраняются в review-queue без необоснованного UUID.
RelocationMemoryV2 ведёт гипотезы смещения предмета, 3 отдельных наблюдения
новой позиции + 2 геометрически независимых RGB-D доказательства освобождения
старого пространства; UUID сохраняется только после такой проверки.
Испытания moved-object сейчас синтетические.

## Исследовательская основа (без заявления о тождественном копировании)

1. Hierarchical Multi-Prototype Appearance Memory (Electronics, 2026):
   https://www.mdpi.com/2079-9292/15/11/2357 — стабильные прототипы,
   качество записи и разделение долгой/краткой памяти.
2. OVI-MAP (CVPR 2026): https://github.com/OVI-MAP/OVI-MAP —
   class-agnostic instance geometry отдельно от semantic VLM.
3. Khronos (RSS 2024; ROS 2 version 2025):
   https://github.com/MIT-SPARK/Khronos — хронология и изменения карты.
4. ConceptGraphs: https://github.com/concept-graphs/concept-graphs —
   долговременные 3D сущности и пространственные связи.
5. ByteTrack: https://github.com/ifzhang/ByteTrack — вторичное
   сопоставление слабых детекций без создания ложных ID.
6. BoT-SORT: https://github.com/NirAharon/BoT-SORT —
   appearance + geometry/motion association.
7. Open-vocabulary perception: https://docs.ultralytics.com/models/yoloe —
   12 фиксированных текстовых классов на YOLOE-26s-seg.

## Методика: DETECTOR ≠ MEMORY

- Hypersim ai_002_001 и ai_005_001: **development** сцены.
  Цель — проверить отдельно качество ассоциации и пропуски детектора.
- Actual YOLOE-26s-seg masks, 12 заданных классов, 19 пригодных RGB-D кадров
  на сцену; исходные GT world XYZ используются для контроля координат,
  НЕ проверяется работа SLAM; GT semantic_instance_id используется только
  в оценке IDF1 и false merges, а не для сопоставления.
- Fixed-input результаты: YOLOE-26 запускается однократно, неизменные
  Observation+GT сохраняются в gzip, восемь вариантов памяти обрабатывают
  идентичные наблюдения.
- Unseen ai_017_001 выбрана **по числу целевых физических объектов до прогона
  модели**: 27 разных физических GT-ID среди 5 выборочных кадров. До теста
  V2 параметры зафиксированы; результат не использован для нового
  заявления независимой оценки после изменения гиперпараметров.

### Ground-truth 3D physical instance IDF1 на development

| Сцена | Маски | Старый IDF1 | V2 IDF1 | False merges V2 |
|---|---|---:|---:|---:|
| ai_002_001 | YOLOE-26s реальные | 0.309278 | **0.326923** | 0 |
| ai_005_001 | YOLOE-26s реальные | 0.025918 | **0.047686** | 0 |
| ai_002_001 | Эталонные | 0.909871 | **0.976190** | 0 |
| ai_005_001 | Эталонные | 0.969836 | **0.997758** | 0 |

Важное ограничение: эти цифры получены **на двух development сценах
после анализа ошибок и введения статических class-specific spatial gates**.
Они не являются независимым подтверждением универсального улучшения.
В модели настоящих масок абсолютный IDF1 4.77% на тесной сцене всё ещё FAIL.
Подтверждённый action: https://github.com/KiRiII-C/cv_vlm/actions/runs/40766015906.

### Отложенная сцена ai_017_001 (настоящий независимый тест после настройки)

| Маски | Старый IDF1 | V2 IDF1 | False merges old -> V2 |
|---|---:|---:|---|
| YOLOE-26s | **0.351779** | 0.347490 | 0 -> 0 |
| Эталонные | **0.955267** | 0.942363 | 1 -> 0 |

Тест воспроизводится:
https://github.com/KiRiII-C/cv_vlm/actions/runs/40766680560.

**Итог по независимой сцене — НЕТ доказанного улучшения IDF1.**
Новая память уменьшила число ложных объединений на oracle masks,
но чаще фрагментировала объекты. Для инженерной системы false merges
особенно опасны, но это не отменяет отрицательных IDF1 измерений.

### Синтетический стресс-тест постоянных ID

Controlled 4-seed, 6 объектов, occlusion/gap, маскированные визуальные
эмбеддинги с помехами: old mean IDF1=0.93354, V2 mean=0.94280;
geometry-only V2=0.86022. Это **синтетика**, не physical real GT benchmark.

### Профиль текущей реализации

- Подтверждение: не менее 2 наблюдений.
- Вытянутые границы 2D маски НЕ увеличивают пространственный radius
  возможного сопоставления неподвижного предмета.
- Confidence + согласованность геометрии регулируют запись новых
  образцов, до шести качественных визуальных прототипов.
- Неоднозначные ID сопоставления не разрешаются наугад.
- Сохранены object observation provenance, mutable class votes,
  timestamps, pose graph revision, review queue.
- SQL не является distributed database: одна writer-thread/процесс для
  MemoryStore без внешней синхронизации.

## Приёмочные критерии

- CI: unit, SQLite recovery, multi-UUID co-visibility, ambiguity, long-gap,
  pose-graph atomic correction, safe moved-object proof, 3D voxel evidence.
- Regression на фиксированных GT inputs: V2 IDF1 не ниже old и false merges=0
  на development scenes; 40766015906 PASS.
- **Полный physical-ID ReID:** FAIL. Нужны GT multi-scene tests,
  HOTA/AssA, detection recall/mAP и архитектура восстановления 3D
  фрагментов на отсечённых/отсутствующих масках YOLOE.
- **RTAB-Map real robot SLAM + corrected surface voxel full replay:** NOT_TESTED.
- **Robot actual NavigateToPose:** NOT_TESTED.
- **Relocated real-world objects with verified depth rays:** NOT_TESTED.

## Воспроизведение

~~~bash
# Общий RGBD e2e по умолчанию использует ObjectMemoryV2
YOLO_DEVICE=0 bash scripts/run_offline_e2e.sh

# Тесты
pytest -q tests/test_object_memory_v2.py \
  tests/test_object_surface_memory_v2.py \
  tests/test_relocation_memory_v2.py tests/test_visibility_evidence_v2.py

# Сравнение на неизменных обнаружениях Hypersim
python -m semantic_map.memory_v2_replay \
  --cache /path/to/observation_inputs_and_gt.json.gz \
  --output memory_sweep.json
~~~

**ЗАПРЕТ:** «меньше false merges» не приравнивается к
«подтверждённая устойчивость физических ID» без IDF1/AssA и GT.
