# Исследование объектно-ориентированной RGB-D карты с семантической памятью

## Статус доказательств (28.10.2026)

Надежная репликация научного результата требует отделения **реальной обработки датасета**,
**детерминированных синтетических испытаний** и **технически реализованных, но ещё не
проверенных гипотез**. В предыдущем разговоре пользователь запросил 100 исследований,
однако точный присланный список из 100 на данный момент недоступен в контексте и файлах;
ниже — целевой анализ первоисточников, а не якобы анализ всех 100 работ.

## Подтвержденные источники и решение для архитектуры

| Источник | Проверяемый основной подход | Следствие для проекта | Реализация и статус |
|---|---|---|---|
| [ConceptGraphs, ICRA 2024](https://concept-graphs.github.io/) | объектно-ориентированный 3D scene graph с несколькими видами и семантическими связями | объекты и отношения — главная долговременная структура | `scene_graph.py`, проверено на искусственной сцене |
| [Khronos, RSS 2024](https://github.com/MIT-SPARK/Khronos) | отделение коротких и долгих временны́х изменений | версионирование поз, история наблюдений, изменение объектов без потери identity | хранение истории и map_version реализовано, динамический SLAM нет |
| [Open3DIS, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Nguyen_Open3DIS_Open-Vocabulary_3D_Instance_Segmentation_with_2D_Mask_Guidance_CVPR_2024_paper.html) | 2D маски + геометрически согласованные 3D proposals | делать 3D фильтрацию масок, не только семантические метки | в TUM YOLO-адаптере фильтр depth outliers, 3D projection |
| [DynaMem, 2024](https://dynamem.github.io/) | динамическая память с регулярным обновлением объектов | хранить последнее свидетельство, не путать потерю видимости с исчезновением | долговременная SQLite, реальный change detection ещё не реализован |
| [FindAnything, 2025](https://arxiv.org/abs/2504.08603) | объектные локальные объемные submaps с коррекцией поз | reproject после loop closure обязательно | map_version защищает от некорректного сопоставления, reproject в будущем |
| [OVI-MAP, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Deng_OVI-MAP_Open-Vocabulary_Instance-Semantic_Mapping_CVPR_2026_paper.html) | независимые 3D instances и семантика по отобранным views | class-agnostic identity + редкие вызовы VLM | отдельные сущности и `semantic_policy.py` проверено синтетически; настоящая VLM не включена |
| [VOIM, 2026](https://arxiv.org/abs/2609.00775) | накопление мягких voxel-level evidence перед окончательным решением | отсроченное присвоение семантической метки | `voxel_votes.py` проверено с шумом меток |
| [ScanNet](https://www.scan-net.org/ScanNet/) | RGB-D, траектории, 3D instance GT | нужен основной официальный benchmark IDF1/PQ/mAP | доступ требует отдельного соглашения; не использовался |
| [ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/) | точные геометрия и instance annotations | тесты против GT на других сценах | доступ и объем >текущей CPU VM; не использовался |

## Реализованные независимые эксперименты

### A — identity association (синтетические трехмерные наблюдения)

120 временных шагов, 12 заданных физических объектов, 622 обнаружения.
Повторные посещения, пропуски, похожие классы, шум геометрии и appearance.

| Метод | IDF1 | False-split objects | False-merge tracks |
|---|---:|---:|---:|
| always-new | 0.01929 | 12 | 0 |
| geometry | 0.99839 | 0 | 0 |
| appearance | 0.91498 | 12 | 1 |
| geometry+appearance | 1.00000 | 0 | 0 |
| geometry+appearance+hold | 1.00000 | 0 | 0 |

**Вывод:** в этой простой фиксированной сцене geometry достаточно сильна; улучшение
за счет explicit ambiguity hold не обнаружено, так как тест не создал случаев
реальной неопределенности. Нельзя распространять IDF1=1.0 на реальный мир.

### B — VLM view selection (синтетическая стоимость вычислений)

800 кандидатных фотографий для 20 объектов; селектор сохраняет до 3 ракурсов:
60 из 800 (−92.5% потенциальных VLM вызовов). Это снижение **числа вызовов**, а не
доказанное ускорение реальной VLM, так как VLM не вызывалась.

### C — Semantic voting (синтетические метки)

400 вокселей, пять шумных свидетельств каждый. Точность первой метки 79%,
точность накопленных голосов — 97.25%. Это моделирование независимых ошибок,
не измерение на real-world RGB-D и не результат обучения 3D модели.

### D — Операторские команды и 2D маршруты

Работает синтетическая 2D occupancy grid с A*. Подтверждены goal selection,
обнаружение неоднозначности и отклонение недостижимых целей.
Это не запуск Nav2 и не физическая навигация.

## Протокол реальной проверки

- `tum_yolo_pipeline.py` — полный, без пропуска кадров TUM Freiburg1 Desk с эталонными позами;
  предобученная YOLO11n-seg (closed-set COCO) делает **реальную** instance segmentation,
  3D backprojection, цветовой HSV-дескриптор, ассоциацию и 3D observed-surface voxel map.
- `visualize_real.py` визуализирует фактическое облако точек и подтвержденные object proposals.
- TUM Freiburg1 Desk не содержит независимых разметок идентичности всех объектов.
  Даже успешный прогон не даст права публиковать real IDF1, false merge/split или 3D mAP.
- Обязательные следующие исследования: сегментационный GT, стабильный pose-graph,
  test-set по целым сценам, moved objects, loop closure, Nav2 in simulation, персонально обученный embedding.

## Архитектурное решение (предварительное)

RGB-D + poses (RTAB-Map) → YOLO/SAM proposals → depth-consistent 3D region →
appearance descriptor → geometry+appearance Hungarian → tentative/confirmed/unresolved →
object instance store (SQLite) + voxel semantic vote → scene graph and view selected VLM
→ natural-language goal resolver → 2D occupancy A*/Nav2 adapter.

**Не объявляется окончательно лучшей на научном уровне**, пока не выполнены реальные
сравнительные тесты с аннотированной идентичностью и сценами train/test.

### E — Реальный TUM Freiburg1 Desk, полный прогон 596 кадров

Запуск [GitHub Actions 40700640646](https://github.com/KiRiII-C/cv_vlm/actions/runs/40700640646)
завершился успешно (CPU; предобученная `yolo11n-seg.pt`). Обработаны все
596 синхронизированных пары RGB-D, имеющие эталонные позы с допуском 30 мс.

- 2541 валидное 3D наблюдение после 2D instance segmentation и фильтрации глубины;
- 27 масок без достаточной 3D опоры;
- 85 созданных треков; 54 подтверждённых; 32 трека с 10+ наблюдениями;
- 15763 3D вокселей наблюдаемых поверхностей;
- 2288 подтвержденных ассоциаций; 121 unresolved; 23 ambiguous; 23 conflict; 1 duplicate detection.

Наблюдались **дублирующиеся/фрагментированные треки** (`tv`: 13;
`mouse`: 14; `book`: 12, включая временные треки) и ошибки 2D классификации:
часть поверхности стола была размечена как `bed` в кадре 0300.
Поэтому требование «никаких дубликатов» в текущей реализации НЕ выполнено.
Нельзя называть 85 треков 85 уникальными физическими предметами.

Отдельный replay-процесс переигрывает одинаковые реальные сохранённые наблюдения
через geometry-only, appearance-only, fused и adaptive Re-ID; результаты
`real_association_ablation.json` после завершения повторного GitHub job.

### Архитектурный риск

Hungarian assignment + HSV appearance сам по себе НЕ обеспечивает устойчивый
object permanence: ошибка маски → нестабильная 3D AABB → новые временные треки.
Рекомендуемые исследования: class-agnostic proposals, 3D aggregation против
instance-oversegmentation, learned crop encoder, occupancy-negative evidence,
submap reprojection on loop closure и оценки на ScanNet/3RScan с GT IDs.
