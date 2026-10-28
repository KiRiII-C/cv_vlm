# YOLOE-26 + temporal RGB-D tracking + persistent 3D Re-ID (28.10.2026)

## Принятая экспериментальная постановка

Ограниченный перечень **12 текстовых категорий**, файл
config/yoloe26_classes.json:
chair, table, sofa, cabinet, bookshelf, bed, laptop, computer monitor,
keyboard, mouse, cup, potted plant.

Промпты фиксируются при старте модели **YOLOE-26s-seg.pt**.
Используются instance segmentation masks, НЕ коробки без масок.
Выбор checkpoint s (а не x) — воспроизводимый CPU-бенчмарк
и разумная стартовая точка для ноутбука RTX 4060 8 GB, НЕ заключение
о лучшей архитектуре. x/m проверяются отдельно на доступном GPU.

## 2-уровневая ассоциация

1. YOLOE-26 с заранее скомпилированными text embeddings (первый
   set_classes требует скачать MobileCLIP); NMS class-agnostic снижает
   межклассовые дубликаты. Результат: текстовый класс, confidence, mask.
2. Калиброванная depth-backprojection внутри маски после фильтрации
   глубины; формируются метрический 3D центр, робастный AABB и цветовые
   признаки (по умолчанию маскированный spatial-HSV; возможно выбрать
   ImageNet MobileNetV3 embedding). Исходные маски не заменяются bbox.
3. TemporalTracker: по одному кадру, Hungarian глобальный
   one-to-one; сначала высокие confidence, затем низкие confidence
   только для обновления существующего трека — при пропадании маски
   трек хранится не более 40 кадров, без выдумывания синтетических масок.
   Матрица совместимости: жёсткий класс и расстояние в 3D, затем
   3D proximity, 2D box IoU, mask IoU, appearance cosine.
   При двух почти одинаковых кандидатах вместо случайного ID —
   состояние AMBIGUOUS. Этот short ID — только временный.
4. SemanticMapper: постоянные мировые UUID в SQLite, глобальный
   one-to-one Hungarian и hard map_version, class, geometry gates.
   Short ID становится только +0.055 soft hint, если глобальное
   геометрическое сопоставление уже правдоподобно.
   Неизвестные/конфликтующие совпадения не создают автоматически
   подтверждённые объекты.
5. Multi-view gallery: до восьми неодинаковых признаковых представлений
   экземпляра в SQLite, а не только средняя цветовая гистограмма.
   Gallery улучшает score лишь ПОСЛЕ hard spatial gate; изменение
   класса и pose-map-version не может быть переопределено признаками.
6. Для длительного отсутствия (более 40 кадров) временный трек
   отбрасывается, повторная привязка осуществляется только по мировой
   3D карте с постоянными UUID.
7. Dense instance pointcloud: каждая точка получает детерминированный
   RGB цвет, зависящий от UUID, шаг 1.5см, source provenance.

## Какие исследования использованы как идеи

- YOLOE-26, официальный Ultralytics:
  https://docs.ultralytics.com/models/yoloe
  Применены text prompting/instance masks/agnostic NMS.
- ByteTrack: https://github.com/ifzhang/ByteTrack
  Идея second-stage low confidence rescue.
- BoT-SORT: https://github.com/NirAharon/BoT-SORT
  Идея appearance+IoU, сохранение short tracks при пропусках.
- Ultralytics Tracking (ByteTrack, BoT-SORT, Deep OC-SORT, TrackTrack):
  https://docs.ultralytics.com/modes/track
  Проведено сравнение принципов, не скопированы обученные ReID веса.
- TRACKGRAPH (2026): https://arxiv.org/abs/2609.31005
  Разделение быстрых 2D short-term и persistent 3D long-term identity.
- OVI-MAP (CVPR 2026): https://ovi-map.github.io/
  Отделение физической идентичности от поздней VLM семантики.
- ConceptGraphs: https://github.com/concept-graphs/concept-graphs
  Multi-view 3D ассоциация и долговременная память.
- Khronos: https://github.com/MIT-SPARK/Khronos
  Временное/пространственное разделение краткосрочных и длительных изменений.
- TrackEverything: https://trackeverything.github.io/
  3D мировая система координат и подавление повторных экземпляров.

Реализован **собственный** двухслойный модуль на основе этих принципов,
а не дословная копия одной библиотеки. Настоящее OCR/optical flow
mask propagation и supervised 3D embedding НЕ реализованы.

## Численные результаты, которые МОЖНО утверждать

- GitHub Actions 40759845090: YOLOE-26s-seg, 12 classes, реальный
  TUM fr1_desk, 40 RGB-D кадров, 330 пригодных масок/наблюдений,
  20 confirmed треков, 15 538 цветных 3D точек.
  Классы не являются GT physical instance ID.
- GitHub Actions 40760362381: тест Hypersim ai_001_001
  с физическим GT: только **один** GT объект относится к выбранному
  подмножеству 12 классов (4 его появления), поэтому он НЕ является
  достоверным независимым тестом качества Re-ID.
  Общесценовый IDF1 (все 54 GT-ID, включая исключённые категории):
  0.0423 (YOLOE fused), 0.0352 (temporal), 0.0352 (temporal multiview).
  Target-only IDF1=0.25 при одном объекте. Улучшения НЕ доказаны.
- Выбран дополнительный benchmark с большим числом реальных
  физически размеченных Hypersim целевых ID: ai_002_001 (различные
  chair/table/sofa/tv), ai_005_001 (плотная группа похожих стульев).
  Сбор GT-ID не используется в inference, только в метриках.
  Проверки: IDF1, IDFP/IDFN, false splits/merges, recovered IDs после
  выхода из кадра, раздельные прогоны оракул-масок и реальных YOLOE.

## Как запускать

~~~
python -m pip install -e '.[test]' 'ultralytics>=8.4.12'
python -m semantic_map.tum_yolo_pipeline \
  --data data/tum --output results/yoloe26 \
  --model yoloe-26s-seg.pt \
  --classes-json config/yoloe26_classes.json --conf .23 \
  --device cpu --appearance hsv_spatial
~~~

Для CUDA: --device 0. Для отдельных тестов можно указать
--appearance mobilenetv3 (pretrained ImageNet, **не** обученная ReID),
--disable-temporal-tracking (абляция), --model yoloe-26x-seg.pt
(при достаточной памяти). Установка torch+torchvision и текстового
CLIP encoder требуется до offline-деплоя. Профиль текста можно
экспортировать через YOLOE26Segmenter.save_prompt_embeddings,
но загрузка из NPZ при старте пока настраивается вручную.

Фактический «успешный CI» обозначает корректный запуск
пайплайна, а не автоматически успешную физическую идентификацию.
Объектные false merges при неоднозначности критичнее false split;
выпуск без независимой GT-оценки заблокирован
config/acceptance.json.


## Проверка после независимой GT-абляции (Hypersim, 28.10.2026)

[Логи восьми завершённых реальных запусков](https://github.com/KiRiII-C/cv_vlm/actions/runs/40761795172).
Использованы настоящие изображения, предсказанные маски YOLOE-26s и GT
физических IDs. 3D XYZ получены из GT для изоляции ошибки идентификации
от ошибки SLAM; это **не** real-robot performance. При подсчёте target IDF1
сравнивались только физические экземпляры, семантика которых соответствует
выбранным 12 категориям; у oracle внецелевые маски исключены из
false-positive штрафа.

| Сцена | Конфигурация | Target IDF1 ↑ | False splits (все GT) | False merges (все GT) |
|---|---|---:|---:|---:|
| ai_002_001, разнообразная мебель | YOLOE + global 3D fused baseline | **0.30928** | 1 | 0 |
| ai_002_001 | YOLOE + temporal + multiview HSV (hardened) | 0.27692 | 1 | 0 |
| ai_002_001 | YOLOE + temporal + pretrained CNN gallery | 0.26733 | 3 | 0 |
| ai_002_001 | GT masks + 3D association upper-control | 0.90987 | 3 | 0 |
| ai_005_001, тесная сцена с похожими стульями | YOLOE + global 3D fused | 0.02592 | 0 | 0 |
| ai_005_001 | YOLOE + temporal + multiview HSV | 0.02863 | 0 | 1 |
| ai_005_001 | YOLOE + temporal + CNN gallery | **0.03015** | 0 | 0 |
| ai_005_001 | GT masks + 3D association upper-control | 0.96984 | 17 | 0 |

**Обязательный отрицательный вывод**: дополнительное слежение и галерея
признаков НЕ улучшили target IDF1 на разнообразной сцене. В многолюдной
(много похожих стульев) CNN дала небольшой рост target IDF1 относительно
baseline и не допустила GT false merges, но абсолютный IDF1 слишком низок.
Механизм пока **не принят как стабильный Re-ID**. Улучшение нельзя
переносить между сценами без независимой повторной проверки.

Приоритетная геометрическая надёжность: временный трек не обходит
3D hard-gate, конфликт классов и версий карты; при неопределённости
система не подтверждает идентичность. CNN descriptor является
ImageNet classification embedding, **не обученной ReID сетью**.

Сохранение физической идентичности при самостоятельном движении робота
требует оценённых без GT поз, корректировки всей истории после loop
closure и расширенной проверки на ScanNet++/ARKitScenes с независимыми
физическими instance IDs. Ни одной из этих проверок пока нет.
