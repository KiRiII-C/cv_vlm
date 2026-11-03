# Почему OVI-MAP выглядит как цельная комната, а D1/B7 — облака точек?

**Дата анализа:** 2026-11-03. Исходники и статья:
- OVI-MAP (CVPR 2026): https://github.com/OVI-MAP/OVI-MAP
- Official paper: https://openaccess.thecvf.com/content/CVPR2026/papers/Deng_OVI-MAP_Open-Vocabulary_Instance-Semantic_Mapping_CVPR_2026_paper.pdf
- Heat-map mesh renderer: https://github.com/OVI-MAP/OVI-MAP/blob/main/scripts/visualizations/search_heatmap.py
- Vertex/GT-mesh post-processing: https://github.com/OVI-MAP/OVI-MAP/blob/main/scripts/utils/mesh_postprocess_utils.py
- Triangle TSDF integration: https://github.com/OVI-MAP/OVI-MAP/blob/main/mapping_ros_ws/src/consistent_panoptic_mapping/global_segment_map/include/global_segment_map/meshing/label_tsdf_mesh_integrator.h
- Current D1 scatter renderer: ../../scripts/d1_multidataset_measured_maps.py
- Current B6/B7 scatter renderer: ../../scripts/b6_render_dense_semantic_map.py

## Проверенная разница в представлении

| Свойство | OVI-MAP | Наши карты D1 / B7 |
|---|---|---|
| Источник RGB-D | Replica, ScanNet; known camera poses in published experiments | Real TUM desk/xyz, Bonn; known GT poses for D1, actual optimized RTAB-Map poses for B7 |
| Трехмерная поверхность | Incremental TSDF + mesh triangulation (Voxblox++ label TSDF) | D1 RGB geometric point voxels 3 cm; B7 A6 TSDF exists, but renderer displays **points** rather than surface triangles |
| Плотность визуализации | Continuous triangle faces with lighting and interpolated TSDF zero-crossing geometry | D1 only one of 16 image pixels every 4 matched frames for geometry, point scatter alpha .25-.45, 3 cm voxels; YOLO every 20 frames, cap 3000 measured mask points |
| Физические экземпляры | Class-agnostic entity proposals, depth refinement, 3D association & persistent instance labels embedded in TSDF voxel map | Predicted categories and short/persistent UUID hypotheses, not trustworthy per-voxel instance mesh labels |
| Семантика | SigLIP feature vectors per 3D instance, view selection, cosine with text for open-set query | 12 YOLOE-26x prompt classes, category colors and YOLO confidence; no equivalent per-instance text embedding in D1 |
| Тепловая карта | Per-instance cosine similarity mapped onto the **triangle mesh** | No meaningful object-level semantic heatmap can be constructed without object-language embeddings |

**Одна конкретная причина, видимая в существующем `D1_tum_desk2_overview.png`:** доминирование большой белой 3D области осей и дальней перспективы, непрозрачного светлого canvas и мелкого маркера точек; RGB-контекст очень прозрачный, видны только цветные фрагменты. Это проблема рендеринга сверх исходной разреженности.

**Важный нюанс сопоставления со статьей:** в официальном `scripts/visualizations/search_heatmap.py` вход heat-map называется `instance_map_gt_*.ply`, т.е. это **предсказанные метки, перенесённые на вершины GT-меша для оценки**, а не обязательно сырой reconstructed mesh. В `scripts/utils/mesh_postprocess_utils.py` реализован kNN-перенос вершин предсказанного меша на вершины GT mesh. Прямое сравнение такого heatmap изображения с нашим неструктурированным облаком несправедливо. OVI также умеет строить собственный mesh; это не означает, что вся красивая поверхность в демонстрациях измерена непосредственно датчиком.

## Что действительно требуется для сопоставимого визуального качества

1. **Плотный TSDF mesh с реальным RGB** и освещением вместо point scatter; сохранять отдельный полный PLY/GLB и браузерный mesh.
2. **Раздельные слои**: естественный RGB, нейтральная неразмеченная геометрия, семантическая раскраска предсказанных классов и, только после проверенного ID-мэтчинга, отдельный instance layer.
3. **Корректная камера**: кадрировать вид по семантически наблюдаемой области, подавить 3D-сетку и белый фон, дать orbit/zoom/hide layers. Это не влияет на научные метрики.
4. **Геометрическая и семантическая интеграция должны оставаться разными.** Если объект не распознан, он не исчезает из RGB/TSDF; не распространять класс за пределы измеренной поверхности.
5. После D2 реализовать class-agnostic spatial instance voting/association на TSDF/суперточках с неопределённостью, отдельный UUID mesh и per-instance feature query.
6. Для прямого scientific сравнения прогнать **одну и ту же Replica office0 и ScanNet сцену** с идентичными позами, глубиной и разрешением и отдельно сравнить качества геометрии / instance / query. Не переносить цифры со сравнительных картинок разных датасетов.

## Экспериментальный ответ: D2

Код: `scripts/d2_ovi_style_tsdf_mesh.py`. Работающее решение должно использовать raw RGB-D из тех же трёх D1 датасетов, GT/mocap poses (честно обозначено как oracle), Open3D Scalable TSDF с voxel 2 cm и truncation 8 cm, а затем спроецировать **исключительно близкие, семантически согласованные YOLOE 3D маски D1** на Mesh вершины. Ненаблюдаемые или противоречивые поверхности остаются нейтрально серыми. Выход — full 3D triangle PLY/GLB, два вида PNG, 3D HTML с переключением RGB/семантика.

Это **визуальное и геометрическое приближение**, а не копия OVI-MAP, не новые физические IDs, не VLM heatmap и не новые SLAM эксперименты. TSDF треугольники являются интерполированными между реально измеренными depth лучами. Качественные изображения нельзя считать доказательством метрической точности: нужны цифровые метрики и GT.
