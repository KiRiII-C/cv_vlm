# C4 — фактические результаты независимого геометрического multi-view аудита

**Статус:** SUCCESS. Основано на оригинальных реальных C2/C3 артефактах TUM desk2.

**Источники:** [C2](https://github.com/KiRiII-C/cv_vlm/actions/runs/41489070935); [C3](https://github.com/KiRiII-C/cv_vlm/actions/runs/41489734662).

Принятых локализаций: **25**; реальных YOLOE 3D-наблюдений: **69**; геометрически сопоставленных: **63**;
 без геометрического сопоставления: **6**.

| Консервативное решение | Количество UUID-гипотез |
|---|---:|
| `MULTIVIEW_GEOMETRY_SUPPORTED_NOT_PHYSICAL_ID` | 7 |
| `PROVISIONAL_NOT_ENOUGH_INDEPENDENT_EVIDENCE` | 5 |
| `REVIEW_DEPTH_GEOMETRY_CONTRADICTION` | 1 |
| `UNOBSERVED_OR_UNMATCHED_UNKNOWN_NOT_ABSENT` | 2 |

| UUID hypothesis | 3D matches | Independent views | Median association score | Center spread (m) | Decision |
|---|---:|---:|---:|---:|---|
| `obj_05ad20831ab4548a` | 8 | 8 | 0.975 | 0.046 | `MULTIVIEW_GEOMETRY_SUPPORTED_NOT_PHYSICAL_ID` |
| `obj_094e103be5cc5330` | 8 | 8 | 0.911 | 0.082 | `REVIEW_DEPTH_GEOMETRY_CONTRADICTION` |
| `obj_1250e9ef2d055552` | 0 | 0 | — | — | `UNOBSERVED_OR_UNMATCHED_UNKNOWN_NOT_ABSENT` |
| `obj_215cf3f244965600` | 13 | 13 | 0.955 | 0.055 | `MULTIVIEW_GEOMETRY_SUPPORTED_NOT_PHYSICAL_ID` |
| `obj_257c2292c6af5dc4` | 5 | 5 | 0.994 | 0.032 | `MULTIVIEW_GEOMETRY_SUPPORTED_NOT_PHYSICAL_ID` |
| `obj_28af14eac8665c26` | 4 | 4 | 0.966 | 0.052 | `MULTIVIEW_GEOMETRY_SUPPORTED_NOT_PHYSICAL_ID` |
| `obj_296284c613a25da4` | 1 | 1 | 0.873 | 0.000 | `PROVISIONAL_NOT_ENOUGH_INDEPENDENT_EVIDENCE` |
| `obj_35eb1d8f7639590e` | 1 | 1 | 0.852 | 0.000 | `PROVISIONAL_NOT_ENOUGH_INDEPENDENT_EVIDENCE` |
| `obj_376f54b558e85e2f` | 2 | 2 | 0.938 | 0.035 | `PROVISIONAL_NOT_ENOUGH_INDEPENDENT_EVIDENCE` |
| `obj_3a56d89e65ff5c3a` | 0 | 0 | — | — | `UNOBSERVED_OR_UNMATCHED_UNKNOWN_NOT_ABSENT` |
| `obj_486c23eecd9c5a4f` | 4 | 4 | 0.969 | 0.053 | `MULTIVIEW_GEOMETRY_SUPPORTED_NOT_PHYSICAL_ID` |
| `obj_587281b8894e5f93` | 2 | 2 | 0.930 | 0.037 | `PROVISIONAL_NOT_ENOUGH_INDEPENDENT_EVIDENCE` |
| `obj_9dc1b7edac265f61` | 1 | 1 | 0.719 | 0.000 | `PROVISIONAL_NOT_ENOUGH_INDEPENDENT_EVIDENCE` |
| `obj_aa75a25a611557a1` | 11 | 11 | 0.910 | 0.069 | `MULTIVIEW_GEOMETRY_SUPPORTED_NOT_PHYSICAL_ID` |
| `obj_f37541a19adc5e8b` | 3 | 3 | 0.862 | 0.013 | `MULTIVIEW_GEOMETRY_SUPPORTED_NOT_PHYSICAL_ID` |

**Не доказывает физическую идентичность.** Эти UUID — гипотезы по измеренной геометрии, без кросс-сессионного GT physical-ID и без appearance embedding. Удаление и перемещение объектов НЕ устанавливались; память C1 не изменялась.

**Следующий этап:** совместная оценка внешнего вида и 3D геометрии с физическим GT изменения/ID; отдельное тестирование влияния ошибочной позы и отсутствующих наблюдений.
