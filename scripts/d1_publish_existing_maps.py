#!/usr/bin/env python3
"""Publish one cross-dataset D1 report only from THREE VERIFIED original RGB-D map artifacts.

No inference, training or fabricated results. Supports GitHub Actions artifact
path nesting and fails closed on duplicates, missing images, wrong dataset or
insufficient source evidence.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

DATASETS=("tum_xyz","tum_desk2","bonn_box")


def load_results(artifacts_root):
    root=Path(artifacts_root)
    results={}
    for dataset in DATASETS:
        artifact=root/("d1-map-"+dataset)
        files=list(artifact.rglob("metrics.json"))
        if len(files)!=1:
            raise RuntimeError(f"{dataset}: expected exactly one metrics JSON, found {len(files)}")
        folder=files[0].parent
        r=json.loads(files[0].read_text())
        if r.get("experiment")!="D1_REAL_MULTIDATASET_MEASURED_SEMANTIC_3D_MAPS":
            raise RuntimeError(f"{dataset}: invalid experiment provenance")
        if r.get("dataset")!=dataset:
            raise RuntimeError(f"{dataset}: source dataset mismatch")
        if r.get("GT_used_for_mapping") is not True or r.get("SLAM_evaluated") is not False:
            raise RuntimeError(f"{dataset}: oracle-vs-SLAM provenance violation")
        if r.get("independent_instance_ground_truth") is not False:
            raise RuntimeError(f"{dataset}: do not invent physical GT")
        if r.get("physical_IDF1")!="NOT_EVALUATED":
            raise RuntimeError(f"{dataset}: cannot claim object-ID score")
        n=r["input"]["synchronized_matched"]
        if n<80 or r["measured_geometry_frames"]!=(n-1)//4+1:
            raise RuntimeError(f"{dataset}: incomplete real geometry evaluation")
        if r["actual_model_inference_frames"]!=(n-1)//20+1:
            raise RuntimeError(f"{dataset}: not all planned YOLOE keyframes evaluated")
        if r["geometry_voxels"]<10000:
            raise RuntimeError(f"{dataset}: no significant measured 3D map")
        for name,min_size in [("map_overview.png",20000),("geometry_rgb.ply",1000),
                              ("semantic_predicted.ply",100),("interactive_map.html",2000)]:
            f=folder/name
            if not f.is_file() or f.stat().st_size<min_size:
                raise RuntimeError(f"{dataset}: missing or truncated {name}")
        results[dataset]=(r,folder)
    return results


def publish(results,out,source_run):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    original_url=f"https://github.com/KiRiII-C/cv_vlm/actions/runs/{source_run}"
    lines=[
      "# D1 — карты реальных RGB-D датасетов: TUM xyz, TUM desk2, Bonn moving box",
      "",
      f"**Статус:** 3/3 real-dataset map jobs PASS. [Исходный GitHub Actions run #{source_run}]({original_url}).",
      "",
      "**Ключевое ограничение:** все три карты D1 реконструированы по GT/mocap позам (oracle registration), "
      "**не являются прогонами RTAB-Map SLAM**. Отдельная RTAB-Map карта TUM desk ранее "
      "построена без GT: [B7](https://github.com/KiRiII-C/cv_vlm/actions/runs/41484368417).",
      "",
      "| Запись | Синхронные RGB-D | Кадры геометрии | Инференсы YOLOE-26x | Геометрические воксели | 3D-маски | Семантические воксели |",
      "|---|---:|---:|---:|---:|---:|---:|"
    ]
    for dataset in DATASETS:
        r,p=results[dataset]
        lines.append(f"| {dataset} | {r['input']['synchronized_matched']} | "
                     f"{r['measured_geometry_frames']} | {r['actual_model_inference_frames']} | "
                     f"{r['geometry_voxels']} | {r['valid_predicted_3D_masks']} | "
                     f"{r['predicted_semantic_voxels_per_class']} |")
        shutil.copyfile(p/"map_overview.png",out/f"D1_{dataset}_overview.png")
        (out/f"D1_{dataset}_metrics.json").write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n")
    lines.extend(["", "## Реальные цветные 3D-карты", ""])
    for dataset in DATASETS:
        r,p=results[dataset]
        items=", ".join(f"`{cls}`: {n}" for cls,n in
                        sorted(r["per_class_mask_observations"].items()))
        lines.extend([
           f"### {dataset}",
           "",
           f"![3D карта {dataset}](D1_{dataset}_overview.png)",
           "",
           f"Обнаружения: {items or 'нет среди целевых категорий'}.",
           "",
           f"Исходный файл-контейнер `d1-map-{dataset}` из "
           f"[запуска {source_run}]({original_url}) содержит "
           "`geometry_rgb.ply`, `semantic_predicted.ply`, "
           "`interactive_map.html`, `map_overview.png`, `metrics.json`.",
           "",
        ])
    lines.extend([
       "## Методология и ограничения", "",
       "Действительно использованные исходные измерения: 16UC1 depth, RGB, синхронизированные "
       "штампы времени, GT или mocap позы камеры. Геометрия: каждые 4 кадра, каждый 4-й "
       "пиксель, 3 см voxel. Семантика YOLOE-26x: каждые 20 кадров, confidence >= 0.40, "
       "2 см voxel по каждому классу.",
       "",
       "Все объекты выделены цветом **по предсказанию YOLOE**, а геометрия окрашена "
       "из оригинального RGB. Неизвестная глубина не была заполнена искусственно. "
       "Две TUM последовательности относятся к одному семейству датасета; Bonn "
       "содержит динамические объекты. Движущиеся поверхности на накопленной карте "
       "Bonn могут создавать фантомную геометрию.",
       "",
       "**НЕЛЬЗЯ считать** число 3D масок, вокселей или визуально красивую карту "
       "точностью семантической сегментации, IDF1, реальным scene-change F1, "
       "SLAM ATE, автоматической релокализацией или готовностью Nav2. Для этого "
       "нужны отдельные датасетные GT и эксперименты без GT-поз внутри SLAM.",
       "",
       "## Следующий эксперимент", "",
       "RTAB-Map без GT на каждом датасете, затем post-hoc SE(3) ATE и сравнение "
       "полученных геометрических карт с oracle; отдельно — физическая смена ID "
       "на последовательностях с размеченными объектами.",
       ""
    ])
    (out/"D1_cross_dataset_results.md").write_text("\n".join(lines))
    print("D1_VERIFIED_ALL_THREE_PUBLISHED",
          {k:{"matched":v[0]["input"]["synchronized_matched"],
               "geometry_voxels":v[0]["geometry_voxels"],
               "masks":v[0]["valid_predicted_3D_masks"]}
          for k,v in results.items()},flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts",required=True)
    p.add_argument("--source-run",required=True,type=int)
    p.add_argument("--out",default="Docs/maps")
    a=p.parse_args()
    publish(load_results(a.artifacts),a.out,a.source_run)


if __name__=="__main__":
    main()
