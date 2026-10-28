"""Geometry generalization to unseen depth frames within the same TUM sequence.

Reconstruct from even-indexed RGB-D frames; score ONLY withheld odd-indexed
depth frames, with the same pose estimation source. The evaluator does not
claim independent scenes/3D ground truth, or SLAM performance.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,numpy as np
from scipy.spatial import cKDTree
from .tum_yolo_pipeline import download_scene,synchronized,K
from .offline_mapping import pose_override

def evaluate(dataset,map_folder,out,pose_file=None,holdout_mod=2,stride=12,max_points=180000):
    import open3d as o3d
    if holdout_mod<2:raise ValueError('Holdout modulus must be at least two')
    map_folder=Path(map_folder);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((map_folder/'mapping_report.json').read_text())
    if summary.get('integration_stride')!=holdout_mod:
        raise ValueError('Refusing apparent holdout: map must explicitly be built at matching integration stride')
    frames,source=pose_override(synchronized(download_scene(dataset)),pose_file)
    sampled=o3d.io.read_point_cloud(str(map_folder/'tsdf_cloud.ply'))
    cloud=np.asarray(sampled.points,dtype=np.float64)
    if len(cloud)<1000:raise ValueError('Map too sparse for held-out evaluation')
    kd=cKDTree(cloud)
    dists=[];valid_count=0;per_frame=[]
    for i,(ts,_,depth_name,T) in enumerate(frames):
        if i%holdout_mod!=1:continue
        d=cv2.imread(str(depth_name),cv2.IMREAD_UNCHANGED)
        if d is None or d.shape!=(480,640):raise ValueError('Missing RGB-D depth frame')
        z=d[::stride,::stride].astype(float)*K.depth_unit_m
        ys,xs=np.mgrid[0:480:stride,0:640:stride]
        valid=np.isfinite(z)&(z>.35)&(z<4.)
        if not np.any(valid):continue
        pts=np.column_stack(((xs[valid]-K.cx)*z[valid]/K.fx,
                             (ys[valid]-K.cy)*z[valid]/K.fy,z[valid]))
        world=pts@T[:3,:3].T+T[:3,3]
        dd=kd.query(world,workers=2)[0]
        dists.append(dd)
        per_frame.append({'frame_index':i,'timestamp':float(ts),'valid_sampled_depth':len(dd),
                          'median_nearest_map_m':float(np.median(dd)),
                          'coverage_at_10cm':float(np.mean(dd<.10))})
        valid_count+=len(dd)
    if not dists:raise ValueError('No withheld depth measurements')
    all_d=np.concatenate(dists)
    results={'status':'WITHHELD_FRAME_DEPTH_TO_MAP_EVALUATION',
             'reference':'withheld odd RGB-D depth frames; not independent scene GT',
             'pose_source':source,'map_integrated_frames':summary['frames_integrated'],
             'evaluated_frames':len(per_frame),'valid_evaluation_points':valid_count,
             'map_surface_points':len(cloud),
             'median_distance_m':float(np.median(all_d)),
             'mean_distance_m':float(np.mean(all_d)),
             'rmse_distance_m':float(np.sqrt(np.mean(all_d**2))),
             'coverage_5cm':float(np.mean(all_d<.05)),
             'coverage_10cm':float(np.mean(all_d<.1)),
             'coverage_20cm':float(np.mean(all_d<.2)),
             'per_frame':per_frame,
             'caveat':'Depth-to-map surface nearest-neighbor coverage is not mesh IoU, semantic ID accuracy or Nav2 safety.'}
    (out/'geometry_holdout.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    print('GEOMETRY_HOLDOUT',json.dumps({k:v for k,v in results.items() if k!='per_frame'}),flush=True)
    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='data/tum')
    p.add_argument('--map',default='results/geometry_holdout_map')
    p.add_argument('--output',default='results/geometry_holdout_eval')
    p.add_argument('--pose-json',default=None)
    args=p.parse_args()
    evaluate(args.dataset,args.map,args.output,args.pose_json)
if __name__=='__main__':main()
