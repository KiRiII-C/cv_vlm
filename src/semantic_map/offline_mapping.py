"""Offline geometric mapping, distinct from object identity/semantic mapping.

Full sequence = every synchronized RGB-D frame, provided a valid camera pose exists.
Reference poses are for validating TSDF/occupancy; they do NOT demonstrate SLAM.
A corrected pose graph always causes a fresh reconstruction, never stale voxel reuse.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import hashlib
import math

import cv2
import numpy as np
from .geometry import Intrinsics
from .tum_yolo_pipeline import download_scene, synchronized

class EvidenceGrid:
    """3D ray-carved voxel occupancy with bounded log-odds, no assumed free space."""
    def __init__(self, voxel_m=0.12):
        if voxel_m <= 0:
            raise ValueError("voxel_m must be positive")
        self.voxel_m = float(voxel_m)
        self.logodds = {}
        self.rays = 0
        self.frames = 0

    def index(self, pts):
        return tuple(int(v) for v in np.floor(np.asarray(pts) / self.voxel_m))

    def update_ray(self, start, endpoint):
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(endpoint, dtype=np.float64)
        if not (np.isfinite(start).all() and np.isfinite(end).all()):
            return
        length = np.linalg.norm(end - start)
        if length < 0.15 or length > 6:
            return
        n = max(2, int(math.ceil(length / self.voxel_m * 1.4)))
        hits = set()
        for fraction in np.linspace(0.0, 1.0, n, endpoint=False):
            key = self.index(start + fraction*(end - start))
            if key in hits:
                continue
            hits.add(key)
            self.logodds[key] = max(-5.0, self.logodds.get(key, 0.0) - 0.27)
        key = self.index(end)
        self.logodds[key] = min(5.0, self.logodds.get(key, 0.0) + 0.88)
        self.rays += 1

    def integrate(self, depth, K, T, stride=20, depth_min=.35, depth_max=4.0):
        validate_pose(T)
        if depth.ndim != 2 or depth.shape != (K.height, K.width):
            raise ValueError("Depth resolution differs from K")
        yy,xx = np.mgrid[0:K.height:stride, 0:K.width:stride]
        z = depth[::stride,::stride].astype(np.float64)*K.depth_unit_m
        valid = np.isfinite(z) & (z>=depth_min) & (z<=depth_max)
        cam = np.column_stack(((xx[valid]-K.cx)*z[valid]/K.fx,
                               (yy[valid]-K.cy)*z[valid]/K.fy,z[valid]))
        world = cam @ T[:3,:3].T + T[:3,3]
        for endpoint in world:
            self.update_ray(T[:3,3], endpoint)
        self.frames += 1

    def summary(self):
        vals = np.fromiter(self.logodds.values(), dtype=float)
        return {"frames":self.frames, "rays":self.rays,
                "free_voxels":int(np.sum(vals<-.65)),
                "occupied_voxels":int(np.sum(vals>.85)),
                "unresolved_voxels":int(np.sum((vals>=-.65)&(vals<=.85))),
                "voxel_size_m":self.voxel_m}

    def export_3d_ply(self,path):
        pts = [(x,y,z,v) for (x,y,z),v in self.logodds.items() if v>.85]
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open("w") as f:
            f.write("ply\nformat ascii 1.0\nelement vertex %d\n"%len(pts))
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for x,y,z,_ in pts:
                f.write("%.4f %.4f %.4f\n"%((x+.5)*self.voxel_m,(y+.5)*self.voxel_m,(z+.5)*self.voxel_m))
        return len(pts)

    def export_2d(self,folder,floor_z=None,min_height=.15,max_height=1.8):
        """Trinary XY projection; if floor_z is inferred it is DIAGNOSTIC, not Nav2-ready."""
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        if not self.logodds:
            raise ValueError("Cannot project empty occupancy grid")
        occupied=[key for key,val in self.logodds.items() if val>.85]
        if floor_z is None:
            if not occupied:raise ValueError("Cannot infer floor from empty observed occupancy")
            floor_z = float(np.percentile([k[2]*self.voxel_m for k in occupied],5))
            nav_ready=False
        else:
            floor_z=float(floor_z)
            nav_ready=True
        selected=[(x,y,val) for (x,y,z),val in self.logodds.items()
                  if floor_z+min_height <= (z+.5)*self.voxel_m <= floor_z+max_height]
        if not selected and not nav_ready:
            # Diagnostic projection of observed evidence; no floor was calibrated.
            selected=[(x,y,val) for (x,y,z),val in self.logodds.items()]
        if not selected:
            raise ValueError("No evidence within chosen navigation height slice")
        xmin=min(x for x,y,v in selected)-2
        xmax=max(x for x,y,v in selected)+2
        ymin=min(y for x,y,v in selected)-2
        ymax=max(y for x,y,v in selected)+2
        grid=np.full((ymax-ymin+1,xmax-xmin+1),-1,dtype=np.int8)
        for x,y,val in selected:
            # Occupied wins over any free projection; unknown otherwise.
            a,b=y-ymin,x-xmin
            if val>.85:grid[a,b]=100
            elif val<-.65 and grid[a,b]!=100:grid[a,b]=0
        raster=np.full(grid.shape,205,dtype=np.uint8)
        raster[grid==0]=254
        raster[grid==100]=0
        with (folder/"occupancy.pgm").open("wb") as f:
            f.write(f"P5\n{raster.shape[1]} {raster.shape[0]}\n255\n".encode())
            f.write(np.flipud(raster).tobytes())
        info={"image":"occupancy.pgm","mode":"trinary",
              "resolution":self.voxel_m,"origin":[xmin*self.voxel_m,ymin*self.voxel_m,0.0],
              "negate":0,"occupied_thresh":0.65,"free_thresh":0.196}
        # JSON is valid YAML 1.2; use plain YAML for ROS compatibility.
        with (folder/"occupancy.yaml").open("w") as f:
            for k,v in info.items():
                f.write(f"{k}: {json.dumps(v)}\n")
        np.save(folder/"occupancy.npy",grid)
        return {"unknown_cells":int(np.sum(grid<0)),
                "free_cells":int(np.sum(grid==0)),
                "occupied_cells":int(np.sum(grid==100)),
                "nav_ready":nav_ready,"floor_z_m":floor_z,
                "warning":None if nav_ready else "Floor Z inferred from occupied voxels; require calibrated floor before Nav2 use"}

def validate_pose(T):
    T=np.asarray(T,dtype=float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1],atol=1e-5):
        raise ValueError("Bad pose: expected finite SE3 homogeneous matrix")
    R=T[:3,:3]
    if not np.allclose(R.T@R,np.eye(3),atol=1e-3) or abs(np.linalg.det(R)-1)>1e-3:
        raise ValueError("Bad pose: rotation is not SO3")

def pose_override(pairs,path):
    """Replace all reference poses by supplied optimized camera poses (no silent fallback)."""
    if not path:
        return pairs, "TUM-reference-poses"
    data=json.loads(Path(path).read_text())
    ts=np.array([float(row["timestamp"]) for row in data["poses"]])
    if len(ts)==0:raise ValueError("Empty pose override")
    order=np.argsort(ts)
    ts=ts[order]
    poses=[np.asarray(data["poses"][int(i)]["T_map_cam"],float) for i in order]
    for p in poses:validate_pose(p)
    replaced=[]
    for t,rgb,depth,_ in pairs:
        i=int(np.searchsorted(ts,t))
        choices=[k for k in (i-1,i) if 0<=k<len(ts)]
        best=min(choices,key=lambda k:abs(ts[k]-t)) if choices else -1
        if best<0 or abs(ts[best]-t)>.03:
            raise ValueError(f"Missing optimized pose for timestamp {t}; refusing stale GT fallback")
        replaced.append((t,rgb,depth,poses[best]))
    return replaced, str(data.get("source","external-optimized-poses"))

def reconstruct(pairs,folder,voxel_m=.06,occupancy_m=.12,frame_limit=0,
                pose_file=None,floor_z=None,tsdf_side=320,rays_stride=20):
    import open3d as o3d
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    pairs,pose_source=pose_override(pairs,pose_file)
    if frame_limit>0:pairs=pairs[:frame_limit]
    if not pairs:raise ValueError("No RGB-D observations")
    K=Intrinsics(517.3,516.5,318.6,255.3,640,480,depth_unit_m=1./5000.)
    w=int(tsdf_side); h=int(480*w/640)
    intrinsic=o3d.camera.PinholeCameraIntrinsic(w,h,K.fx*w/640,K.fy*h/480,
                                               K.cx*w/640,K.cy*h/480)
    sdf=o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_m,sdf_trunc=voxel_m*4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    grid=EvidenceGrid(occupancy_m)
    digest=hashlib.sha256()
    records=[]
    for i,(t,pr,pd,T) in enumerate(pairs):
        validate_pose(T)
        color=cv2.imread(str(pr),cv2.IMREAD_COLOR)
        depth=cv2.imread(str(pd),cv2.IMREAD_UNCHANGED)
        if color is None or depth is None or color.shape[:2]!=depth.shape or depth.shape!=(480,640):
            raise ValueError(f"Invalid aligned RGB-D pair at {t}")
        digest.update(np.asarray(T,dtype=np.float64).tobytes())
        digest.update(str(t).encode())
        grid.integrate(depth,K,T,stride=rays_stride)
        cr=cv2.cvtColor(cv2.resize(color,(w,h),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        dr=cv2.resize(depth,(w,h),interpolation=cv2.INTER_NEAREST)
        image=o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(cr)),
            o3d.geometry.Image(np.ascontiguousarray(dr)),
            depth_scale=5000.,depth_trunc=4.0,convert_rgb_to_intensity=False)
        sdf.integrate(image,intrinsic,np.linalg.inv(T))
        records.append({"timestamp":float(t),"T_map_cam":np.asarray(T).tolist()})
        if (i+1)%100==0: print("MAPPING_PROGRESS",i+1,"/",len(pairs),flush=True)
    mesh=sdf.extract_triangle_mesh()
    if len(mesh.vertices)==0 or len(mesh.triangles)==0:
        raise RuntimeError("TSDF returned empty mesh: mapping has FAILED")
    o3d.io.write_triangle_mesh(str(folder/"tsdf_mesh.ply"),mesh)
    surface=sdf.extract_point_cloud()
    o3d.io.write_point_cloud(str(folder/"tsdf_cloud.ply"),surface)
    occ3d=grid.export_3d_ply(folder/"occupancy_3d.ply")
    # Retain complete signed occupancy evidence for physically aligned 2D planning.
    keys=np.asarray(list(grid.logodds.keys()),dtype=np.int32).reshape(-1,3)
    vals=np.asarray(list(grid.logodds.values()),dtype=np.float32)
    np.savez_compressed(folder/"occupancy_3d_evidence.npz",keys=keys,
                        logodds=vals,voxel_m=np.array(grid.voxel_m))
    occ2d=grid.export_2d(folder,floor_z=floor_z)
    report={
      "geometric_mapping_status":"CANDIDATE_UNVALIDATED" if len(mesh.vertices)>=5000 and len(mesh.triangles)>=7500 and len(surface.points)>=3000 and occ3d>=1000 else "FAIL_LOW_COVERAGE",
      "geometric_accuracy_status":"NOT_TESTED_AGAINST_INDEPENDENT_GEOMETRY",
      "integrity_thresholds":{"mesh_vertices_min":5000,"mesh_triangles_min":7500,"surface_points_min":3000,"occupied_voxels_min":1000},
      "pose_estimation_status":"NOT_TESTED" if pose_file is None else "EXTERNAL_INPUT_UNVALIDATED",
      "loop_closure_status":"NOT_TESTED",
      "object_identity_status":"NOT_ACCEPTED",
      "map_source":"TUM Freiburg1 Desk",
      "camera_pose_source":pose_source,
      "pose_graph_digest_sha256":digest.hexdigest(),
      "frames_integrated":len(records),"frames_requested":len(pairs),
      "voxel_m":voxel_m,"occupancy_voxel_m":occupancy_m,
      "mesh_vertices":len(mesh.vertices),"mesh_triangles":len(mesh.triangles),
      "surface_points":len(surface.points),"occupied_3d_voxels":occ3d,
      "occupancy_evidence":grid.summary(),"occupancy_2d":occ2d,
      "accuracy_3d_against_independent_gt":"NOT_EVALUATED",
      "pose_accuracy_ate_rpe":"NOT_EVALUATED",
      "navigation_integration":"NOT_TESTED",
      "map_reconstruction_rule":"Rebuild entirely on any updated pose file; never reuse old TSDF or 3D observations",
      "methodology":"Full-frame RGB-D TSDF fusion using externally supplied camera poses, not SLAM"}
    (folder/"mapping_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (folder/"poses_used.json").write_text(json.dumps({"poses":records,"source":pose_source}))
    print("MAPPING_RESULT",json.dumps({k:v for k,v in report.items() if k!="occupancy_evidence"}),flush=True)
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",default="data/tum")
    parser.add_argument("--output",default="results/full_geometry")
    parser.add_argument("--pose-json",default=None)
    parser.add_argument("--floor-z",type=float,default=None)
    parser.add_argument("--max-frames",type=int,default=0)
    parser.add_argument("--voxel",type=float,default=.06)
    parser.add_argument("--occupancy-voxel",type=float,default=.12)
    parser.add_argument("--tsdf-side",type=int,default=320)
    parser.add_argument("--ray-stride",type=int,default=20)
    args=parser.parse_args()
    pairs=synchronized(download_scene(args.dataset))
    reconstruct(pairs,args.output,voxel_m=args.voxel,occupancy_m=args.occupancy_voxel,
                frame_limit=args.max_frames,pose_file=args.pose_json,floor_z=args.floor_z,
                tsdf_side=args.tsdf_side,rays_stride=args.ray_stride)

if __name__=="__main__":main()
