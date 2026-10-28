"""Recover a gravity-aligned 2D occupancy map from RGB-D 3D evidence.

No lidar or IMU is presumed. The floor estimate is validated against planar
geometry and low-rank camera-motion normal. If these disagree, no navigation
ready map is declared; never silently treat unknown cells as free.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np

def estimate_floor(geom_root, min_inliers=200):
    import open3d as o3d
    geom_root=Path(geom_root)
    cloud=o3d.io.read_point_cloud(str(geom_root/"tsdf_cloud.ply"))
    if len(cloud.points)<3000:raise RuntimeError("Insufficient 3D points for floor plane")
    poses=json.loads((geom_root/"poses_used.json").read_text())["poses"]
    centers=np.array([np.array(x["T_map_cam"])[:3,3] for x in poses])
    _,_,vh=np.linalg.svd(centers-centers.mean(0),full_matrices=False)
    trajectory_normal=vh[-1]
    work=cloud
    candidates=[]
    for i in range(7):
        if len(work.points)<min_inliers:break
        equation,inliers=work.segment_plane(distance_threshold=.06,ransac_n=3,num_iterations=1400)
        if len(inliers)<min_inliers:break
        n=np.array(equation[:3],float);n/=max(np.linalg.norm(n),1e-9)
        point=-equation[3]/max(np.linalg.norm(equation[:3]),1e-9)*n
        horiz=float(abs(n@trajectory_normal))
        distances=np.abs((centers-point)@n)
        height=float(np.median(distances))
        pts=np.asarray(work.select_by_index(inliers).points)
        _,_,planar=np.linalg.svd(pts-pts.mean(0),full_matrices=False)
        xy=(pts-pts.mean(0))@planar[:2].T
        extent=np.percentile(xy,95,axis=0)-np.percentile(xy,5,axis=0)
        area=float(extent[0]*extent[1])
        c=dict(index=i,inliers=len(inliers),normal=n.tolist(),point=point.tolist(),
               trajectory_normal_alignment=horiz,camera_height_m=height,
               support_area_m2=area)
        candidates.append(c)
        work=work.select_by_index(inliers,invert=True)
    # Reject table/wall imposters; gravity is independently suggested by camera motion.
    valid=[c for c in candidates if c["trajectory_normal_alignment"]>.65
           and .70<c["camera_height_m"]<2.3
           and c["support_area_m2"]>.25 and c["inliers"]>=min_inliers]
    if not valid:
        return None,candidates
    best=max(valid,key=lambda c:(c["support_area_m2"]*c["inliers"])**.5)
    n=np.array(best["normal"],float)
    point=np.array(best["point"],float)
    if np.dot(n,centers.mean(0)-point)<0:n=-n
    # Rodrigues formula rotate observed floor normal into +Z.
    z=np.array([0,0,1.]);cross=np.cross(n,z);sn=np.linalg.norm(cross);cs=float(n@z)
    if sn<1e-8:R=np.eye(3) if cs>0 else np.diag([1.,-1.,-1.])
    else:
        axis=cross/sn
        K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        R=np.eye(3)+math.sin(math.atan2(sn,cs))*K+(1-cs)*(K@K)
    T=np.eye(4);T[:3,:3]=R;T[2,3]=-float((R@point)[2])
    return T,candidates

def project_evidence(geom_root,out_root,robot_clearance=.18):
    geom_root=Path(geom_root);out_root=Path(out_root);out_root.mkdir(parents=True,exist_ok=True)
    T,candidates=estimate_floor(geom_root)
    if T is None:
        report={"floor_detection":"FAILED","nav_ready":False,
                "candidates":candidates,"safe_goal_export":False}
        (out_root/"gravity_report.json").write_text(json.dumps(report,indent=2))
        raise RuntimeError("No independently plausible floor plane; fail closed")
    raw=np.load(geom_root/"occupancy_3d_evidence.npz",allow_pickle=False)
    keys=raw["keys"];vals=raw["logodds"]
    v=float(raw["voxel_m"])
    pts=(keys.astype(float)+.5)*v
    mapped=pts@T[:3,:3].T+T[:3,3]
    mask=(mapped[:,2]>.12)&(mapped[:,2]<1.65)&((vals>.85)|(vals<-.65))
    candidates_xy=mapped[mask,:2];codes=vals[mask]
    if len(candidates_xy)<300:raise RuntimeError("Too few known free/occupied 2D cells")
    xmin=np.floor(candidates_xy.min(0)/v).astype(int)-2
    xmax=np.ceil(candidates_xy.max(0)/v).astype(int)+2
    sz=xmax-xmin+1
    if np.any(sz>2000):raise RuntimeError("Implausibly large navigation map")
    grid=np.full((sz[1],sz[0]),-1,np.int8)
    ij=np.floor(candidates_xy/v).astype(int)-xmin
    for (col,row),value in zip(ij,codes):
        if not (0<=row<grid.shape[0] and 0<=col<grid.shape[1]):continue
        if value>.85:grid[row,col]=100
        elif value<-.65 and grid[row,col]!=100:grid[row,col]=0
    free=int(np.sum(grid==0));occupied=int(np.sum(grid==100))
    if free<25 or occupied<25:
        raise RuntimeError("Map lacks enough observed free/occupied evidence")
    np.save(out_root/"occupancy.npy",grid)
    raster=np.full(grid.shape,205,np.uint8);raster[grid==0]=254;raster[grid==100]=0
    with (out_root/"occupancy.pgm").open("wb") as f:
        f.write(f"P5\n{grid.shape[1]} {grid.shape[0]}\n255\n".encode())
        f.write(np.flipud(raster).tobytes())
    o=[float(x) for x in xmin*v]
    # Correct scalar conversion for 2D origin: x,y,theta.
    origin=[float(xmin[0]*v),float(xmin[1]*v),0.0]
    config=dict(image="occupancy.pgm",mode="trinary",resolution=v,origin=origin,
                negate=0,occupied_thresh=.65,free_thresh=.196)
    with (out_root/"occupancy.yaml").open("w") as f:
        for key,value in config.items():f.write(f"{key}: {json.dumps(value)}\n")
    from scipy.ndimage import binary_dilation
    pixels=math.ceil(robot_clearance/v)
    yy,xx=np.mgrid[-pixels:pixels+1,-pixels:pixels+1]
    blocked=grid!=0
    grown=binary_dilation(blocked,structure=xx*xx+yy*yy<=(robot_clearance/v)**2+.001)
    navigable=int(np.sum(~grown))
    status="FLOOR_ESTIMATED_3D_NAV_CANDIDATE" if navigable>10 else "INSUFFICIENT_SAFE_FREE_SPACE"
    report={"floor_detection":"PLAUSIBLE_PLANE","candidate_status":status,
            "robot_actuation":"NOT_VALIDATED","nav_ready":False,
            "gravity_aligned_for_offline_planning":True,
            "safe_goal_export":False,"T_aligned_map":T.tolist(),
            "floor_candidates":candidates,
            "origin":origin,"resolution_m":v,
            "observed_free_cells":free,"observed_occupied_cells":occupied,
            "unknown_cells":int(np.sum(grid<0)),
            "inflated_traversable_cells":navigable}
    (out_root/"gravity_report.json").write_text(json.dumps(report,indent=2))
    print("GRAVITY_2D_MAP",json.dumps({k:v for k,v in report.items() if k not in ("floor_candidates","T_aligned_map")}),flush=True)
    return report

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--geometry",default="results/full_geometry")
    ap.add_argument("--out",default="results/gravity_map")
    args=ap.parse_args()
    project_evidence(args.geometry,args.out)
