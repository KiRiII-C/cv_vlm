"""Diagnose TUM RGB-D TSDF density using one and many frames, report depth distributions."""
import json
from pathlib import Path
import numpy as np,cv2,open3d as o3d
from semantic_map.tum_yolo_pipeline import download_scene,synchronized
from semantic_map.offline_mapping import Intrinsics

def main():
    root=download_scene("data/tum"); pairs=synchronized(root)
    K=Intrinsics(517.3,516.5,318.6,255.3,640,480,depth_unit_m=1./5000.)
    subsets=[(0,1),(0,5),(0,30),(0,100),(0,596)]
    for start,end in subsets:
        for mode in ('rgbd_depth_scale_5000','explicit_float_meters'):
            volume=o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=0.04,sdf_trunc=0.12,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            single=[]
            for i,(stamp,pr,pd,T) in enumerate(pairs[start:min(end,len(pairs))]):
                bgr=cv2.imread(str(pr));raw=cv2.imread(str(pd),cv2.IMREAD_UNCHANGED)
                # sample actual measurements; do not infer validity from output mesh
                if i==0:
                    valid=raw[(raw>1000)&(raw<20000)]
                    print("PROBE",start,end,mode,"RAW_DEPTH",str(raw.dtype),
                          "percentiles",np.percentile(valid,[1,25,50,75,99]).tolist(),
                          "valid_frac",float(len(valid)/raw.size),flush=True)
                im=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                depth=raw if mode=='rgbd_depth_scale_5000' else raw.astype(np.float32)/5000.
                depth_scale=5000 if mode=='rgbd_depth_scale_5000' else 1.
                intrinsic=o3d.camera.PinholeCameraIntrinsic(
                    640,480,K.fx,K.fy,K.cx,K.cy)
                rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.ascontiguousarray(im)),
                    o3d.geometry.Image(np.ascontiguousarray(depth)),
                    depth_scale=depth_scale,depth_trunc=4.0,convert_rgb_to_intensity=False)
                if i==0:
                    z=np.asarray(rgbd.depth)
                    print("RGBD",mode,z.dtype,"nonzero_frac",float(np.mean(z>0)),
                          "mean",float(z[z>0].mean()) if (z>0).any() else 0,flush=True)
                    pc=o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,intrinsic)
                    print("SINGLE_FRAME_CLOUD_SIZE",len(pc.points),flush=True)
                volume.integrate(rgbd,intrinsic,np.linalg.inv(T))
            mesh=volume.extract_triangle_mesh()
            points=volume.extract_point_cloud()
            print("TSDF_PROBE",start,end,mode,len(mesh.vertices),len(mesh.triangles),len(points.points),flush=True)
if __name__=='__main__':main()
