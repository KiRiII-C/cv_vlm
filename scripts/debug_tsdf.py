"""Investigate TSDF drop-off: inspect raw depth and frame-by-frame reconstructions."""
import os, cv2, numpy as np, open3d as o3d, json
from pathlib import Path
from semantic_map.tum_yolo_pipeline import download_scene,synchronized,K

data=synchronized(download_scene(os.getenv("TUM_DATA","/tmp/tum")))
samples=[0,1,20,100,250,500]
print("PAIRS",len(data),flush=True)
for n in samples:
 t,r,d,T=data[n];z=cv2.imread(str(d),cv2.IMREAD_UNCHANGED)
 valid=z[(z>500)&(z<20000)].astype(float)/5000
 print("SAMPLE",n,"depth_type",str(z.dtype),"p_valid",round(len(valid)/z.size,3),
       "range_p05_50_95",np.percentile(valid,[5,50,95]).tolist(),
       "T_translation",T[:3,3].tolist(),flush=True)
def run(upto=1,size=320,scale=5000,extrinsic_mode="inverse"):
 volume=o3d.pipelines.integration.ScalableTSDFVolume(
   voxel_length=.04,sdf_trunc=.16,
   color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
 for n in range(upto):
  _,r,d,T=data[n]; rgb=cv2.imread(str(r)); dep=cv2.imread(str(d),-1)
  cr=cv2.cvtColor(cv2.resize(rgb,(size,int(size*3/4))),cv2.COLOR_BGR2RGB)
  dr=cv2.resize(dep,(size,int(size*3/4)),interpolation=cv2.INTER_NEAREST)
  intrinsic=o3d.camera.PinholeCameraIntrinsic(size,int(size*3/4),
    K.fx*size/640,K.fy*size/640,K.cx*size/640,K.cy*size/640)
  im=o3d.geometry.RGBDImage.create_from_color_and_depth(
   o3d.geometry.Image(np.ascontiguousarray(cr)),
   o3d.geometry.Image(np.ascontiguousarray(dr)),depth_scale=scale,
   depth_trunc=4.,convert_rgb_to_intensity=False)
  pose=np.linalg.inv(T) if extrinsic_mode=="inverse" else T
  volume.integrate(im,intrinsic,pose)
 mesh=volume.extract_triangle_mesh()
 pts=volume.extract_point_cloud()
 print("EXPERIMENT",upto,scale,extrinsic_mode,"verts",len(mesh.vertices),
       "triangles",len(mesh.triangles),"points",len(pts.points),flush=True)
 if len(mesh.vertices):
  a=np.asarray(mesh.vertices)
  print("EXPERIMENT_BBOX",a.min(axis=0).tolist(),a.max(axis=0).tolist(),flush=True)
 return len(mesh.vertices)

for n in [1,10,100,596]:
 run(n)
run(100,extrinsic_mode="forward")
run(100,scale=1000)
