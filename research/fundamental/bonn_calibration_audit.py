"""Bonn RGB-D extrinsic audit using measured depth and published camera-marker matrix.
Candidate transforms are checked on first half and evaluated on held-out second half.
This is calibration verification, not a novel 3D representation.
"""
import argparse,json,pathlib,numpy as np
from ego_motion_warp import load,compare
M=np.array([[1.0157,.1828,-.2389,.0113],
            [.0009,-.8431,-.6413,-.0098],
            [-.3009,.6147,-.8085,.0111],
            [0,0,0,1.]])
R=np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1.]])
def rigid(A):
    Z=A.copy();u,s,v=np.linalg.svd(Z[:3,:3]);Z[:3,:3]=u@v
    return Z
M=rigid(M)
candidates={'raw':np.eye(4),'marker_to_camera':M,'camera_to_marker':np.linalg.inv(M),
            'marker_with_ros_right':M@R,'ros_then_marker':R@M,'ros_marker_inverse':np.linalg.inv(R@M)}
def transformed(frame,E):
    rgb,z,K,R0,T0=frame
    T=np.eye(4);T[:3,:3]=R0;T[:3,3]=T0
    T=T@E
    return (rgb,z,K,T[:3,:3],T[:3,3])
def score(frames):
    measures=[]
    for a,b in zip(frames[:-2],frames[2:]):
        v=compare(a,b)
        if v is not None:measures.append(v)
    if not measures:return {'n_pairs':0,'median_depth_m':None,'mean_depth_m':None,'overlap':0,'n_reprojections':0}
    return {'n_pairs':len(measures),'median_depth_m':round(float(np.median([r['warp_depth_mae_m'] for r in measures])),6),
            'mean_depth_m':round(float(np.average([r['warp_depth_mae_m'] for r in measures],weights=[r['n'] for r in measures])),6),
            'overlap':round(float(np.mean([r['overlap_fraction'] for r in measures])),5),
            'n_reprojections':sum(r['n'] for r in measures)}
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',default='results/real_rgbd/bonn_box');args=p.parse_args()
    k=[542.822841,542.576870,315.593520,237.756098]
    frames=load(args.root,k,56);split=len(frames)//2;res={}
    for name,E in candidates.items():
        f=[transformed(x,E) for x in frames]
        res[name]={'train':score(f[:split]),'heldout':score(f[split:])}
    eligible=[x for x in res if res[x]['train']['median_depth_m'] is not None and res[x]['train']['overlap']>.1]
    if not eligible:raise RuntimeError('No candidate passes reprojection coverage threshold')
    selected=min(eligible,key=lambda x:res[x]['train']['median_depth_m'])
    out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
    result={'calibration_source':'Official Bonn RGB-D page T_m and T_ROS, candidate right-extrinsic conventions',
        'selection':'minimum median training reprojection error with >10% image support',
        'selected_on_train':selected,'train_and_heldout':res,
        'limitations':'No static/dynamic pixel labels; trajectory convention may require further frame alignment; do not interpret model selection as ground truth calibration.'}
    (out/'bonn_calibration_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
