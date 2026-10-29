"""Class-agnostic geometry-assisted 3D instance proposals from dense 2D classes.

Inspired by class-agnostic 3D instance mapping (OVI-MAP). No GT physical IDs
or GT semantic class labels enter segmentation. Inputs: independently
predicted per-pixel semantic classes and measured XYZ, potentially corrected
world XYZ from RGB-D SLAM. Candidate components are separated using
connected components followed by density clustering in XYZ.

This is an EXPERIMENTAL detector proposal stage, not a guarantee of
correct physical instance segmentation or SLAM-pose invariance.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np,cv2
from scipy.ndimage import label as ndlabel
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

def _roi_instances(semantic_mask,world_xyz,*,eps=.12,subsample=3,
                   min_support=8,min_area=160,max_component_pixels=350000):
    # pixel coordinate masks should represent one class only
    if semantic_mask.ndim!=2 or world_xyz.shape[:2]!=semantic_mask.shape:
        raise ValueError("Semantic mask and measured XYZ alignment required")
    H,W=semantic_mask.shape
    structure=np.ones((3,3),np.uint8)
    components,ncomp=ndlabel(semantic_mask.astype("uint8"),structure=structure)
    results=[]
    for cid in range(1,ncomp+1):
        comp=components==cid
        area=int(comp.sum())
        if area<min_area:continue
        # Limit exceptionally large outlier components; split into tiles
        # rather than silently dropping a large multi-object region.
        boxes=cv2.boundingRect(comp.astype("uint8"))
        x,y,w,h=boxes
        crop=comp[y:y+h,x:x+w]
        # stride-3 sampled positions; local XYZ are world-coordinate meters
        subsampled=crop[::subsample,::subsample]
        samp=world_xyz[y:y+h:subsample,x:x+w:subsample]
        valid=subsampled & np.isfinite(samp).all(axis=2)
        if not valid.any():continue
        pts=samp[valid].astype("float32")
        if len(pts)<min_support:continue
        # This operation does not access any instance annotations.
        clusters=DBSCAN(eps=float(eps),min_samples=int(min_support),
                        metric="euclidean",algorithm="kd_tree").fit_predict(pts)
        candidates=np.unique(clusters)
        validlabels=[int(c) for c in candidates if c>=0 and int((clusters==c).sum())>=min_support]
        if not validlabels:continue
        # High-quality 3D points of each cluster, supports retaining
        # separate instances even if 2D semantic pixels are touching.
        samples=[];sampleid=[]
        for c in validlabels:
            cloud=pts[clusters==c]
            samples.append(cloud)
            sampleid.extend([int(c)]*len(cloud))
        seeds=np.concatenate(samples,axis=0)
        tree=cKDTree(seeds)
        fullXYZ=world_xyz[y:y+h,x:x+w]
        ys,xs=np.where(crop & np.isfinite(fullXYZ).all(axis=2))
        if not len(xs):continue
        pixelpoints=fullXYZ[ys,xs].astype("float32")
        distances,closest=tree.query(pixelpoints,k=1,workers=1)
        pixel_ids=np.full(len(xs),-1,dtype="int32")
        sampleid=np.asarray(sampleid,dtype="int32")
        accepted=distances<=max(float(eps)*1.5,.11)
        pixel_ids[accepted]=sampleid[closest[accepted]]
        for c in validlabels:
            select=pixel_ids==c
            if select.sum()<min_area:continue
            mask=np.zeros((H,W),bool)
            mask[y+ys[select],x+xs[select]]=True
            # Local spatial support prevents sending disconnected noise as
            # one RGB-D object observation.
            r,num=cv2.connectedComponents(mask.astype("uint8"),connectivity=8)
            # cv2.connectedComponents returns (num, labels)
            num_labels=r
            largest=np.bincount(num.ravel())[1:].argmax()+1 if num_labels>1 else 0
            if largest:
                mask &= (num==largest)
            if int(mask.sum())>=min_area:results.append(mask)
    return results

def instance_proposals(semantic_ids,world_xyz,class_id_to_canonical,
                       *,eps=.12,subsample=3,min_support=8,min_area=160):
    """Returns list of (canonical class, instance mask).

    Physical GT ids / NYU40 semantic GT and motion labels are absent.
    """
    if semantic_ids.ndim!=2 or world_xyz.ndim!=3 or world_xyz.shape[2]!=3:
        raise ValueError("Need (H,W) semantic IDs and (H,W,3) XYZ")
    if semantic_ids.shape!=world_xyz.shape[:2]:raise ValueError("Shapes differ")
    grouped=defaultdict(list)
    for idx,label in class_id_to_canonical.items():
        grouped[label].append(int(idx))
    predictions=[]
    for label,indices in sorted(grouped.items()):
        mask=np.isin(semantic_ids,indices)
        for proposal in _roi_instances(mask,world_xyz,eps=eps,
                  subsample=subsample,min_support=min_support,
                  min_area=min_area):
            predictions.append((label,proposal))
    return predictions
