"""Lightweight view-robust mask appearance and optional pretrained CNN embedding.

Spatial HSV is a training-free baseline. CNN embeddings are ImageNet features
(not a dedicated ReID model) and MUST be evaluated on physical instance GT.
"""
import numpy as np
import cv2

def hsv_spatial(rgb,mask):
    mask=np.asarray(mask,dtype=bool)
    if rgb.shape[:2]!=mask.shape:raise ValueError("Image/mask size mismatch")
    yy,xx=np.where(mask)
    if len(xx)<50:return None
    hsv=cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV)
    x0,x1=int(xx.min()),int(xx.max())+1
    y0,y1=int(yy.min()),int(yy.max())+1
    features=[]
    # sqrt-Hellinger histogram provides more robust color matching than
    # plain raw HSV dot product. Coarse spatial quadrants retain layout.
    def hist(m,bins):
        val=cv2.calcHist([hsv],[0,1,2],m.astype(np.uint8),
                         bins,[0,180,0,256,0,256]).flatten()
        val=np.sqrt(val/max(val.sum(),1.))
        return val
    features.append(hist(mask,[8,4,4]))
    yc=(y0+y1)//2;xc=(x0+x1)//2
    for xa,xb,ya,yb in [(x0,xc,y0,yc),(xc,x1,y0,yc),
                        (x0,xc,yc,y1),(xc,x1,yc,y1)]:
        sub=np.zeros_like(mask);sub[ya:yb,xa:xb]=mask[ya:yb,xa:xb]
        features.append(hist(sub,[8,2,2]))
    emb=np.concatenate(features).astype(np.float32)
    norm=np.linalg.norm(emb)
    return (emb/norm).tolist() if norm>1e-8 else None

class AppearanceEncoder:
    def __init__(self,mode="hsv_spatial",device="cpu"):
        if mode not in ("hsv_spatial","mobilenetv3"):
            raise ValueError("mode must be hsv_spatial or mobilenetv3")
        self.mode=mode;self.device=device
        self.net=None
        if mode=="mobilenetv3":
            import torch
            from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
            # Pretrained weights are downloaded explicitly on first startup.
            model=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            self.net=model.features.eval().to(device)
            self.torch=torch
    def __call__(self,rgb,mask):
        if self.net is None:return hsv_spatial(rgb,mask)
        yy,xx=np.where(mask.astype(bool))
        if len(xx)<50:return None
        x0,x1=int(xx.min()),int(xx.max())+1
        y0,y1=int(yy.min()),int(yy.max())+1
        crop=rgb[y0:y1,x0:x1].copy()
        region=mask[y0:y1,x0:x1].astype(bool)
        crop[~region]=[114,114,114]
        crop=cv2.resize(crop,(224,224),interpolation=cv2.INTER_LINEAR)
        rgbcrop=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
        x=self.torch.from_numpy(rgbcrop).permute(2,0,1).float().div_(255.)
        x=(x-self.torch.tensor([.485,.456,.406])[:,None,None])/self.torch.tensor([.229,.224,.225])[:,None,None]
        with self.torch.inference_mode():
            out=self.net(x[None].to(self.device)).mean((-1,-2))[0].cpu().numpy()
        out=out/np.maximum(np.linalg.norm(out),1e-9)
        return out.astype(np.float32).tolist()
