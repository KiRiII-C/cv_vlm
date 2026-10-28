"""Read only named Hypersim files from huge official ZIP via byte-range reads."""
from __future__ import annotations
import io,zipfile
import requests

class RangeFile:
    def __init__(self,url,timeout=60):
        self.url=url;self.s=requests.Session();self.timeout=timeout;self.pos=0
        r=self.s.head(url,timeout=timeout);r.raise_for_status()
        self.size=int(r.headers['content-length'])
        if r.headers.get('Accept-Ranges','').lower()!='bytes':
            raise RuntimeError('Server does not support partial HTTP downloads')
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        if whence==0:self.pos=offset
        if whence==1:self.pos+=offset
        if whence==2:self.pos=self.size+offset
        if self.pos<0:raise ValueError('Negative seek')
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n<=0:return b''
        a=self.pos;b=self.pos+n-1
        response=self.s.get(self.url,headers={'Range':f'bytes={a}-{b}'},timeout=self.timeout)
        if response.status_code!=206:raise RuntimeError('Expected 206 Partial Content')
        data=response.content
        if len(data)!=n:raise IOError('Unexpected short remote read')
        self.pos+=len(data)
        return data
    def close(self):self.s.close()

def open_scene(scene="ai_001_001"):
    f=RangeFile(f'https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{scene}.zip')
    return f,zipfile.ZipFile(f)

def hdf5_array(ziparchive,name):
    import h5py
    with h5py.File(io.BytesIO(ziparchive.read(name)),'r') as h:
        return h['dataset'][:]

def main():
    import numpy as np
    f,z=open_scene()
    try:
        frames=[0,1,2,5,10,20]
        for i in frames:
            prefix=f'ai_001_001/images/scene_cam_00_geometry_hdf5/frame.{i:04d}.'
            pos=hdf5_array(z,prefix+'position.hdf5')
            inst=hdf5_array(z,prefix+'semantic_instance.hdf5')
            cls=hdf5_array(z,prefix+'semantic.hdf5')
            vals,count=np.unique(inst,return_counts=True)
            print('HYPERSIM_INSTANCE_FRAME',i,'shapes',pos.shape,inst.shape,cls.shape,
                  'xyz_range',np.nanpercentile(pos.reshape(-1,3),[5,50,95],axis=0).tolist(),
                  'instance_ids_count',len(vals),'top_ids',
                  sorted(zip(vals.tolist(),count.tolist()),key=lambda x:-x[1])[:12],
                  flush=True)
    finally:
        z.close();f.close()

if __name__=='__main__':main()
