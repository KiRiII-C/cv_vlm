"""Ablations of semantic voting and selective VLM view scheduling; synthetic labels only."""
from pathlib import Path
import json, math
import numpy as np
from .semantic_policy import View,choose_views
from .voxel_votes import VoxelSemanticMemory

def run(output='results/semantic_ablation',seed=71):
    rng=np.random.default_rng(seed)
    # Independent per-voxel categorical measurements with controlled 24% label noise.
    nvox=400;nobs=5;nclasses=6;p_wrong=.24
    true=rng.integers(0,nclasses,nvox)
    pred=rng.integers(0,nclasses,(nvox,nobs))
    coin=rng.random((nvox,nobs))>p_wrong
    pred=np.where(coin,true[:,None],pred)
    bas=np.mean(pred[:,0]==true)
    voting=VoxelSemanticMemory(min_views=2)
    for i in range(nvox):
        for j in range(nobs):voting.observe((i,0,0),str(pred[i,j]),.9,f'{i}:{j}')
    guess=[voting.query((i,0,0)) for i in range(nvox)]
    acc=np.mean([g['label']==str(true[i]) for i,g in enumerate(guess)])
    # Simulated repeated observations for expensive VLM view budgeting.
    views=[View(f'obj{i}',f'f{i}_{j}',float(rng.uniform(.4,.99)),float(rng.uniform(-math.pi,math.pi)),float(j))
           for i in range(20) for j in range(40)]
    picked=choose_views(views,max_views=3)
    selected_quality=np.mean([v.quality for v in picked])
    result={'provenance':'synthetic class noise and synthetic view qualities, NOT real VLM scores',
            'seed':seed,'voxels':nvox,'observations_per_voxel':nobs,
            'single_view_accuracy':float(bas),'five_view_vote_accuracy':float(acc),
            'vlm_queries_without_selection':len(views),'vlm_queries_after_selection':len(picked),
            'reduction_vlm_queries_pct':round(100*(1-len(picked)/len(views)),2),
            'selected_mean_quality':float(selected_quality)}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'ablations.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('SEMANTIC_ABLATIONS',json.dumps(result,ensure_ascii=False))
    return result
if __name__=='__main__':run()
