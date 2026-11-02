"""EXP-20261102-08: verify all 24 REAL training artifacts before synthesizing results.

Fails closed when any seed/arm/model run or dataset fingerprint is missing.
Never imputes results; no statistics over pretend independent rooms.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, pathlib, statistics
from itertools import product

MODELS=('mlp_plus','vit_4layer','geometry_vit')
ARMS=('xyz_256','xyz_512','mixed_256','mixed_512')
SEEDS=(0,1)


def validated(root):
    paths=list(pathlib.Path(root).rglob('metrics.json'))
    expected=set(product(MODELS, ARMS, SEEDS))
    data={}
    for path in paths:
        x=json.loads(path.read_text())
        if x.get('experiment')!='EXP-20261102-08':
            raise RuntimeError('Unexpected experiment artifact: '+str(path))
        key=(x['model'],x['arm'],x['seed'])
        if key not in expected or key in data:
            raise RuntimeError('Unknown or duplicate artifact: '+str(key))
        n=int(x['arm'].split('_')[-1])
        split={'xyz':n//2,'desk':n//2} if x['arm'].startswith('mixed') else {'xyz':n,'desk':0}
        if x['train_unique_correspondence_indices']!=n or x['train_source_counts']!=split:
            raise RuntimeError('Matched-support violation '+str(key))
        if x['epochs']!=28 or len(x['curves'])!=28:
            raise RuntimeError('Missing training epochs '+str(key))
        if x['updates_per_epoch']!=15 or x['positive_draws_per_epoch']!=960:
            raise RuntimeError('Training budget mismatch '+str(key))
        if x.get('unique_positive_index_per_minibatch') is not True:
            raise RuntimeError('In-batch duplicate positive-ID risk '+str(key))
        if x['train_hard_query_indices']!=192:
            raise RuntimeError('Hard query support mismatch '+str(key))
        if x['Bonn_blind']['n_queries']<100 or x['TUM_desk_heldout']['n_queries']<45:
            raise RuntimeError('Insufficient blind sensor evidence '+str(key))
        if not 0<=x['Bonn_blind']['rank1']<=1 or not 0<=x['TUM_desk_heldout']['rank1']<=1:
            raise RuntimeError('Invalid measured result '+str(key))
        if not path.with_name('best.pt').is_file():
            raise RuntimeError('Missing trained checkpoint '+str(key))
        data[key]=x
    if set(data)!=expected:
        raise RuntimeError('Expected 24 distinct successful model/arm/seed results. Missing: '
                           +str(sorted(expected-set(data))))
    fingerprints={tuple(sorted(v['dataset_index_sha256'].items())) for v in data.values()}
    if len(fingerprints)!=1:
        raise RuntimeError('Dataset manifests differ across jobs')
    for seed in SEEDS:
        hs={data[(m,a,seed)]['train_hard_query_sha256'] for m,a in product(MODELS,ARMS)}
        if len(hs)!=1:
            raise RuntimeError('Non-identical hard-negative sampling per seed')
        for arm in ARMS:
            support={data[(m,arm,seed)]['train_positive_index_sha256'] for m in MODELS}
            if len(support)!=1:
                raise RuntimeError('Positive sampling differs across models')
    for key in data:
        for heldout in ('Bonn_blind','TUM_desk_heldout'):
            ns={data[k][heldout]['n_queries'] for k in data}
            if len(ns)!=1:
                raise RuntimeError('Inconsistent heldout query set')
    return data


def synthesize(data, run_id=None, source_sha=None):
    lines=['# EXP-20261102-08: factorial real RGB-D evidence',
        '', '**Status:** 24/24 train checkpoints and metric JSONs verified, across 3 architectures, 4 training supports, 2 seeds.',
        '', 'Training observations: TUM fr1/xyz and fr1/desk; blind test: Bonn moving_obstructing_box.',
        'Same architecture, seed, 28 epochs, 420 optimizer steps, 960 positive draws per epoch,',
        '192 fixed xyz-only hard negatives per seed, matched 256 or 512 distinct correspondence indices.',
        'All minibatches sample 64 distinct indices without replacement (no within-batch exact duplicates).',
        'Validation: TUM xyz/desk frames 64–78; late desk and Bonn are evaluation-only.',
        'GT camera poses are **oracle labels**; no instance ID or SLAM tested.',
        '']
    if run_id:
        lines+=['**Verified GitHub Actions:** https://github.com/KiRiII-C/cv_vlm/actions/runs/'+str(run_id),'']
    if source_sha:
        lines+=['**Source commit:** '+source_sha,'']
    lines+=['| Model | Bonn xyz-256 | Bonn mixed-256 | Δ diversity (pp) | Bonn xyz-512 | Bonn mixed-512 | Δ diversity (pp) |',
            '|---|---:|---:|---:|---:|---:|---:|']
    summary={}
    for m in MODELS:
        means={}
        for arm in ARMS:
            vals=[data[(m,arm,s)]['Bonn_blind']['rank1'] for s in SEEDS]
            means[arm]=statistics.mean(vals)
        deltas={n:[100*(data[(m,'mixed_'+str(n),s)]['Bonn_blind']['rank1']-
                           data[(m,'xyz_'+str(n),s)]['Bonn_blind']['rank1']) for s in SEEDS]
                for n in (256,512)}
        summary[m]={'means':means, 'paired_differences_pp':deltas}
        lines.append('| '+m+' | '+' | '.join([
            f"{means['xyz_256']*100:.2f}%",f"{means['mixed_256']*100:.2f}%",
            f"{statistics.mean(deltas[256]):+.2f}",f"{means['xyz_512']*100:.2f}%",
            f"{means['mixed_512']*100:.2f}%",f"{statistics.mean(deltas[512]):+.2f}"])+' |')
    lines+=['', '## Paired changes by seed (percentage points)', '']
    for m in MODELS:
        q=summary[m]['paired_differences_pp']
        lines.append(f"- {m}: support=256 {q[256]}; support=512 {q[512]}.")
    lines+=['', '## Interpretation guardrails', '',
      '1. Δ diversity is the matched-number comparison; comparing xyz_256 to mixed_512 would confound number and source.',
      '2. A negative result rejects only this training protocol; a positive result does **not** establish new architectural principles.',
      '3. Two seeds, correlated sampled patches, two trajectories in the SAME TUM laboratory, and one Bonn sequence preclude a credible confidence interval for unseen buildings.',
      '4. Hard negatives always use xyz and may favor xyz-only; additional cross-domain experiments and independent buildings are required.',
      '5. Distinct correspondence indices are not proven to be spatially independent objects or locations. All Bonn metric queries are correlated in one recording.',
      '6. This experiment evaluates 5-way local retrieval only, not persistent object re-ID, scene-change F1, relocalization, navigation, or safety.',
      '']
    return '\n'.join(lines),summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--artifacts',required=True)
    p.add_argument('--out',default='research/fundamental/reports/EXP-20261102-08-results.md')
    p.add_argument('--run-id')
    p.add_argument('--source-sha')
    a=p.parse_args()
    data=validated(a.artifacts)
    report,summary=synthesize(data,a.run_id,a.source_sha)
    path=pathlib.Path(a.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(report)
    path.with_suffix('.json').write_text(json.dumps({
        'experiment':'EXP-20261102-08',
        'verified_runs':len(data),
        'run_id':a.run_id,'source_sha':a.source_sha,
        'model_summary':summary,
        'scope':'Real measured patch correspondence retrieval; Bonn is a single sequence.',
    },indent=2))
    with path.with_suffix('.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['model','arm','seed','Bonn_rank1','TUM_desk_rank1','best_epoch'])
        for k,x in sorted(data.items()):
            w.writerow([*k,x['Bonn_blind']['rank1'],x['TUM_desk_heldout']['rank1'],x['best_epoch']])
    print('EXP08 VERIFIED: 24/24 real RGB-D checkpoints; results generated',path)


if __name__=='__main__':
    main()
