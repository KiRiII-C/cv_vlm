"""EXP-20261102-08: 2x2 scene-diversity x unique-correspondence study on real RGB-D.

Controls number of distinct matched-pair indices, optimization budget, hard-negative
training support, seeds and architecture. Uses GT camera poses only as an oracle
for correspondence labels. Bonn is never used in training or checkpoint selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import random

import numpy as np
import torch
import torch.nn.functional as F

from real_neural_contrastive import sequence, pairs
from strong_encoder_training import StrongEncoder, augment, rank5, bootstrap, to_data
from hard_negative_retrieval import corrected_frames, create_queries

BASE_SEED = 20261102
ARMS = ('xyz_256', 'xyz_512', 'mixed_256', 'mixed_512')
MODELS = ('mlp_plus', 'vit_4layer', 'geometry_vit')
TUM_K = [517.3, 516.5, 318.6, 255.3]
BONN_K = [542.822841, 542.576870, 315.593520, 237.756098]


def matched_indices(n_xyz: int, n_desk: int, arm: str, seed: int):
    """Draw deterministic nested, no-replacement subsets of physical-match indices."""
    if arm not in ARMS:
        raise ValueError('Unknown arm: ' + arm)
    n = int(arm.rsplit('_', 1)[-1])
    mixed = arm.startswith('mixed_')
    nx, nd = (n // 2, n // 2) if mixed else (n, 0)
    if n_xyz < nx or n_desk < nd:
        raise ValueError(f'Not enough real correspondences: {n_xyz=}, {n_desk=}, {nx=}, {nd=}')
    ia = np.random.default_rng(seed + 101).permutation(n_xyz)[:nx]
    ib = np.random.default_rng(seed + 102).permutation(n_desk)[:nd]
    assert len(np.unique(ia)) == nx and len(np.unique(ib)) == nd
    return ia, ib


def fingerprint(root):
    h = hashlib.sha256()
    for name in ('rgb.txt', 'depth.txt', 'groundtruth.txt'):
        h.update((pathlib.Path(root) / name).read_bytes())
    return h.hexdigest()


def sample_positives(xyz, desk, arm, seed):
    xa, ya = xyz
    xb, yb = desk
    ia, ib = matched_indices(len(xa), len(xb), arm, seed)
    X = torch.cat([xa[ia], xb[ib]], 0) if len(ib) else xa[ia]
    Y = torch.cat([ya[ia], yb[ib]], 0) if len(ib) else ya[ia]
    n = int(arm.rsplit('_', 1)[-1])
    assert len(X) == len(Y) == n
    digest = hashlib.sha256(ia.tobytes() + b'|' + ib.tobytes()).hexdigest()
    return X, Y, {'xyz': len(ia), 'desk': len(ib)}, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xyz-root', required=True)
    p.add_argument('--desk-root', required=True)
    p.add_argument('--bonn-root', required=True)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=28)
    p.add_argument('--out', default='results/real_rgbd/matched_unique')
    args = p.parse_args()
    if args.seed < 0 or args.epochs < 16:
        raise ValueError('Require nonnegative seed and at least 16 epochs')
    seed = BASE_SEED + args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)

    train = {}
    for root, name in ((args.xyz_root, 'xyz'), (args.desk_root, 'desk')):
        frames = sequence(root)
        x, y, times = pairs(frames)
        mask = times < 54
        train[name] = (x[mask], y[mask])
        print('REAL_TRAIN', name, 'frames', len(frames), 'pairs', len(train[name][0]), flush=True)
    X, Y, counts, index_hash = sample_positives(train['xyz'], train['desk'], args.arm, seed)

    fx, _ = corrected_frames(args.xyz_root, TUM_K, 'tum_xyz', 100)
    fd, _ = corrected_frames(args.desk_root, TUM_K, 'tum_desk', 100)
    fb, _ = corrected_frames(args.bonn_root, BONN_K, 'bonn_box', 84)
    # IDENTICAL xyz-only hard-negative support across all four arms.
    H = to_data(create_queries(fx[:54]))
    if len(H[0]) < 192:
        raise RuntimeError('Need at least 192 real xyz hard-negative queries')
    hidx = np.random.default_rng(seed + 103).permutation(len(H[0]))[:192]
    hx, hy = H[0][hidx], H[1][hidx]
    # Create TUM queries independently to avoid invalid cross-sequence correspondences.
    val = to_data(create_queries(fx[64:79]) + create_queries(fd[64:79]))
    test_desk = to_data(create_queries(fd[82:]))
    test_bonn = to_data(create_queries(fb))
    if min(len(val[0]), len(test_desk[0])) < 45 or len(test_bonn[0]) < 100:
        raise RuntimeError('Insufficient real sensor test correspondences')

    model = StrongEncoder(args.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=.00005)
    out = pathlib.Path(args.out) / args.model / args.arm / ('seed_' + str(args.seed))
    out.mkdir(parents=True, exist_ok=True)
    curve = []
    best = -1.
    best_epoch = 0
    for epoch in range(args.epochs):
        model.train()
        inds = torch.randint(0, len(X), (960,))
        hperm = torch.randperm(len(hx))
        total = 0.
        for step, ix in enumerate(inds.split(64)):
            j = hperm[(torch.arange(step * 24, (step + 1) * 24) % len(hx))]
            a, b = model(augment(X[ix])), model(augment(Y[ix]))
            similarity = a @ b.T / .12
            target = torch.arange(len(ix))
            positive_loss = .5 * (F.cross_entropy(similarity, target)
                                  + F.cross_entropy(similarity.T, target))
            query = model(augment(hx[j]))
            candidates = model(augment(hy[j].reshape(-1, 7, 9, 9))).reshape(len(j), 5, -1)
            hard_loss = F.cross_entropy(torch.einsum('bd,bkd->bk', query, candidates) / .12,
                                        torch.zeros(len(j), dtype=torch.long))
            loss = positive_loss + .7 * hard_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.item())
        scheduler.step()
        rank = rank5(model, val)
        row = {'epoch': epoch + 1, 'train_loss': total / 15., 'validation_rank1': rank}
        curve.append(row)
        print('MATCHED_UNIQUE_EPOCH', args.model, args.arm, args.seed, json.dumps(row), flush=True)
        if rank > best:
            best, best_epoch = rank, epoch + 1
            torch.save(model.state_dict(), out / 'best.pt')
    model.load_state_dict(torch.load(out / 'best.pt', weights_only=True, map_location='cpu'))
    result = {
        'experiment': 'EXP-20261102-08',
        'model': args.model, 'arm': args.arm, 'seed': args.seed,
        'epochs': args.epochs, 'best_epoch': best_epoch,
        'train_unique_correspondence_indices': len(X), 'train_source_counts': counts,
        'train_positive_index_sha256': index_hash,
        'train_hard_query_indices': len(hx),
        'train_hard_query_sha256': hashlib.sha256(hidx.tobytes()).hexdigest(),
        'updates_per_epoch': 15, 'positive_draws_per_epoch': 960,
        'train_available': {name: len(v[0]) for name, v in train.items()},
        'dataset_index_sha256': {
            'tum_xyz': fingerprint(args.xyz_root),
            'tum_desk': fingerprint(args.desk_root),
            'bonn_box': fingerprint(args.bonn_root),
        },
        'validation_rank1': best,
        'TUM_desk_heldout': bootstrap(model, test_desk, seed + 70),
        'Bonn_blind': bootstrap(model, test_bonn, seed + 80),
        'curves': curve,
        'parameter_count': sum(v.numel() for v in model.parameters()),
        'limitations': ('Matched number of distinct correspondence indices, not independent '
            'physical objects/locations. All arms use the SAME xyz-hard-negative training '
            'support, which might favor xyz-only. TUM sequences share one laboratory; '
            'Bonn is one blind scene. GT poses serve as oracle labels. No instance-GT, '
            'end-to-end SLAM, relocalization, or navigation assessment.')
    }
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    print('EXP08_RESULT',
          json.dumps({k: v for k, v in result.items() if k != 'curves'}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
