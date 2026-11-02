# Fundamental experiments: local CPU results (2026-11-02)

**Status:** executed locally, synthetic ray-cast RGB-D data; no actual YOLOE/SLAM, no validation on real objects. These are exploratory results, not a claim of 1000 independent neural architectures. The full local scripts and CSV are provided separately in the research archive.

## 1000 algorithmic compositions
10 image representations × 10 geometric representations × 10 temporal aggregation operators. 80 baseline scenes (391 observed identities), 40 OOD scenes (196 observed identities); OOD includes lighting shift, missing/noisy depth and swaps of near-identical object positions.

Best exploratory observed pairing: FFT magnitude + 3D pair-distance histogram + arithmetic mean temporal memory = **77.04%** identity matching on OOD scenes. Worst = **37.24%**. Results were *ranked on the OOD data*, therefore best score is optimistic without an untouched blind test set.

Average across all 1000 combinations:
- Image: FFT **62.67%**; gradient-orientation histogram **45.51%**.
- 3D: point pair-distances **66.21%**; centroid **56.76%**.
- Temporal: mean **59.25%**; first observation **57.05%**.

## Five independently trained small neural models
Train: 1037 object samples; held-out OOD: 403 samples; 6 epochs, CPU; 3D primitives only. Primitive class accuracy / metric radius MAE:
- RGB CNN: 49.13% / 0.0646 m
- RGB-D CNN: 49.13% / 0.0607 m
- RGB-D patch Transformer: 64.27% / 0.0572 m
- PointNet-lite: 49.13% / 0.0232 m
- CNN + PointNet: 49.13% / 0.0349 m

Most primitive classifiers were at chance; **do not infer architecture superiority**. Models are scratch-trained, not pretrained DINOv3/VGGT.

## Spatial relation representation: independent neural experiment
MLP trained for predicate 'between', using either absolute points `(a,b,p)` or translation-invariant relative vectors `(b-a,p-a)`, test on +9m global scene translation.
- Absolute model: ID F1 **0.8076**, translated F1 **0.0000**.
- Relative model: ID F1 **0.8384**, translated F1 **0.8459**.

The effect follows known translation invariance: `(b+t)-(a+t)=b-a`. It is a useful architectural constraint but **not novel mathematics**.

## Compact 3D geometry: 64-component RBF implicit SDF
Exact sphere/box/cylinder scenes on a 5832-cell grid, fitting 64 coefficients to 900 noisy samples:
- Sphere occupancy IoU **0.8138**, SDF MAE 0.0673m
- Box occupancy IoU **0.6745**, SDF MAE 0.0756m
- Cylinder occupancy IoU **0.7173**, SDF MAE 0.0706m

Implicit representation saves per-object samples at a measurable geometry-fidelity cost, especially at sharp edges. Fixed basis geometry is additional shared storage.

## Identity identifiability
For K truly indistinguishable objects with unobserved permutation, K! assignments have identical likelihood. Maximum probability of any one assignment under symmetric prior is `1/K!` (e.g. K=5 ⇒ 0.83%). Preserve an unordered state or multiple identity hypotheses rather than inventing certainty.

## Research decision
The strongest next scientific test is to compare voxel, point-set, implicit SDF, sparse keyframe, and learned object-relational memory **at an equal byte budget** and under real object movements, imperfect masks, and SLAM pose error. Use independent train/validation/test scenes, repeated seeds, confidence intervals, and downstream navigation loss. Keep depth-based safety independent of semantic policy.

## Reproducibility boundaries
Local result numbers are **not** the output of the separate GitHub workflow. GitHub's `research/fundamental/experiment_1000.py` is a smaller standalone reproduction of the physical 1000-composition concept; its own results must be taken from the workflow artifact.
