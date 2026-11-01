# Fundamental research on RGB-D scene memory — 2026-11-02

## Consolidated corrected real-data results (supersedes preliminary Bonn metrics below)

After an explicit **calibration audit** against the published Bonn marker-to-camera extrinsic, selected on the first half and evaluated on the last half, the preferred convention was \`T_world_camera = T_world_marker @ T_ROS @ T_m\`. Held-out median inter-frame depth reprojection error changed from **0.2835 m** (uncorrected) to **0.0458 m**. The originally reported Bonn descriptor and voxel coverage numbers were therefore invalid; use the **corrected** numbers below (GitHub Actions [successful run](https://github.com/KiRiII-C/cv_vlm/actions/runs/41357393506)).

Corrected Bonn results: RGB descriptor 90.632%, DCT 89.520%, depth geometry 89.871%, pairwise 3D-distance spectrum 89.754%, camera-frame XYZ 90.340%, fused RGB+3D 93.443% over **1708** measured correspondences. Eight-frame memory held-out observed-voxel recall: uniform 0.70181; random single seed 0.62033; greedy 0.72575; Jaccard-diverse 0.56321. Oracle-pose ego-motion compensation: RGB MAE 0.17940 → 0.11724, depth MAE 0.36588m → 0.15227m, over 54 frame pairs; this is not SLAM and does not certify dynamic object segmentation.

**Actually trained neural representations** on TUM real RGB-D: 100 frames, 1719 GT-posed positive patch pairs, 1155 train/564 temporal holdout, 8 epochs from scratch. Patch-ranking scores: MLP 0.950355 (45,728 parameters), RGB CNN 0.907801 (8,672), RGB-D CNN 0.943262 (8,888), RGB + locally-centered XYZ CNN 0.904255 (9,320), one-layer patch Transformer 0.969858 (22,736). Unequal capacities, one room, relatively easy random negatives; does not establish Transformer superiority in 3D mapping.

**Negative control: predictive memory.** 140 real frames per sequence; one-step normalized RGB/inverse-depth MAE with rank-8 latent encoding, persistence vs per-frame PCA vs 3-frame Hankel SVD:
- TUM RGB: 0.128754 / 0.171787 / 0.198222; inverse-depth: 0.042743 / 0.058815 / 0.065920.
- Bonn RGB: 0.047163 / 0.108804 / 0.125065; inverse-depth: 0.007422 / 0.038717 / 0.041831.
The simplistic compressed predictive states **lost** to persistence. Formulating a latent state without separating ego-motion does not by itself yield useful spatial memory.

The source implementation and real-data download workflows are in this branch. All measured results are GitHub Actions artifacts; the old Bonn table below should be considered an audit trail, **not the final result**.

---



## Evidence levels

- **Executed on real RGB-D:** TUM fr1/xyz (798 synchronized pairs; 56 sampled frames), Bonn moving_obstructing_box (590 pairs; 56 frames). Full source and raw CI logs: [GitHub Actions run](https://github.com/KiRiII-C/cv_vlm/actions/runs/41356590629).
- **Exploratory:** local neural-model and implicit SDF experiments were run on constructed primitives (reported separately in Docs/fundamental-local-results-20261102.md).
- **Unverified hypotheses:** theoretical approaches proposed below. NONE is established as a new scientific invention or shown to outperform pretrained VLM / semantic 3D systems.
- Published results are cited to distinguish known mathematical ideas and existing robotics uses from proposed extensions.

## Real recorded-sensor findings

### Local 3D correspondence ranking, no neural learning
A measured 3D point in frame t was projected into frame t+delta using published GT pose, and retained only when observed depth agreed within 8cm. For each true match and random spatial negative, compare descriptor distances. This measures binary correspondence ranking, **not** global object ID accuracy.

| Descriptor | TUM (n=1301) | Bonn (n=643) |
|---|---:|---:|
| RGB mean+variance | 0.93082 | 0.73406 |
| DCT | 0.89470 | 0.63764 |
| Depth eigenvalue/geometric scales | 0.81322 | 0.73717 |
| 3D pairwise-distance spectrum | 0.82244 | 0.77760 |
| camera-frame XYZ + spread | 0.71637 | 0.93002 |
| RGB + pairwise spectrum | 0.89700 | 0.79938 |

Contradicts simplistic claims of universally superior invariant 3D descriptors: *which nuisance variables to quotient depends on the decision task and motion regime*. The camera-frame XYZ score on Bonn may exploit small inter-frame camera movement rather than reliable long-term identity: test far-apart frames, cross-session displacement, and independent scene splits before concluding.

### Eight-frame scene-memory budget: held-out observed voxel recall
Voxel size 12cm. Train candidates: chronological first 72% of sampled frames; held-out surface voxels: last 28%, pose by GT. Not total ground-truth object geometry.

| Frame selection | TUM | Bonn |
|---|---:|---:|
| Uniform | 0.69098 | 0.30466 |
| Random, single seed | 0.66446 | 0.26277 |
| Greedy coverage | 0.82626 | 0.36044 |
| Jaccard diversity | 0.76393 | 0.27219 |

Greedy coverage wins in both sequences on this specific proxy. It is **not novel** and not a learned task-sufficient memory. Changes in Bonn are not semantically annotated; do not interpret low coverage as object-change detection.

## Competing fundamental mathematical paradigms

### I. Observable equivalence classes, prediction, and control sufficiency
Define an action-dependent observational equivalence:
s ~ s' iff for all admissible future action sequences a(0:H), their conditional future observation-and-task-outcome laws coincide.
Represent a belief on the equivalence class rather than an arbitrary object's historical ID. Requires testing which *admissible actions* sufficiently resolve ambiguity.

**This concept is not new.** See *Observable quotient world models for knowledge-state abstraction under partial observability*, Knowledge-Based Systems, Oct 2026, DOI 10.1016/j.knosys.2026.116768. The opportunity is operationalizing it for RGB-D local geometry + semantic instance memory + safety.

Falsify by comparing object-centric memory, RGB-D keyframe memory, and predictive latent memory at fixed bytes, under unseen camera start poses and object rearrangements. Target success/re-localization error, not reconstruction PSNR alone.

### II. Local invariant geometry: Gram matrices and Euclidean symmetry
For point set X, centered Gram G = (X-mean(X))(X-mean(X))^T is invariant to rotation and translation, but not to missing surfaces or changed point ordering. Histograms of interpoint distances are permutation/SE(3)-invariant but lose spatial arrangement and may create collisions.

Empirical real RGB-D result above highlights that invariance may discard useful camera-frame information. Study adaptive symmetry quotienting rather than always enforcing maximal invariance.

### III. Ray visibility, existence belief, and unbalanced matching
For a candidate old object at x, an undetected observation is meaningful negative evidence only if x was predicted to be visible. A valid depth ray provides occupied/free-space evidence while unobserved regions supply none.

Bayesian log-odds update under visibility and non-detection:
logit P(E|no detection, visible) = logit P(E) + log[(1-P_D)/(1-P_FA)].
If not visible, no negative update.

A candidate association objective can combine:
E = <Gamma,C_SE3> + epsilon*KL(Gamma||mu tensor nu)
    + tau*KL(Gamma*1||visible(mu)) + tau*KL(Gamma^T*1||nu)
    + lambda*ray_free_space_penalty.
Gamma is a soft, possibly unbalanced transport plan. Ambiguous identical objects are represented as set-level or multimodal correspondences rather than arbitrary IDs.

**Partial/unbalanced optimal transport for 3D point-cloud alignment is ALREADY published**, including Bai et al. (2025), and 3D Zernike + UOT (ROBIO 2025). Novelty would have to be demonstrated specifically in coupling visibility, uncertainty, task sufficiency, and scene-state updates.

### IV. Local-chart compatibility (sheaf-like spatiotemporal consistency)
Each camera keyframe has a local image/depth chart and locally estimated features. On overlaps, transition maps T_ij transfer feature/geometry across charts.
Minimize sum over view overlap edges (i,j):
||F_ij(z_i)-F_ji(z_j)||^2_{Sigma_ij^{-1}}.
Use cycle consistency to separate pose drift from local dynamic-object changes. This is a research design, not an implemented or proved new sheaf construction.

### V. Predictive state and finite-rank Hankel operators
Rather than reconstruct all pixels, seek a low-dimensional predictive state for future sensory/task outcomes:
h_t = P(o_{t+1:t+H} | history, candidate actions).
Operational proxy: empirical past-block Hankel SVD compared to single-frame PCA under equal rank and fixed chronological held-out split.
A first real RGB-D one-step prediction test is implemented in research/fundamental/predictive_state.py. It does NOT evaluate action-conditioned dynamics or prove semantic memory sufficiency.

### VI. Rate–distortion / information bottleneck, conditioned on tasks
Seek M minimizing information retained from sensor history under constraint on downstream decision risk:
min_M I(M;O_1:t) + beta E[L_task(M,S_t)].
Approximate with constrained memory bytes, calibrated uncertainty, navigation cost and false-confidence penalty.
The known rate-distortion principle is not a novel discovery; its robotics application must be experimentally distinguished from existing work.

## Critical negative controls before scientific claims

1. Test correspondence with far-separated revisits, motion-amplified sequences, duplicates and changes, not just nearby poses.
2. TUM has no full per-instance ID ground truth; use explicit instance annotated data for IDF1/3D mAP (e.g. ScanNet through approved agreement, Arti4D-Semantic where permissible).
3. Compare learned point transformer, DINOv3, and VGGT weights after installing CUDA-capable compute. Current CI only uses CPU and numerical methods.
4. Use at least three independent environments per shift and multiple random seeds, uncertainty intervals, and unmodified test scenes.
5. Compare *downstream* relocalization, object-disappearance false positives, semantic query accuracy, route success and computational resource use.
6. Full physically motivated representation learning is not a finite enumeration of all algorithms. Focus on falsifiable families with distinct inductive biases.

## Dataset attribution

- TUM RGB-D Benchmark: J. Sturm et al., *A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS 2012, https://cvg.cit.tum.de/data/datasets/rgbd-dataset.
- Bonn RGB-D Dynamic: E. Palazzolo et al., *ReFusion*, IROS 2019, https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/.
- Source recordings are downloaded by GitHub Actions for experimentation; only numerical results are uploaded as workflow artifacts.
