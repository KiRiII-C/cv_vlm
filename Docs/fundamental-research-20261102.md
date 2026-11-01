# Fundamental spatial representations: first controlled study (2026-11-02)

## Scope and provenance
The purpose is **not** to claim 1000 new neural network architectures. One thousand physically grounded, executable mathematical **pipeline compositions** are evaluated: 10 image descriptors × 10 geometry descriptors × 10 temporal memory operators. Operators use ray-cast RGB-D observations of analytic spheres/boxes, depth noise, missing pixels, known masks, and a controlled object-position swap. This is an exploratory *surrogate* study, not a real RGB-D benchmark.

The experiment in `research/fundamental/experiment_1000.py` is independently executable on GitHub Actions with NumPy/SciPy. It records a complete CSV and summary JSON as build artifacts.

## Why these experiments are preliminary
- Primitive renderer uses ideal segmentation and camera pose (with no full SLAM, no real sensor intrinsics, no sensor-specific calibration).
- Color and object geometry can allow overly easy recognition; swapped near-identical objects are an identifiability challenge.
- Hyperparameter selection on a reported test set is optimistic. A separate blind test split is necessary.
- The learned model investigation (tiny RGB CNN, RGB-D CNN, patch Transformer, PointNet-lite, cross-modal fusion) was executed locally on CPU; it is **not** included in the GitHub workflow until the full experimental implementation and dependencies are uploaded.
- 1000 combinations do not imply 1000 foundational physical principles. The relevant independent mathematical operators are counted in the methods, not the combinations.

## Fundamental hypotheses
1. A task-sufficient 3D latent state may be more efficient than a full dense reconstruction.
2. Relative spatial encoding provides translation invariance by construction; absolute-coordinate MLPs can fail under global transforms.
3. Signed-distance implicit fields trade geometric fidelity for compact storage and continuous queryability.
4. When identical objects are permuted without discriminating observations, stable historical identity is not identifiable; use set-level state or multi-hypothesis memory.
5. Pose-coupled memory should reproject evidence after graph pose corrections; independent 3D centroids can become inconsistent.
6. Neural spatial policies require a separately verified local safety controller.

## Acceptance criteria for later scientific claims
Use independent real RGB-D scenes, multiple random seeds, confidence intervals, balanced object identities and appearances, exact train/validation/test separation, calibration and pose-error stress tests, plus downstream navigation impact. Do not report synthetic matching accuracy as field performance.

## References for concept comparison
- NeRF (ECCV 2020): https://arxiv.org/abs/2003.08934
- 3D Gaussian Splatting (SIGGRAPH 2023): https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- DUSt3R (CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.pdf
- VGGT (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html
- DINOv3 (2025): https://arxiv.org/abs/2508.10104
