"""Object-level 3D and appearance association; conservative under uncertainty."""
import numpy as np


def overlap_3d(low1,hi1,low2,hi2):
    inter=np.prod(np.maximum(np.minimum(hi1,hi2)-np.maximum(low1,low2),0))
    v1=np.prod(np.maximum(hi1-low1,0));v2=np.prod(np.maximum(hi2-low2,0))
    return float(inter/max(v1+v2-inter,1e-12))


def score(track,obs):
    """Similarity [0..1] or None if the hard identity gate fails.

    Same class is required in the MVP; no embedding -> conservative geometry-only.
    Calibration in a future validation set is essential; thresholds are provisional.
    """
    if track.label.lower().strip()!=obs.label.lower().strip():
        return None
    if track.map_version!=obs.map_version:
        return None  # pose-graph corrections must reproject old tracks first
    delta=float(np.linalg.norm(track.center-obs.center))
    extent=np.linalg.norm(np.maximum(track.high-track.low,obs.high-obs.low))
    radius=max(.20, min(1.0,.5*extent))
    if delta>max(.45,1.5*extent):
        return None
    iou=overlap_3d(track.low,track.high,obs.low,obs.high)
    proximity=float(np.exp(-.5*(delta/radius)**2))
    if track.appearance is not None and obs.appearance is not None:
        if track.appearance.shape!=obs.appearance.shape:
            return None
        cosine=float(np.dot(track.appearance,obs.appearance))
        # Strongly different appearances cannot match without strong 3D overlap.
        if cosine<.35 and iou<.6:return None
        appearance=max(0.,cosine)
        return .45*proximity+.20*iou+.35*appearance
    # Without appearance, require much stronger geometric evidence.
    if delta>.30 and iou<.25:
        return None
    return .75*proximity+.25*iou
