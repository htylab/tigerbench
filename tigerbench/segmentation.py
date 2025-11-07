import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion


def _to_label_map(a):
    if a.ndim >= 2 and a.shape[-1] <= 64 and a.dtype not in (np.int32, np.int64):
        return np.argmax(a, axis=-1)
    return a


def _flatten_int(a):
    return a.astype(int).reshape(-1)


def _unique_union(a, b):
    return np.unique(np.concatenate([a, b]))


def dice_multilabel(mask_true, mask_pred, labels=None, ignore_background=True, eps=1e-6):
    mask_true = _to_label_map(mask_true)
    mask_pred = _to_label_map(mask_pred)
    m1, m2 = _flatten_int(mask_true), _flatten_int(mask_pred)
    if labels is None:
        labels = _unique_union(m1, m2)
    labels = np.array(labels, dtype=int)
    if ignore_background and 0 in labels:
        labels = labels[labels != 0]
    scores = {}
    for lab in labels:
        a, b = (m1 == lab), (m2 == lab)
        inter = np.sum(a & b)
        union = np.sum(a) + np.sum(b)
        d = 1.0 if union == 0 else (2.0 * inter) / (union + eps)
        scores[int(lab)] = float(d)
    mean_v = float(np.mean(list(scores.values()))) if scores else 1.0
    return {"per_label": scores, "mean": mean_v}


def iou_multilabel(mask_true, mask_pred, labels=None, ignore_background=True, eps=1e-6):
    mask_true = _to_label_map(mask_true)
    mask_pred = _to_label_map(mask_pred)
    m1, m2 = _flatten_int(mask_true), _flatten_int(mask_pred)
    if labels is None:
        labels = _unique_union(m1, m2)
    labels = np.array(labels, dtype=int)
    if ignore_background and 0 in labels:
        labels = labels[labels != 0]
    scores = {}
    for lab in labels:
        a, b = (m1 == lab), (m2 == lab)
        inter = np.sum(a & b)
        union = np.sum(a | b)
        val = 1.0 if union == 0 else inter / (union + eps)
        scores[int(lab)] = float(val)
    mean_v = float(np.mean(list(scores.values()))) if scores else 1.0
    return {"per_label": scores, "mean": mean_v}


def _surface(mask):
    if not np.any(mask):
        return mask
    return mask ^ binary_erosion(mask)


def _percentile95_surface_distance(a_surf, b_bin):
    if not np.any(a_surf) or not np.any(b_bin):
        return 0.0
    dt = distance_transform_edt(~b_bin)
    dists = dt[a_surf]
    return float(np.percentile(dists, 95)) if dists.size > 0 else 0.0


def hd95(mask_true, mask_pred, labels=None):
    mask_true = _to_label_map(mask_true)
    mask_pred = _to_label_map(mask_pred)
    if labels is None:
        labels = _unique_union(mask_true.reshape(-1), mask_pred.reshape(-1))
    out = {}
    for lab in labels:
        if lab == 0:
            continue
        a, b = (mask_true == lab), (mask_pred == lab)
        sa, sb = _surface(a), _surface(b)
        d1 = _percentile95_surface_distance(sa, b)
        d2 = _percentile95_surface_distance(sb, a)
        out[int(lab)] = float((d1 + d2) / 2.0)
    mean_v = float(np.mean(list(out.values()))) if out else 0.0
    return {"per_label": out, "mean": mean_v}


def _avg_surface_distance(a_surf, b_bin):
    if not np.any(a_surf) or not np.any(b_bin):
        return 0.0
    dt = distance_transform_edt(~b_bin)
    dists = dt[a_surf]
    return float(np.mean(dists)) if dists.size > 0 else 0.0


def asd(mask_true, mask_pred, labels=None):
    mask_true = _to_label_map(mask_true)
    mask_pred = _to_label_map(mask_pred)
    if labels is None:
        labels = _unique_union(mask_true.reshape(-1), mask_pred.reshape(-1))
    out = {}
    for lab in labels:
        if lab == 0:
            continue
        a, b = (mask_true == lab), (mask_pred == lab)
        sa, sb = _surface(a), _surface(b)
        d1 = _avg_surface_distance(sa, b)
        d2 = _avg_surface_distance(sb, a)
        out[int(lab)] = float((d1 + d2) / 2.0)
    mean_v = float(np.mean(list(out.values()))) if out else 0.0
    return {"per_label": out, "mean": mean_v}
