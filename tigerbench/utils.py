import numpy as np


def flatten(a):
    return np.asarray(a).reshape(-1)


def normalize_minmax(arr):
    arr = np.asarray(arr).astype(np.float64)
    mn, mx = arr.min(), arr.max()
    return np.zeros_like(arr) if mx == mn else (arr - mn) / (mx - mn)
