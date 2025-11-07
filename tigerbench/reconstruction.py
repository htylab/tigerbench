import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.special import digamma


def mae(img, ref):
    return float(np.mean(np.abs(img - ref)))


def mse(img, ref):
    return float(np.mean((img - ref) ** 2))


def psnr(img, ref, data_range=None):
    if data_range is None:
        data_range = ref.max() - ref.min()
    m = mse(img, ref)
    return float("inf") if m == 0 else float(20.0 * np.log10(data_range) - 10.0 * np.log10(m))


def ssim_nd(img, ref, data_range=None, sigma=1.5, c1=None, c2=None):
    img = img.astype(np.float64)
    ref = ref.astype(np.float64)
    if data_range is None:
        data_range = ref.max() - ref.min() + 1e-8
    if c1 is None:
        c1 = (0.01 * data_range) ** 2
    if c2 is None:
        c2 = (0.03 * data_range) ** 2
    mu_x = gaussian_filter(img, sigma)
    mu_y = gaussian_filter(ref, sigma)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = gaussian_filter(img * img, sigma) - mu_x2
    sigma_y2 = gaussian_filter(ref * ref, sigma) - mu_y2
    sigma_xy = gaussian_filter(img * ref, sigma) - mu_xy
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / (den + 1e-12)
    return float(np.mean(ssim_map))


def ncc(img, ref, eps=1e-8):
    x = img.reshape(-1).astype(np.float64)
    y = ref.reshape(-1).astype(np.float64)
    x, y = x - x.mean(), y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y) + eps
    return float(np.dot(x, y) / denom)


def histogram_mi(img, ref, bins=64):
    x = img.reshape(-1)
    y = ref.reshape(-1)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    if x_max == x_min:
        x_max += 1e-6
    if y_max == y_min:
        y_max += 1e-6
    x_idx = np.floor((x - x_min) / (x_max - x_min) * (bins - 1)).astype(int)
    y_idx = np.floor((y - y_min) / (y_max - y_min) * (bins - 1)).astype(int)
    joint = np.zeros((bins, bins), dtype=np.float64)
    for i in range(len(x_idx)):
        joint[x_idx[i], y_idx[i]] += 1.0
    joint /= joint.sum() + 1e-12
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if joint[i, j] > 0:
                mi += joint[i, j] * (np.log(joint[i, j] + 1e-12) - np.log(px[i, 0] * py[0, j] + 1e-12))
    return {"value": float(mi), "bins": bins}


def ksg_mi(img, ref, k=3):
    x = img.reshape(-1).astype(np.float64)
    y = ref.reshape(-1).astype(np.float64)
    n = len(x)
    if n < 100:
        return {"value": 0.0, "warning": "n < 100"}
    xy = np.column_stack((x, y))
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x.reshape(-1, 1))
    tree_y = cKDTree(y.reshape(-1, 1))
    dist_xy, _ = tree_xy.query(xy, k=k + 1)
    eps = dist_xy[:, k]
    nx = np.array([tree_x.query_ball_point(x[i], r=eps[i], return_length=True) - 1 for i in range(n)])
    ny = np.array([tree_y.query_ball_point(y[i], r=eps[i], return_length=True) - 1 for i in range(n)])
    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    return {"value": float(mi), "k": k}
