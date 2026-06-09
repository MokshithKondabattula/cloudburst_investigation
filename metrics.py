# ============================================================
# FILE: metrics.py
# ============================================================

from typing import Dict

import numpy as np
from scipy.signal import convolve2d


# ====================================================================== #
# BASIC METRICS
# ====================================================================== #

def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(pred - target)))


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def compute_psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio.

    Returns
    -------
    float : PSNR in dB.  Returns 100.0 if MSE == 0 (perfect prediction).
    """
    mse = float(np.mean((pred - target) ** 2))
    if mse == 0.0:
        return 100.0
    return float(10.0 * np.log10(max_val ** 2 / mse))


def compute_pearson(pred: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation coefficient between flattened arrays."""
    corr_matrix = np.corrcoef(pred.flatten(), target.flatten())
    return float(corr_matrix[0, 1])


# ====================================================================== #
# SSIM (pure NumPy / SciPy)
# ====================================================================== #

def _gaussian_kernel_2d(window_size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """Return a normalised 2-D Gaussian kernel as a numpy array."""
    coords = np.arange(window_size, dtype=np.float64) - window_size // 2
    gauss_1d = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    gauss_1d /= gauss_1d.sum()
    gauss_2d = np.outer(gauss_1d, gauss_1d)
    gauss_2d /= gauss_2d.sum()
    return gauss_2d


def _conv2d_numpy(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D convolution with 'same' boundary ('fill' = 0)."""
    return convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0.0)


def compute_ssim_metric(
    pred: np.ndarray,
    target: np.ndarray,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> float:
    """
    Compute the mean SSIM between two 2-D grayscale images.

    Parameters
    ----------
    pred   : 2-D numpy array in [0, 1]
    target : 2-D numpy array in [0, 1]

    Returns
    -------
    float : mean SSIM value (NOT 1-SSIM).
    """
    kernel = _gaussian_kernel_2d(window_size, sigma)

    mu1 = _conv2d_numpy(pred,   kernel)
    mu2 = _conv2d_numpy(target, kernel)

    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _conv2d_numpy(pred   * pred,   kernel) - mu1_sq
    sigma2_sq = _conv2d_numpy(target * target, kernel) - mu2_sq
    sigma12   = _conv2d_numpy(pred   * target, kernel) - mu1_mu2

    numerator   = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map    = numerator / (denominator + 1e-8)

    return float(np.mean(ssim_map))


# ====================================================================== #
# CONTINGENCY-TABLE METRICS
# ====================================================================== #

def compute_csi_pod_far(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Compute CSI, POD, and FAR at a given threshold.

    Parameters
    ----------
    pred      : numpy array (any shape), values in [0, 1]
    target    : numpy array (same shape as pred), values in [0, 1]
    threshold : float — binarisation threshold

    Returns
    -------
    dict with keys "POD", "FAR", "CSI"
    """
    pred_bin   = pred   >= threshold
    target_bin = target >= threshold

    TP = float(( pred_bin &  target_bin).sum())
    FP = float(( pred_bin & ~target_bin).sum())
    FN = float((~pred_bin &  target_bin).sum())

    pod = TP / (TP + FN + 1e-8)
    far = FP / (TP + FP + 1e-8)
    csi = TP / (TP + FP + FN + 1e-8)

    return {"POD": float(pod), "FAR": float(far), "CSI": float(csi)}


# ====================================================================== #
# AGGREGATED METRICS
# ====================================================================== #

def compute_all_metrics(
    pred_batch: np.ndarray,
    target_batch: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all metrics over a batch of predictions.

    Parameters
    ----------
    pred_batch   : (N, 1, H, W) numpy array, values in [0, 1]
    target_batch : (N, 1, H, W) numpy array, values in [0, 1]

    Returns
    -------
    dict with keys:
        MAE, RMSE, PSNR, Pearson, SSIM,
        CSI_0.2, POD_0.2, FAR_0.2,
        CSI_0.4, POD_0.4, FAR_0.4,
        CSI_0.6, POD_0.6, FAR_0.6
    """
    N = pred_batch.shape[0]

    # Flatten per-sample to (H, W)
    preds   = pred_batch[:, 0, :, :]    # (N, H, W)
    targets = target_batch[:, 0, :, :]  # (N, H, W)

    mae_list     = []
    rmse_list    = []
    psnr_list    = []
    pearson_list = []
    ssim_list    = []

    csi_02_list = []; pod_02_list = []; far_02_list = []
    csi_04_list = []; pod_04_list = []; far_04_list = []
    csi_06_list = []; pod_06_list = []; far_06_list = []

    for i in range(N):
        p = preds[i]    # (H, W)
        t = targets[i]  # (H, W)

        mae_list.append(compute_mae(p, t))
        rmse_list.append(compute_rmse(p, t))
        psnr_list.append(compute_psnr(p, t))
        pearson_list.append(compute_pearson(p, t))
        ssim_list.append(compute_ssim_metric(p, t))

        for thresh, csi_l, pod_l, far_l in [
            (0.2, csi_02_list, pod_02_list, far_02_list),
            (0.4, csi_04_list, pod_04_list, far_04_list),
            (0.6, csi_06_list, pod_06_list, far_06_list),
        ]:
            ctab = compute_csi_pod_far(p, t, thresh)
            csi_l.append(ctab["CSI"])
            pod_l.append(ctab["POD"])
            far_l.append(ctab["FAR"])

    results: Dict[str, float] = {
        "MAE":     float(np.mean(mae_list)),
        "RMSE":    float(np.mean(rmse_list)),
        "PSNR":    float(np.mean(psnr_list)),
        "Pearson": float(np.mean(pearson_list)),
        "SSIM":    float(np.mean(ssim_list)),

        "CSI_0.2": float(np.mean(csi_02_list)),
        "POD_0.2": float(np.mean(pod_02_list)),
        "FAR_0.2": float(np.mean(far_02_list)),

        "CSI_0.4": float(np.mean(csi_04_list)),
        "POD_0.4": float(np.mean(pod_04_list)),
        "FAR_0.4": float(np.mean(far_04_list)),

        "CSI_0.6": float(np.mean(csi_06_list)),
        "POD_0.6": float(np.mean(pod_06_list)),
        "FAR_0.6": float(np.mean(far_06_list)),
    }

    return results
