# ============================================================
# FILE: losses.py
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


# ====================================================================== #
# GAUSSIAN KERNEL
# ====================================================================== #

def create_gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    Create a 2-D Gaussian kernel of shape (1, 1, window_size, window_size).

    Parameters
    ----------
    window_size : int
        Side length of the square kernel (should be odd).
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    torch.Tensor  shape (1, 1, window_size, window_size), dtype float32
    """
    # 1-D Gaussian
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2                          # centre at 0
    gauss_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()               # normalise

    # Outer product → 2-D kernel
    gauss_2d = torch.outer(gauss_1d, gauss_1d)         # (W, W)
    gauss_2d = gauss_2d / gauss_2d.sum()               # normalise (safety)

    # Reshape to (1, 1, W, W) for F.conv2d
    kernel = gauss_2d.unsqueeze(0).unsqueeze(0)        # (1, 1, window_size, window_size)
    return kernel


# ====================================================================== #
# SSIM LOSS  (returns 1 − SSIM so it can be minimised)
# ====================================================================== #

def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """
    Compute the SSIM-based loss:  ``1 − mean(SSIM_map)``.

    Always runs in float32 internally so it is safe inside autocast.
    float16 underflows the Gaussian denominator (values < 6e-5 flush to zero)
    producing inf/nan gradients that poison subsequent batches.

    Parameters
    ----------
    pred   : (B, 1, H, W) predicted values in [0, 1]
    target : (B, 1, H, W) ground-truth values in [0, 1]
    window_size : int  — Gaussian kernel size
    C1, C2 : float   — SSIM stability constants

    Returns
    -------
    loss : scalar tensor (same dtype as input pred)
    """
    orig_dtype = pred.dtype

    # Always compute in float32 — critical for numerical stability under AMP
    pred   = pred.float()
    target = target.float()

    kernel  = create_gaussian_kernel(window_size, sigma=1.5).to(pred.device)  # float32
    padding = window_size // 2

    # Local means
    mu1 = F.conv2d(pred,   kernel, padding=padding, groups=1)
    mu2 = F.conv2d(target, kernel, padding=padding, groups=1)

    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # Local variances and covariance
    sigma1_sq = F.conv2d(pred   * pred,   kernel, padding=padding, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=padding, groups=1) - mu2_sq
    sigma12   = F.conv2d(pred   * target, kernel, padding=padding, groups=1) - mu1_mu2

    # SSIM map — use clamp instead of + epsilon so denominator can never be negative-small
    numerator   = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map    = numerator / denominator.clamp(min=1e-8)

    # Clamp to valid SSIM range before averaging to stop any stray inf/nan
    ssim_map = ssim_map.clamp(-1.0, 1.0)

    loss = 1.0 - ssim_map.mean()
    return loss.to(orig_dtype)


# ====================================================================== #
# COMBINED LOSS
# ====================================================================== #

class CombinedLoss(nn.Module):
    """
    Combined loss:
        total = mae_weight * (mae + weighted_mae) / 2  +  ssim_weight * ssim_loss

    Heavy-precipitation weighting is applied to the MAE component:
    pixels where the ground truth exceeds ``heavy_thresh`` are up-weighted
    by ``heavy_weight``.

    Parameters
    ----------
    mae_weight   : float  — weight for the MAE term  (default 0.5)
    ssim_weight  : float  — weight for the SSIM term (default 0.5)
    heavy_weight : float  — multiplier for high-rain pixels (default 3.0)
    heavy_thresh : float  — normalised intensity threshold for "heavy rain" (default 0.5)
    """

    def __init__(
        self,
        mae_weight: float = 0.5,
        ssim_weight: float = 0.5,
        heavy_weight: float = 3.0,
        heavy_thresh: float = 0.5,
    ):
        super().__init__()
        self.mae_weight   = mae_weight
        self.ssim_weight  = ssim_weight
        self.heavy_weight = heavy_weight
        self.heavy_thresh = heavy_thresh

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        pred   : (B, 1, H, W)
        target : (B, 1, H, W)

        Returns
        -------
        total_loss : scalar tensor (graph-connected, suitable for .backward())
        components : dict with keys "mae", "weighted_mae", "ssim_loss", "total"
        """
        # Standard mean absolute error
        mae = F.l1_loss(pred, target)

        # Heavy-rain pixel weighted MAE
        weight_map = torch.where(
            target > self.heavy_thresh,
            torch.full_like(target, self.heavy_weight),
            torch.ones_like(target),
        )
        weighted_mae = (weight_map * (pred - target).abs()).mean()

        # SSIM loss (1 − SSIM)
        sl = ssim_loss(pred, target)

        # Combined total
        total = (
            self.mae_weight * (mae + weighted_mae) / 2.0
            + self.ssim_weight * sl
        )

        components: Dict[str, float] = {
            "mae":          mae.item(),
            "weighted_mae": weighted_mae.item(),
            "ssim_loss":    sl.item(),
            "total":        total.item(),
        }

        return total, components