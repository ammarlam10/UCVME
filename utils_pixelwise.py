"""
Utility functions for pixel-wise regression tasks.
Includes loss functions and metrics for spatial outputs.
"""
import torch
import numpy as np
import sklearn.metrics
from typing import Tuple


def pixel_wise_loss_with_uncertainty(
    pred_mean: torch.Tensor,
    pred_var: torch.Tensor,
    target: torch.Tensor,
    y_mean: float = 0.0,
    y_std: float = 1.0
) -> torch.Tensor:
    """
    Compute pixel-wise loss with uncertainty (negative log-likelihood).
    
    Loss = 0.5 * (exp(-var) * (pred - target)^2 + var)
    
    Args:
        pred_mean: Predicted mean (B, H, W) or (B, 1, H, W)
        pred_var: Predicted log-variance (B, H, W) or (B, 1, H, W)
        target: Ground truth (B, H, W) or (B, 1, H, W)
        y_mean: Mean for target normalization
        y_std: Std for target normalization
    
    Returns:
        Loss value (scalar)
    """
    # Ensure shapes match
    if pred_mean.dim() == 4 and pred_mean.size(1) == 1:
        pred_mean = pred_mean.squeeze(1)
    if pred_var.dim() == 4 and pred_var.size(1) == 1:
        pred_var = pred_var.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)
    
    # Normalize target
    target_norm = (target - y_mean) / y_std
    
    # Compute MSE
    mse = (pred_mean - target_norm) ** 2
    
    # Uncertainty-weighted loss
    loss = 0.5 * (torch.exp(-pred_var) * mse + pred_var)
    
    return loss.mean()


def pixel_wise_mse_loss(
    pred_mean: torch.Tensor,
    target: torch.Tensor,
    y_mean: float = 0.0,
    y_std: float = 1.0
) -> torch.Tensor:
    """
    Compute pixel-wise MSE loss.
    
    Args:
        pred_mean: Predicted mean (B, H, W) or (B, 1, H, W)
        target: Ground truth (B, H, W) or (B, 1, H, W)
        y_mean: Mean for target normalization
        y_std: Std for target normalization
    
    Returns:
        Loss value (scalar)
    """
    # Ensure shapes match
    if pred_mean.dim() == 4 and pred_mean.size(1) == 1:
        pred_mean = pred_mean.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)
    
    # Normalize target
    target_norm = (target - y_mean) / y_std
    
    # Compute MSE
    mse = (pred_mean - target_norm) ** 2
    
    return mse.mean()


def compute_pixelwise_metrics(
    pred: np.ndarray,
    target: np.ndarray
) -> dict:
    """
    Compute pixel-wise regression metrics.
    
    Args:
        pred: Predictions (N, H, W) or (N*H*W,)
        target: Ground truth (N, H, W) or (N*H*W,)
    
    Returns:
        Dictionary with metrics: mae, rmse, r2
    """
    # Flatten if needed
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Compute metrics
    mae = np.abs(pred_flat - target_flat).mean()
    rmse = np.sqrt(((pred_flat - target_flat) ** 2).mean())
    r2 = sklearn.metrics.r2_score(target_flat, pred_flat)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def is_pixelwise_output(output: torch.Tensor) -> bool:
    """
    Check if model output is pixel-wise (spatial) or image-level (scalar).
    
    Args:
        output: Model output tensor
    
    Returns:
        True if pixel-wise (3D or 4D with H,W > 1), False if image-level
    """
    if output.dim() == 2:
        # (B, C) - image-level
        return False
    elif output.dim() == 3:
        # (B, H, W) - pixel-wise
        return output.size(1) > 1 and output.size(2) > 1
    elif output.dim() == 4:
        # (B, C, H, W)
        return output.size(2) > 1 and output.size(3) > 1
    else:
        return False


def denormalize_pixelwise(
    pred: torch.Tensor,
    y_mean: float,
    y_std: float
) -> torch.Tensor:
    """
    Denormalize pixel-wise predictions.
    
    Args:
        pred: Normalized predictions
        y_mean: Mean used for normalization
        y_std: Std used for normalization
    
    Returns:
        Denormalized predictions
    """
    return pred * y_std + y_mean
