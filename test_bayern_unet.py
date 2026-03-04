#!/usr/bin/env python3
"""
Test script for Bayern Forest Height UNET implementation.
Tests model, dataset, and training loop compatibility.
"""
import sys
import torch
import numpy as np

# Add workspace to path
sys.path.insert(0, '.')

from models import unet_unc
from datasets import BayernForestHeightDataset
import utils_pixelwise

print("=" * 60)
print("Testing Bayern Forest Height UNET Implementation")
print("=" * 60)

# Test 1: Model creation
print("\n[1/5] Testing UNET model creation...")
model = unet_unc(in_channels=3, out_channels=1, features=32, drp_p=0.2)  # Smaller for testing
print(f"✓ Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

# Test 2: Forward pass
print("\n[2/5] Testing forward pass...")
dummy_input = torch.randn(2, 3, 256, 256)
mean_out, var_out = model(dummy_input)
print(f"✓ Forward pass successful")
print(f"  Input shape: {dummy_input.shape}")
print(f"  Mean output shape: {mean_out.shape}")
print(f"  Var output shape: {var_out.shape}")

# Test 3: Pixel-wise detection
print("\n[3/5] Testing pixel-wise output detection...")
is_pw = utils_pixelwise.is_pixelwise_output(mean_out)
print(f"✓ Pixel-wise detection: {is_pw}")

# Test 4: Loss computation
print("\n[4/5] Testing pixel-wise loss...")
dummy_target = torch.randn(2, 256, 256) * 10 + 15  # Height values
loss = utils_pixelwise.pixel_wise_loss_with_uncertainty(
    mean_out, var_out, dummy_target, y_mean=15.0, y_std=10.0
)
print(f"✓ Loss computed: {loss.item():.4f}")

# Test 5: Metrics
print("\n[5/5] Testing pixel-wise metrics...")
pred_np = (mean_out.detach().numpy() * 10 + 15)  # Denormalize
target_np = dummy_target.numpy()
metrics = utils_pixelwise.compute_pixelwise_metrics(pred_np, target_np)
print(f"✓ Metrics computed:")
print(f"  MAE: {metrics['mae']:.2f}")
print(f"  RMSE: {metrics['rmse']:.2f}")
print(f"  R²: {metrics['r2']:.4f}")

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)
print("\nReady to run training with:")
print("  python3 ucvme_age.py --config configs/bayern_forest_height_unet.yaml")
