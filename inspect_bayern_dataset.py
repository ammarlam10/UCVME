#!/usr/bin/env python3
"""Inspect Bayern forest height dataset structure."""
import h5py
import os

data_dir = "/workspace/data/Bayern_forest_height_reduced"
files = sorted([f for f in os.listdir(data_dir) if f.endswith(".h5")])

print(f"Total files: {len(files)}")
print(f"First 5 files: {files[:5]}")

f = h5py.File(os.path.join(data_dir, files[0]), "r")
print(f"\nDataset structure:")
print(f"  Input (rgb): {f['rgb'].shape} - RGB images")
print(f"  Target (ndsm): {f['ndsm'].shape} - Height maps")

rgb_data = f["rgb"][:]
ndsm_data = f["ndsm"][:]
print(f"\nValue ranges:")
print(f"  RGB: min={rgb_data.min():.2f}, max={rgb_data.max():.2f}")
print(f"  NDSM: min={ndsm_data.min():.2f}, max={ndsm_data.max():.2f}")

print(f"\nThis is a pixel-wise regression dataset:")
print(f"  - Input: RGB images (256x256x3)")
print(f"  - Target: Height maps (256x256x1)")
print(f"  - Samples per file: {f['rgb'].shape[0]}")
print(f"  - Total samples: {len(files) * f['rgb'].shape[0]}")

f.close()
