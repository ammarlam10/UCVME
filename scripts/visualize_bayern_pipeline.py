"""
Visualize Bayern Forest Height data preprocessing pipeline.

This script shows how the data looks at each stage:
1. Raw data from HDF5 files
2. After normalization
3. After augmentation (horizontal flip, vertical flip, random crop)
4. Final tensor format fed to the model

Usage:
    python scripts/visualize_bayern_pipeline.py
"""

import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.bayern_forest_height import BayernForestHeightDataset


def denormalize_rgb(rgb_normalized, mean, std):
    """Denormalize RGB for visualization."""
    if isinstance(mean, (float, int)):
        return rgb_normalized * std + mean
    else:
        mean_reshaped = np.array(mean).reshape(3, 1, 1)
        std_reshaped = np.array(std).reshape(3, 1, 1)
        return rgb_normalized * std_reshaped + mean_reshaped


def visualize_raw_data(data_dir, num_samples=3):
    """Visualize raw data directly from HDF5 files."""
    print("\n" + "="*80)
    print("STAGE 1: RAW DATA FROM HDF5")
    print("="*80)
    
    # Get first h5 file
    h5_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
    if not h5_files:
        print(f"No .h5 files found in {data_dir}")
        return
    
    first_file = os.path.join(data_dir, h5_files[0])
    print(f"Reading from: {first_file}")
    
    with h5py.File(first_file, 'r') as f:
        print(f"Keys in HDF5: {list(f.keys())}")
        
        rgb = f['rgb'][:num_samples]  # (N, 256, 256, 3)
        ndsm = f['ndsm'][:num_samples]  # (N, 256, 256, 1)
        
        print(f"RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
        print(f"RGB range: [{rgb.min():.2f}, {rgb.max():.2f}]")
        print(f"NDSM shape: {ndsm.shape}, dtype: {ndsm.dtype}")
        print(f"NDSM range: [{ndsm.min():.2f}, {ndsm.max():.2f}]")
        
        # Create visualization
        fig = plt.figure(figsize=(15, 5 * num_samples))
        gs = gridspec.GridSpec(num_samples, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        for i in range(num_samples):
            # RGB image
            ax_rgb = fig.add_subplot(gs[i, 0])
            rgb_img = rgb[i].astype(np.uint8)
            ax_rgb.imshow(rgb_img)
            ax_rgb.set_title(f"Sample {i}: RGB (raw)\nRange: [{rgb[i].min():.0f}, {rgb[i].max():.0f}]")
            ax_rgb.axis('off')
            
            # NDSM (height map)
            ax_ndsm = fig.add_subplot(gs[i, 1])
            ndsm_img = ndsm[i, :, :, 0]
            im_ndsm = ax_ndsm.imshow(ndsm_img, cmap='viridis')
            ax_ndsm.set_title(f"Sample {i}: NDSM/Height (raw)\nRange: [{ndsm_img.min():.2f}, {ndsm_img.max():.2f}]m")
            ax_ndsm.axis('off')
            plt.colorbar(im_ndsm, ax=ax_ndsm, fraction=0.046, pad=0.04)
            
            # Overlay visualization
            ax_overlay = fig.add_subplot(gs[i, 2])
            # Normalize RGB to [0, 1] for overlay
            rgb_normalized = rgb_img / 255.0
            # Create a semi-transparent overlay
            ax_overlay.imshow(rgb_normalized)
            im_overlay = ax_overlay.imshow(ndsm_img, cmap='jet', alpha=0.5)
            ax_overlay.set_title(f"Sample {i}: RGB + Height Overlay")
            ax_overlay.axis('off')
            plt.colorbar(im_overlay, ax=ax_overlay, fraction=0.046, pad=0.04)
        
        plt.suptitle("Stage 1: Raw Data from HDF5", fontsize=16, fontweight='bold')
        plt.savefig('/work/ammar/sslrp/UCVME/output/bayern_viz_1_raw.png', dpi=150, bbox_inches='tight')
        print(f"Saved: output/bayern_viz_1_raw.png")
        plt.close()


def visualize_dataset_pipeline(data_dir, num_samples=3):
    """Visualize data after going through the dataset pipeline."""
    print("\n" + "="*80)
    print("STAGE 2: AFTER DATASET PREPROCESSING (Normalization + Augmentation)")
    print("="*80)
    
    # Create dataset instances
    print("\n--- Creating TRAIN dataset (with augmentation) ---")
    train_dataset = BayernForestHeightDataset(
        root=data_dir,
        split="train",
        mean=0.0,
        std=1.0,
        pad=5,  # Same as config
        seed=0
    )
    
    print("\n--- Creating VAL dataset (no augmentation) ---")
    val_dataset = BayernForestHeightDataset(
        root=data_dir,
        split="val",
        mean=0.0,
        std=1.0,
        pad=None,  # No augmentation for val
        seed=0
    )
    
    # Get samples from train (augmented)
    print("\n--- Sampling from TRAIN dataset (augmented) ---")
    train_samples = []
    for i in range(num_samples):
        rgb, ndsm = train_dataset[i]
        train_samples.append((rgb.numpy(), ndsm.numpy()))
        print(f"Train sample {i}: RGB shape={rgb.shape}, NDSM shape={ndsm.shape}")
        print(f"  RGB range: [{rgb.min():.2f}, {rgb.max():.2f}]")
        print(f"  NDSM range: [{ndsm.min():.2f}, {ndsm.max():.2f}]")
    
    # Get samples from val (no augmentation)
    print("\n--- Sampling from VAL dataset (no augmentation) ---")
    val_samples = []
    for i in range(num_samples):
        rgb, ndsm = val_dataset[i]
        val_samples.append((rgb.numpy(), ndsm.numpy()))
        print(f"Val sample {i}: RGB shape={rgb.shape}, NDSM shape={ndsm.shape}")
        print(f"  RGB range: [{rgb.min():.2f}, {rgb.max():.2f}]")
        print(f"  NDSM range: [{ndsm.min():.2f}, {ndsm.max():.2f}]")
    
    # Visualize train samples
    fig = plt.figure(figsize=(18, 5 * num_samples))
    gs = gridspec.GridSpec(num_samples, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    for i in range(num_samples):
        rgb, ndsm = train_samples[i]
        
        # RGB (normalized) - denormalize for visualization
        ax_rgb = fig.add_subplot(gs[i, 0])
        rgb_denorm = denormalize_rgb(rgb, 0.0, 1.0)
        # Clip to valid range and convert to uint8
        rgb_vis = np.clip(rgb_denorm, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        ax_rgb.imshow(rgb_vis)
        ax_rgb.set_title(f"Train {i}: RGB (normalized)\nTensor range: [{rgb.min():.2f}, {rgb.max():.2f}]")
        ax_rgb.axis('off')
        
        # NDSM (height map)
        ax_ndsm = fig.add_subplot(gs[i, 1])
        im_ndsm = ax_ndsm.imshow(ndsm, cmap='viridis')
        ax_ndsm.set_title(f"Train {i}: NDSM/Height\nRange: [{ndsm.min():.2f}, {ndsm.max():.2f}]m")
        ax_ndsm.axis('off')
        plt.colorbar(im_ndsm, ax=ax_ndsm, fraction=0.046, pad=0.04)
        
        # Overlay
        ax_overlay = fig.add_subplot(gs[i, 2])
        rgb_norm_vis = (rgb_vis / 255.0)
        ax_overlay.imshow(rgb_norm_vis)
        im_overlay = ax_overlay.imshow(ndsm, cmap='jet', alpha=0.5)
        ax_overlay.set_title(f"Train {i}: RGB + Height Overlay")
        ax_overlay.axis('off')
        plt.colorbar(im_overlay, ax=ax_overlay, fraction=0.046, pad=0.04)
        
        # Histogram of height values
        ax_hist = fig.add_subplot(gs[i, 3])
        ax_hist.hist(ndsm.flatten(), bins=50, color='green', alpha=0.7, edgecolor='black')
        ax_hist.set_title(f"Train {i}: Height Distribution")
        ax_hist.set_xlabel("Height (m)")
        ax_hist.set_ylabel("Pixel count")
        ax_hist.grid(True, alpha=0.3)
    
    plt.suptitle("Stage 2: TRAIN Dataset (with augmentation: flips + random crop)", 
                 fontsize=16, fontweight='bold')
    plt.savefig('/work/ammar/sslrp/UCVME/output/bayern_viz_2_train_augmented.png', 
                dpi=150, bbox_inches='tight')
    print(f"\nSaved: output/bayern_viz_2_train_augmented.png")
    plt.close()
    
    # Visualize val samples (no augmentation)
    fig = plt.figure(figsize=(18, 5 * num_samples))
    gs = gridspec.GridSpec(num_samples, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    for i in range(num_samples):
        rgb, ndsm = val_samples[i]
        
        # RGB (normalized) - denormalize for visualization
        ax_rgb = fig.add_subplot(gs[i, 0])
        rgb_denorm = denormalize_rgb(rgb, 0.0, 1.0)
        rgb_vis = np.clip(rgb_denorm, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        ax_rgb.imshow(rgb_vis)
        ax_rgb.set_title(f"Val {i}: RGB (normalized)\nTensor range: [{rgb.min():.2f}, {rgb.max():.2f}]")
        ax_rgb.axis('off')
        
        # NDSM (height map)
        ax_ndsm = fig.add_subplot(gs[i, 1])
        im_ndsm = ax_ndsm.imshow(ndsm, cmap='viridis')
        ax_ndsm.set_title(f"Val {i}: NDSM/Height\nRange: [{ndsm.min():.2f}, {ndsm.max():.2f}]m")
        ax_ndsm.axis('off')
        plt.colorbar(im_ndsm, ax=ax_ndsm, fraction=0.046, pad=0.04)
        
        # Overlay
        ax_overlay = fig.add_subplot(gs[i, 2])
        rgb_norm_vis = (rgb_vis / 255.0)
        ax_overlay.imshow(rgb_norm_vis)
        im_overlay = ax_overlay.imshow(ndsm, cmap='jet', alpha=0.5)
        ax_overlay.set_title(f"Val {i}: RGB + Height Overlay")
        ax_overlay.axis('off')
        plt.colorbar(im_overlay, ax=ax_overlay, fraction=0.046, pad=0.04)
        
        # Histogram
        ax_hist = fig.add_subplot(gs[i, 3])
        ax_hist.hist(ndsm.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax_hist.set_title(f"Val {i}: Height Distribution")
        ax_hist.set_xlabel("Height (m)")
        ax_hist.set_ylabel("Pixel count")
        ax_hist.grid(True, alpha=0.3)
    
    plt.suptitle("Stage 2: VAL Dataset (no augmentation)", 
                 fontsize=16, fontweight='bold')
    plt.savefig('/work/ammar/sslrp/UCVME/output/bayern_viz_3_val_no_augment.png', 
                dpi=150, bbox_inches='tight')
    print(f"Saved: output/bayern_viz_3_val_no_augment.png")
    plt.close()


def visualize_augmentation_effects(data_dir, sample_idx=0, num_augmentations=6):
    """Show the same sample with different random augmentations."""
    print("\n" + "="*80)
    print("STAGE 3: AUGMENTATION VARIATIONS (Same sample, different random transforms)")
    print("="*80)
    
    # Create train dataset
    train_dataset = BayernForestHeightDataset(
        root=data_dir,
        split="train",
        mean=0.0,
        std=1.0,
        pad=5,
        seed=0
    )
    
    # Get multiple augmented versions of the same sample
    print(f"\nGenerating {num_augmentations} augmented versions of sample {sample_idx}")
    
    fig = plt.figure(figsize=(18, 3 * num_augmentations))
    gs = gridspec.GridSpec(num_augmentations, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    for aug_idx in range(num_augmentations):
        # Each call will apply random augmentation
        rgb, ndsm = train_dataset[sample_idx]
        rgb = rgb.numpy()
        ndsm = ndsm.numpy()
        
        print(f"Augmentation {aug_idx}: RGB shape={rgb.shape}, NDSM shape={ndsm.shape}")
        
        # RGB
        ax_rgb = fig.add_subplot(gs[aug_idx, 0])
        rgb_denorm = denormalize_rgb(rgb, 0.0, 1.0)
        rgb_vis = np.clip(rgb_denorm, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        ax_rgb.imshow(rgb_vis)
        ax_rgb.set_title(f"Aug {aug_idx}: RGB")
        ax_rgb.axis('off')
        
        # NDSM
        ax_ndsm = fig.add_subplot(gs[aug_idx, 1])
        im_ndsm = ax_ndsm.imshow(ndsm, cmap='viridis')
        ax_ndsm.set_title(f"Aug {aug_idx}: Height [{ndsm.min():.1f}, {ndsm.max():.1f}]m")
        ax_ndsm.axis('off')
        plt.colorbar(im_ndsm, ax=ax_ndsm, fraction=0.046, pad=0.04)
        
        # Overlay
        ax_overlay = fig.add_subplot(gs[aug_idx, 2])
        ax_overlay.imshow(rgb_vis / 255.0)
        im_overlay = ax_overlay.imshow(ndsm, cmap='jet', alpha=0.5)
        ax_overlay.set_title(f"Aug {aug_idx}: Overlay")
        ax_overlay.axis('off')
        plt.colorbar(im_overlay, ax=ax_overlay, fraction=0.046, pad=0.04)
    
    plt.suptitle(f"Stage 3: Random Augmentations (Sample {sample_idx}, pad=5)\n"
                 f"Each row shows a different random augmentation (flips + crop)",
                 fontsize=16, fontweight='bold')
    plt.savefig('/work/ammar/sslrp/UCVME/output/bayern_viz_4_augmentation_variations.png', 
                dpi=150, bbox_inches='tight')
    print(f"Saved: output/bayern_viz_4_augmentation_variations.png")
    plt.close()


def visualize_pixel_correspondence(data_dir, sample_idx=0):
    """Verify that RGB and NDSM pixels correspond correctly after augmentation."""
    print("\n" + "="*80)
    print("STAGE 4: PIXEL CORRESPONDENCE CHECK")
    print("="*80)
    
    # Create train dataset
    train_dataset = BayernForestHeightDataset(
        root=data_dir,
        split="train",
        mean=0.0,
        std=1.0,
        pad=5,
        seed=42  # Fixed seed for reproducibility
    )
    
    # Get augmented sample
    rgb, ndsm = train_dataset[sample_idx]
    rgb = rgb.numpy()
    ndsm = ndsm.numpy()
    
    print(f"Sample {sample_idx} after augmentation:")
    print(f"  RGB shape: {rgb.shape}, range: [{rgb.min():.2f}, {rgb.max():.2f}]")
    print(f"  NDSM shape: {ndsm.shape}, range: [{ndsm.min():.2f}, {ndsm.max():.2f}]")
    
    # Create detailed visualization with pixel sampling
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.4)
    
    # RGB
    ax_rgb = fig.add_subplot(gs[0, 0])
    rgb_denorm = denormalize_rgb(rgb, 0.0, 1.0)
    rgb_vis = np.clip(rgb_denorm, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    ax_rgb.imshow(rgb_vis)
    ax_rgb.set_title("RGB Image (augmented)")
    ax_rgb.axis('off')
    
    # NDSM
    ax_ndsm = fig.add_subplot(gs[0, 1])
    im_ndsm = ax_ndsm.imshow(ndsm, cmap='viridis')
    ax_ndsm.set_title("Height Map (augmented)")
    ax_ndsm.axis('off')
    plt.colorbar(im_ndsm, ax=ax_ndsm, fraction=0.046, pad=0.04)
    
    # Overlay
    ax_overlay = fig.add_subplot(gs[0, 2])
    ax_overlay.imshow(rgb_vis / 255.0)
    im_overlay = ax_overlay.imshow(ndsm, cmap='jet', alpha=0.5)
    ax_overlay.set_title("RGB + Height Overlay")
    ax_overlay.axis('off')
    plt.colorbar(im_overlay, ax=ax_overlay, fraction=0.046, pad=0.04)
    
    # Sample specific regions to verify correspondence
    ax_regions = fig.add_subplot(gs[0, 3])
    ax_regions.imshow(rgb_vis / 255.0)
    
    # Sample 5 random points
    np.random.seed(42)
    sample_points = []
    for _ in range(5):
        y = np.random.randint(20, 236)
        x = np.random.randint(20, 236)
        sample_points.append((y, x))
        
        # Draw circle and label with height value
        circle = plt.Circle((x, y), 8, color='red', fill=False, linewidth=2)
        ax_regions.add_patch(circle)
        height_val = ndsm[y, x]
        ax_regions.text(x, y-12, f"{height_val:.1f}m", 
                       color='white', fontsize=9, ha='center',
                       bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    
    ax_regions.set_title("Sampled Points with Height Values")
    ax_regions.axis('off')
    
    # Bottom row: Show cross-sections
    # Horizontal cross-section
    ax_cross_h = fig.add_subplot(gs[1, :2])
    mid_row = ndsm.shape[0] // 2
    ax_cross_h.plot(ndsm[mid_row, :], 'g-', linewidth=2, label='Height profile')
    ax_cross_h.set_title(f"Horizontal Cross-section (row {mid_row})")
    ax_cross_h.set_xlabel("Column (pixel)")
    ax_cross_h.set_ylabel("Height (m)")
    ax_cross_h.grid(True, alpha=0.3)
    ax_cross_h.legend()
    
    # Vertical cross-section
    ax_cross_v = fig.add_subplot(gs[1, 2:])
    mid_col = ndsm.shape[1] // 2
    ax_cross_v.plot(ndsm[:, mid_col], 'b-', linewidth=2, label='Height profile')
    ax_cross_v.set_title(f"Vertical Cross-section (col {mid_col})")
    ax_cross_v.set_xlabel("Row (pixel)")
    ax_cross_v.set_ylabel("Height (m)")
    ax_cross_v.grid(True, alpha=0.3)
    ax_cross_v.legend()
    
    plt.suptitle(f"Stage 4: Pixel Correspondence Check (Sample {sample_idx})\n"
                 f"Verifying RGB pixels align with correct height values",
                 fontsize=16, fontweight='bold')
    plt.savefig('/work/ammar/sslrp/UCVME/output/bayern_viz_5_pixel_correspondence.png', 
                dpi=150, bbox_inches='tight')
    print(f"Saved: output/bayern_viz_5_pixel_correspondence.png")
    plt.close()


def compare_augmented_vs_original(data_dir, sample_idx=0):
    """Compare the same sample with and without augmentation."""
    print("\n" + "="*80)
    print("STAGE 5: AUGMENTATION COMPARISON (Same sample, with vs without augmentation)")
    print("="*80)
    
    # No augmentation
    dataset_no_aug = BayernForestHeightDataset(
        root=data_dir,
        split="val",  # Val has no augmentation
        mean=0.0,
        std=1.0,
        pad=None,
        seed=0
    )
    
    # With augmentation
    dataset_with_aug = BayernForestHeightDataset(
        root=data_dir,
        split="train",
        mean=0.0,
        std=1.0,
        pad=5,
        seed=42  # Fixed seed
    )
    
    # Get the same underlying sample (need to map indices)
    # For simplicity, just get first sample from each
    rgb_no_aug, ndsm_no_aug = dataset_no_aug[0]
    rgb_with_aug, ndsm_with_aug = dataset_with_aug[0]
    
    rgb_no_aug = rgb_no_aug.numpy()
    ndsm_no_aug = ndsm_no_aug.numpy()
    rgb_with_aug = rgb_with_aug.numpy()
    ndsm_with_aug = ndsm_with_aug.numpy()
    
    print(f"No augmentation: RGB {rgb_no_aug.shape}, NDSM {ndsm_no_aug.shape}")
    print(f"With augmentation: RGB {rgb_with_aug.shape}, NDSM {ndsm_with_aug.shape}")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: No augmentation
    rgb_vis_no_aug = np.clip(denormalize_rgb(rgb_no_aug, 0.0, 1.0), 0, 255).astype(np.uint8).transpose(1, 2, 0)
    axes[0, 0].imshow(rgb_vis_no_aug)
    axes[0, 0].set_title("NO AUGMENTATION: RGB")
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(ndsm_no_aug, cmap='viridis')
    axes[0, 1].set_title(f"NO AUGMENTATION: Height\n[{ndsm_no_aug.min():.1f}, {ndsm_no_aug.max():.1f}]m")
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    axes[0, 2].imshow(rgb_vis_no_aug / 255.0)
    im1_overlay = axes[0, 2].imshow(ndsm_no_aug, cmap='jet', alpha=0.5)
    axes[0, 2].set_title("NO AUGMENTATION: Overlay")
    axes[0, 2].axis('off')
    plt.colorbar(im1_overlay, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Row 2: With augmentation
    rgb_vis_with_aug = np.clip(denormalize_rgb(rgb_with_aug, 0.0, 1.0), 0, 255).astype(np.uint8).transpose(1, 2, 0)
    axes[1, 0].imshow(rgb_vis_with_aug)
    axes[1, 0].set_title("WITH AUGMENTATION: RGB\n(flips + random crop)")
    axes[1, 0].axis('off')
    
    im2 = axes[1, 1].imshow(ndsm_with_aug, cmap='viridis')
    axes[1, 1].set_title(f"WITH AUGMENTATION: Height\n[{ndsm_with_aug.min():.1f}, {ndsm_with_aug.max():.1f}]m")
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    axes[1, 2].imshow(rgb_vis_with_aug / 255.0)
    im2_overlay = axes[1, 2].imshow(ndsm_with_aug, cmap='jet', alpha=0.5)
    axes[1, 2].set_title("WITH AUGMENTATION: Overlay")
    axes[1, 2].axis('off')
    plt.colorbar(im2_overlay, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.suptitle("Stage 5: Augmentation Comparison\n"
                 "Top: Original | Bottom: Augmented (flips + pad/crop)",
                 fontsize=16, fontweight='bold')
    plt.savefig('/work/ammar/sslrp/UCVME/output/bayern_viz_6_augmentation_comparison.png', 
                dpi=150, bbox_inches='tight')
    print(f"Saved: output/bayern_viz_6_augmentation_comparison.png")
    plt.close()


def check_alignment(data_dir, sample_idx=0):
    """Detailed check that RGB and NDSM are properly aligned after augmentation."""
    print("\n" + "="*80)
    print("STAGE 6: ALIGNMENT VERIFICATION")
    print("="*80)
    
    # Create dataset with fixed seed for reproducibility
    dataset = BayernForestHeightDataset(
        root=data_dir,
        split="train",
        mean=0.0,
        std=1.0,
        pad=5,
        seed=123
    )
    
    rgb, ndsm = dataset[sample_idx]
    rgb = rgb.numpy()
    ndsm = ndsm.numpy()
    
    print(f"Checking alignment for sample {sample_idx}")
    print(f"RGB shape: {rgb.shape}, NDSM shape: {ndsm.shape}")
    
    # Create grid visualization
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    rgb_vis = np.clip(denormalize_rgb(rgb, 0.0, 1.0), 0, 255).astype(np.uint8).transpose(1, 2, 0)
    
    # Full images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb_vis)
    ax1.set_title("RGB (Full)")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(ndsm, cmap='viridis')
    ax2.set_title("Height (Full)")
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(rgb_vis / 255.0)
    im3 = ax3.imshow(ndsm, cmap='jet', alpha=0.5)
    ax3.set_title("Overlay (Full)")
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Zoomed regions
    zoom_size = 64
    center_y, center_x = 128, 128
    y_start = center_y - zoom_size // 2
    y_end = center_y + zoom_size // 2
    x_start = center_x - zoom_size // 2
    x_end = center_x + zoom_size // 2
    
    rgb_zoom = rgb_vis[y_start:y_end, x_start:x_end]
    ndsm_zoom = ndsm[y_start:y_end, x_start:x_end]
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(rgb_zoom)
    ax4.set_title(f"RGB (Zoomed {zoom_size}x{zoom_size})")
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(ndsm_zoom, cmap='viridis')
    ax5.set_title(f"Height (Zoomed {zoom_size}x{zoom_size})")
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(rgb_zoom / 255.0)
    im6 = ax6.imshow(ndsm_zoom, cmap='jet', alpha=0.5)
    ax6.set_title(f"Overlay (Zoomed {zoom_size}x{zoom_size})")
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    
    # Add grid lines to zoomed overlay
    for i in range(0, zoom_size, 8):
        ax6.axhline(i, color='white', linewidth=0.5, alpha=0.3)
        ax6.axvline(i, color='white', linewidth=0.5, alpha=0.3)
    
    plt.suptitle("Stage 6: Alignment Verification\n"
                 "Checking that RGB pixels correspond to correct height values",
                 fontsize=16, fontweight='bold')
    plt.savefig('/work/ammar/sslrp/UCVME/output/bayern_viz_7_alignment_check.png', 
                dpi=150, bbox_inches='tight')
    print(f"Saved: output/bayern_viz_7_alignment_check.png")
    plt.close()
    
    # Print some pixel values for manual verification
    print("\n--- Sample pixel values for manual verification ---")
    print("Format: (row, col) -> RGB=[R,G,B], Height=X.XX m")
    for y, x in sample_points[:3]:
        rgb_pixel = rgb_vis[y, x]
        height_pixel = ndsm[y, x]
        print(f"  ({y:3d}, {x:3d}) -> RGB={rgb_pixel}, Height={height_pixel:.2f}m")


def main():
    # Data directory (check both host and Docker paths)
    data_dir_options = [
        "/workspace/data/Bayern_forest_height",  # Docker mount
        "/work/ammar/sslrp/data/Bayern_forest_height"  # Host path
    ]
    
    data_dir = None
    for path in data_dir_options:
        if os.path.exists(path):
            data_dir = path
            break
    
    if data_dir is None:
        print(f"ERROR: Data directory not found in any of these locations:")
        for path in data_dir_options:
            print(f"  - {path}")
        return
    
    # Create output directory if needed
    output_dir = "/work/ammar/sslrp/UCVME/output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("BAYERN FOREST HEIGHT DATA PIPELINE VISUALIZATION")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Run all visualizations
    try:
        visualize_raw_data(data_dir, num_samples=3)
        visualize_dataset_pipeline(data_dir, num_samples=3)
        visualize_augmentation_effects(data_dir, sample_idx=0, num_augmentations=6)
        visualize_pixel_correspondence(data_dir, sample_idx=0)
        compare_augmented_vs_original(data_dir, sample_idx=0)
        
        print("\n" + "="*80)
        print("VISUALIZATION COMPLETE!")
        print("="*80)
        print("\nGenerated visualizations:")
        print("  1. output/bayern_viz_1_raw.png - Raw data from HDF5")
        print("  2. output/bayern_viz_2_train_augmented.png - Train samples (with augmentation)")
        print("  3. output/bayern_viz_3_val_no_augment.png - Val samples (no augmentation)")
        print("  4. output/bayern_viz_4_augmentation_variations.png - Multiple augmentations of same sample")
        print("  5. output/bayern_viz_5_pixel_correspondence.png - Detailed pixel correspondence check")
        print("  6. output/bayern_viz_6_augmentation_comparison.png - Side-by-side comparison")
        print("\nAll visualizations saved to output/ directory.")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
