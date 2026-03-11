"""
Fast visualization of Bayern Forest Height data preprocessing pipeline.

This script loads only a few samples to quickly show how data looks at each stage.
Does NOT load the entire dataset - just reads directly from one HDF5 file.

Usage:
    python scripts/visualize_bayern_pipeline_fast.py
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def apply_augmentation(rgb, ndsm, pad=5, seed=None):
    """
    Apply the same augmentation as BayernForestHeightDataset.
    
    Args:
        rgb: (3, H, W) array
        ndsm: (1, H, W) array
        pad: Padding for random crop
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Random horizontal flip
    if np.random.rand() > 0.5:
        rgb = rgb[:, :, ::-1].copy()
        ndsm = ndsm[:, :, ::-1].copy()
    
    # Random vertical flip
    if np.random.rand() > 0.5:
        rgb = rgb[:, ::-1, :].copy()
        ndsm = ndsm[:, ::-1, :].copy()
    
    # Random padding crop (if specified)
    if pad is not None and pad > 0:
        c, h, w = rgb.shape
        
        # Add padding
        rgb_padded = np.zeros((c, h + 2 * pad, w + 2 * pad), dtype=rgb.dtype)
        rgb_padded[:, pad:-pad, pad:-pad] = rgb
        
        ndsm_padded = np.zeros((1, h + 2 * pad, w + 2 * pad), dtype=ndsm.dtype)
        ndsm_padded[:, pad:-pad, pad:-pad] = ndsm
        
        # Random crop
        i = np.random.randint(0, 2 * pad + 1)
        j = np.random.randint(0, 2 * pad + 1)
        
        rgb = rgb_padded[:, i:i+h, j:j+w].copy()
        ndsm = ndsm_padded[:, i:i+h, j:j+w].copy()
    
    return rgb, ndsm


def visualize_raw_data(h5_file, num_samples=3):
    """Visualize raw data directly from HDF5 file."""
    print("\n" + "="*80)
    print("STAGE 1: RAW DATA FROM HDF5")
    print("="*80)
    print(f"Reading from: {h5_file}")
    
    with h5py.File(h5_file, 'r') as f:
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
            rgb_normalized = rgb_img / 255.0
            ax_overlay.imshow(rgb_normalized)
            im_overlay = ax_overlay.imshow(ndsm_img, cmap='jet', alpha=0.5)
            ax_overlay.set_title(f"Sample {i}: RGB + Height Overlay")
            ax_overlay.axis('off')
            plt.colorbar(im_overlay, ax=ax_overlay, fraction=0.046, pad=0.04)
        
        plt.suptitle("Stage 1: Raw Data from HDF5", fontsize=16, fontweight='bold')
        output_path = '/workspace/ucvme/output/bayern_viz_1_raw.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


def visualize_after_preprocessing(h5_file, num_samples=3):
    """Visualize data after normalization (before augmentation)."""
    print("\n" + "="*80)
    print("STAGE 2: AFTER NORMALIZATION (before augmentation)")
    print("="*80)
    
    with h5py.File(h5_file, 'r') as f:
        rgb = f['rgb'][:num_samples]  # (N, 256, 256, 3)
        ndsm = f['ndsm'][:num_samples]  # (N, 256, 256, 1)
    
    fig = plt.figure(figsize=(18, 5 * num_samples))
    gs = gridspec.GridSpec(num_samples, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    for i in range(num_samples):
        # Transpose to PyTorch format
        rgb_sample = rgb[i].astype(np.float32).transpose(2, 0, 1)  # (3, 256, 256)
        ndsm_sample = ndsm[i].astype(np.float32).transpose(2, 0, 1)  # (1, 256, 256)
        
        # Normalize RGB (mean=0, std=1 as in config)
        rgb_normalized = (rgb_sample - 0.0) / 1.0  # No-op but shows the step
        
        # For visualization, denormalize back
        rgb_vis = np.clip(rgb_normalized, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        ndsm_vis = ndsm_sample[0]  # (256, 256)
        
        print(f"Sample {i}: RGB tensor range [{rgb_normalized.min():.2f}, {rgb_normalized.max():.2f}]")
        print(f"Sample {i}: NDSM range [{ndsm_vis.min():.2f}, {ndsm_vis.max():.2f}]")
        
        # RGB
        ax_rgb = fig.add_subplot(gs[i, 0])
        ax_rgb.imshow(rgb_vis)
        ax_rgb.set_title(f"Sample {i}: RGB (normalized)")
        ax_rgb.axis('off')
        
        # NDSM
        ax_ndsm = fig.add_subplot(gs[i, 1])
        im_ndsm = ax_ndsm.imshow(ndsm_vis, cmap='viridis')
        ax_ndsm.set_title(f"Sample {i}: Height\n[{ndsm_vis.min():.2f}, {ndsm_vis.max():.2f}]m")
        ax_ndsm.axis('off')
        plt.colorbar(im_ndsm, ax=ax_ndsm, fraction=0.046, pad=0.04)
        
        # Overlay
        ax_overlay = fig.add_subplot(gs[i, 2])
        ax_overlay.imshow(rgb_vis / 255.0)
        im_overlay = ax_overlay.imshow(ndsm_vis, cmap='jet', alpha=0.5)
        ax_overlay.set_title(f"Sample {i}: Overlay")
        ax_overlay.axis('off')
        plt.colorbar(im_overlay, ax=ax_overlay, fraction=0.046, pad=0.04)
        
        # Histogram
        ax_hist = fig.add_subplot(gs[i, 3])
        ax_hist.hist(ndsm_vis.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax_hist.set_title(f"Sample {i}: Height Distribution")
        ax_hist.set_xlabel("Height (m)")
        ax_hist.set_ylabel("Pixel count")
        ax_hist.grid(True, alpha=0.3)
    
    plt.suptitle("Stage 2: After Normalization (before augmentation)", 
                 fontsize=16, fontweight='bold')
    output_path = '/workspace/ucvme/output/bayern_viz_2_normalized.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def visualize_augmentation_variations(h5_file, sample_idx=0, num_augmentations=6):
    """Show the same sample with different random augmentations."""
    print("\n" + "="*80)
    print("STAGE 3: AUGMENTATION VARIATIONS (Same sample, different random transforms)")
    print("="*80)
    
    with h5py.File(h5_file, 'r') as f:
        rgb = f['rgb'][sample_idx]  # (256, 256, 3)
        ndsm = f['ndsm'][sample_idx]  # (256, 256, 1)
    
    # Transpose to PyTorch format
    rgb = rgb.astype(np.float32).transpose(2, 0, 1)  # (3, 256, 256)
    ndsm = ndsm.astype(np.float32).transpose(2, 0, 1)  # (1, 256, 256)
    
    # Normalize
    rgb = (rgb - 0.0) / 1.0
    
    print(f"Generating {num_augmentations} augmented versions of sample {sample_idx}")
    
    fig = plt.figure(figsize=(18, 3 * num_augmentations))
    gs = gridspec.GridSpec(num_augmentations, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    for aug_idx in range(num_augmentations):
        # Apply random augmentation
        rgb_aug, ndsm_aug = apply_augmentation(rgb.copy(), ndsm.copy(), pad=5, seed=None)
        
        # For visualization
        rgb_vis = np.clip(rgb_aug, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        ndsm_vis = ndsm_aug[0]  # (256, 256)
        
        print(f"Aug {aug_idx}: RGB range [{rgb_aug.min():.2f}, {rgb_aug.max():.2f}], "
              f"NDSM range [{ndsm_vis.min():.2f}, {ndsm_vis.max():.2f}]")
        
        # RGB
        ax_rgb = fig.add_subplot(gs[aug_idx, 0])
        ax_rgb.imshow(rgb_vis)
        ax_rgb.set_title(f"Aug {aug_idx}: RGB")
        ax_rgb.axis('off')
        
        # NDSM
        ax_ndsm = fig.add_subplot(gs[aug_idx, 1])
        im_ndsm = ax_ndsm.imshow(ndsm_vis, cmap='viridis')
        ax_ndsm.set_title(f"Aug {aug_idx}: Height [{ndsm_vis.min():.1f}, {ndsm_vis.max():.1f}]m")
        ax_ndsm.axis('off')
        plt.colorbar(im_ndsm, ax=ax_ndsm, fraction=0.046, pad=0.04)
        
        # Overlay
        ax_overlay = fig.add_subplot(gs[aug_idx, 2])
        ax_overlay.imshow(rgb_vis / 255.0)
        im_overlay = ax_overlay.imshow(ndsm_vis, cmap='jet', alpha=0.5)
        ax_overlay.set_title(f"Aug {aug_idx}: Overlay")
        ax_overlay.axis('off')
        plt.colorbar(im_overlay, ax=ax_overlay, fraction=0.046, pad=0.04)
    
    plt.suptitle(f"Stage 3: Random Augmentations (Sample {sample_idx}, pad=5)\n"
                 f"Each row shows a different random augmentation (flips + crop)",
                 fontsize=16, fontweight='bold')
    output_path = '/workspace/ucvme/output/bayern_viz_3_augmentation_variations.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def visualize_pixel_correspondence(h5_file, sample_idx=0):
    """Verify that RGB and NDSM pixels correspond correctly after augmentation."""
    print("\n" + "="*80)
    print("STAGE 4: PIXEL CORRESPONDENCE CHECK")
    print("="*80)
    
    with h5py.File(h5_file, 'r') as f:
        rgb = f['rgb'][sample_idx]  # (256, 256, 3)
        ndsm = f['ndsm'][sample_idx]  # (256, 256, 1)
    
    # Transpose and normalize
    rgb = rgb.astype(np.float32).transpose(2, 0, 1)  # (3, 256, 256)
    ndsm = ndsm.astype(np.float32).transpose(2, 0, 1)  # (1, 256, 256)
    rgb = (rgb - 0.0) / 1.0
    
    # Apply augmentation with fixed seed
    rgb_aug, ndsm_aug = apply_augmentation(rgb, ndsm, pad=5, seed=42)
    
    print(f"Sample {sample_idx} after augmentation:")
    print(f"  RGB shape: {rgb_aug.shape}, range: [{rgb_aug.min():.2f}, {rgb_aug.max():.2f}]")
    print(f"  NDSM shape: {ndsm_aug.shape}, range: [{ndsm_aug.min():.2f}, {ndsm_aug.max():.2f}]")
    
    rgb_vis = np.clip(rgb_aug, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    ndsm_vis = ndsm_aug[0]  # (256, 256)
    
    # Create detailed visualization
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.4)
    
    # Full images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb_vis)
    ax1.set_title("RGB (augmented)")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(ndsm_vis, cmap='viridis')
    ax2.set_title("Height (augmented)")
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(rgb_vis / 255.0)
    im3 = ax3.imshow(ndsm_vis, cmap='jet', alpha=0.5)
    ax3.set_title("Overlay (augmented)")
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Sample points with height values
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(rgb_vis / 255.0)
    
    # Sample 5 random points
    np.random.seed(42)
    sample_points = []
    for _ in range(5):
        y = np.random.randint(20, 236)
        x = np.random.randint(20, 236)
        sample_points.append((y, x))
        
        # Draw circle and label with height value
        circle = plt.Circle((x, y), 8, color='red', fill=False, linewidth=2)
        ax4.add_patch(circle)
        height_val = ndsm_vis[y, x]
        ax4.text(x, y-12, f"{height_val:.1f}m", 
                color='white', fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    
    ax4.set_title("Sampled Points with Heights")
    ax4.axis('off')
    
    # Zoomed regions
    zoom_size = 64
    center_y, center_x = 128, 128
    y_start = center_y - zoom_size // 2
    y_end = center_y + zoom_size // 2
    x_start = center_x - zoom_size // 2
    x_end = center_x + zoom_size // 2
    
    rgb_zoom = rgb_vis[y_start:y_end, x_start:x_end]
    ndsm_zoom = ndsm_vis[y_start:y_end, x_start:x_end]
    
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(rgb_zoom)
    ax5.set_title(f"RGB (Zoomed {zoom_size}x{zoom_size})")
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(ndsm_zoom, cmap='viridis')
    ax6.set_title(f"Height (Zoomed)")
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(rgb_zoom / 255.0)
    im7 = ax7.imshow(ndsm_zoom, cmap='jet', alpha=0.5)
    ax7.set_title(f"Overlay (Zoomed)")
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    
    # Add grid to zoomed overlay
    for i in range(0, zoom_size, 8):
        ax7.axhline(i, color='white', linewidth=0.5, alpha=0.3)
        ax7.axvline(i, color='white', linewidth=0.5, alpha=0.3)
    
    # Cross-sections
    ax8 = fig.add_subplot(gs[1, 3])
    mid_row = ndsm_vis.shape[0] // 2
    mid_col = ndsm_vis.shape[1] // 2
    ax8.plot(ndsm_vis[mid_row, :], 'g-', linewidth=2, label=f'Horizontal (row {mid_row})')
    ax8.plot(ndsm_vis[:, mid_col], 'b-', linewidth=2, label=f'Vertical (col {mid_col})')
    ax8.set_title("Height Cross-sections")
    ax8.set_xlabel("Position (pixel)")
    ax8.set_ylabel("Height (m)")
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    
    plt.suptitle(f"Stage 4: Pixel Correspondence Check (Sample {sample_idx})\n"
                 f"Verifying RGB pixels align with correct height values",
                 fontsize=16, fontweight='bold')
    output_path = '/workspace/ucvme/output/bayern_viz_4_pixel_correspondence.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print pixel values for manual verification
    print("\n--- Sample pixel values for manual verification ---")
    print("Format: (row, col) -> RGB=[R,G,B], Height=X.XX m")
    for y, x in sample_points[:3]:
        rgb_pixel = rgb_vis[y, x]
        height_pixel = ndsm_vis[y, x]
        print(f"  ({y:3d}, {x:3d}) -> RGB={rgb_pixel}, Height={height_pixel:.2f}m")


def compare_augmented_vs_original(h5_file, sample_idx=0):
    """Compare the same sample with and without augmentation side-by-side."""
    print("\n" + "="*80)
    print("STAGE 5: AUGMENTATION COMPARISON")
    print("="*80)
    
    with h5py.File(h5_file, 'r') as f:
        rgb = f['rgb'][sample_idx]  # (256, 256, 3)
        ndsm = f['ndsm'][sample_idx]  # (256, 256, 1)
    
    # Transpose and normalize
    rgb = rgb.astype(np.float32).transpose(2, 0, 1)
    ndsm = ndsm.astype(np.float32).transpose(2, 0, 1)
    rgb = (rgb - 0.0) / 1.0
    
    # Original (no augmentation)
    rgb_orig = rgb.copy()
    ndsm_orig = ndsm.copy()
    
    # With augmentation
    rgb_aug, ndsm_aug = apply_augmentation(rgb.copy(), ndsm.copy(), pad=5, seed=99)
    
    # Convert for visualization
    rgb_orig_vis = np.clip(rgb_orig, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    ndsm_orig_vis = ndsm_orig[0]
    
    rgb_aug_vis = np.clip(rgb_aug, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    ndsm_aug_vis = ndsm_aug[0]
    
    print(f"Original: RGB {rgb_orig.shape}, NDSM {ndsm_orig.shape}")
    print(f"Augmented: RGB {rgb_aug.shape}, NDSM {ndsm_aug.shape}")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Original
    axes[0, 0].imshow(rgb_orig_vis)
    axes[0, 0].set_title("ORIGINAL: RGB")
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(ndsm_orig_vis, cmap='viridis')
    axes[0, 1].set_title(f"ORIGINAL: Height\n[{ndsm_orig_vis.min():.1f}, {ndsm_orig_vis.max():.1f}]m")
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    axes[0, 2].imshow(rgb_orig_vis / 255.0)
    im1_overlay = axes[0, 2].imshow(ndsm_orig_vis, cmap='jet', alpha=0.5)
    axes[0, 2].set_title("ORIGINAL: Overlay")
    axes[0, 2].axis('off')
    plt.colorbar(im1_overlay, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Row 2: Augmented
    axes[1, 0].imshow(rgb_aug_vis)
    axes[1, 0].set_title("AUGMENTED: RGB\n(flips + random crop)")
    axes[1, 0].axis('off')
    
    im2 = axes[1, 1].imshow(ndsm_aug_vis, cmap='viridis')
    axes[1, 1].set_title(f"AUGMENTED: Height\n[{ndsm_aug_vis.min():.1f}, {ndsm_aug_vis.max():.1f}]m")
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    axes[1, 2].imshow(rgb_aug_vis / 255.0)
    im2_overlay = axes[1, 2].imshow(ndsm_aug_vis, cmap='jet', alpha=0.5)
    axes[1, 2].set_title("AUGMENTED: Overlay")
    axes[1, 2].axis('off')
    plt.colorbar(im2_overlay, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.suptitle("Stage 5: Augmentation Comparison\n"
                 "Top: Original | Bottom: Augmented (flips + pad/crop)",
                 fontsize=16, fontweight='bold')
    output_path = '/workspace/ucvme/output/bayern_viz_5_augmentation_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


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
    
    # Get first h5 file (we only need one for visualization)
    h5_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
    if not h5_files:
        print(f"ERROR: No .h5 files found in {data_dir}")
        return
    
    h5_file = os.path.join(data_dir, h5_files[0])
    
    # Create output directory
    output_dir = "/workspace/ucvme/output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("BAYERN FOREST HEIGHT DATA PIPELINE VISUALIZATION (FAST)")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Using HDF5 file: {h5_files[0]}")
    print(f"Output directory: {output_dir}")
    
    # Run all visualizations
    try:
        visualize_raw_data(h5_file, num_samples=3)
        visualize_after_preprocessing(h5_file, num_samples=3)
        visualize_augmentation_variations(h5_file, sample_idx=0, num_augmentations=6)
        visualize_pixel_correspondence(h5_file, sample_idx=0)
        compare_augmented_vs_original(h5_file, sample_idx=0)
        
        print("\n" + "="*80)
        print("VISUALIZATION COMPLETE!")
        print("="*80)
        print("\nGenerated visualizations:")
        print("  1. output/bayern_viz_1_raw.png - Raw data from HDF5")
        print("  2. output/bayern_viz_2_normalized.png - After normalization")
        print("  3. output/bayern_viz_3_augmentation_variations.png - Multiple augmentations")
        print("  4. output/bayern_viz_4_pixel_correspondence.png - Detailed alignment check")
        print("  5. output/bayern_viz_5_augmentation_comparison.png - Side-by-side comparison")
        print("\nAll visualizations saved to output/ directory.")
        print("\nYou can verify that:")
        print("  - RGB and NDSM are always transformed together")
        print("  - Same flips are applied to both")
        print("  - Same random crop is applied to both")
        print("  - Pixel correspondence is maintained")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
