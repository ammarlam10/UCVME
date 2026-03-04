#!/usr/bin/env python3
"""
Calculate the CORRECT test MAE by re-evaluating the best model.
This fixes the bug in lines 599-603 of ucvme_age.py
"""

import sys
import os
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tqdm

# Add workspace to path
sys.path.insert(0, '/workspace')

import datasets
import models

def calculate_test_metrics(experiment_name, config):
    """Calculate correct test metrics for an experiment."""
    
    y_mean = config['y_mean']
    y_std = config['y_std']
    data_dir = config['data_dir']
    batch_size = config['batch_size']
    output_dir = config['output_dir']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Loading best model from: {output_dir}")
    
    # Load the best model
    checkpoint_path = os.path.join(output_dir, 'best.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize models
    model = models.efficientnetb0_unc(pretrained=False, drp_p=0.05)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    model_1 = models.efficientnetb0_unc(pretrained=False, drp_p=0.05)
    model_1 = torch.nn.DataParallel(model_1)
    model_1.load_state_dict(checkpoint['state_dict_1'], strict=False)
    model_1.to(device)
    model_1.eval()
    
    print(f"✓ Models loaded (Epoch {checkpoint['epoch']}, Val Loss {checkpoint['best_model_loss']:.4f})")
    
    # Load test dataset
    dataset_class = datasets.So2SatDatasetCustom
    
    # Get image normalization from training data
    print("Calculating image normalization...")
    kwargs_base = {
        "target_type": ["POP"],
        "mean": 0.,
        "std": 1.,
        "image_dir": "So2Sat_POP_Part1",
        "file_list_name": "FileList.csv"
    }
    
    temp_ds = dataset_class(root=data_dir, split="train", **kwargs_base)
    
    # Calculate mean and std from a sample
    sample_size = min(1000, len(temp_ds))
    print(f"Sampling {sample_size} images for normalization...")
    
    sum_img = 0
    sum_sq = 0
    count = 0
    
    for i in range(sample_size):
        img, _ = temp_ds[i]
        sum_img += img.sum()
        sum_sq += (img ** 2).sum()
        count += img.size
    
    mean = sum_img / count
    std = np.sqrt(sum_sq / count - mean ** 2)
    
    print(f"Image normalization: mean={mean:.4f}, std={std:.4f}")
    
    # Create test dataset with proper normalization
    kwargs = {
        "target_type": ["POP"],
        "mean": mean,
        "std": std,
        "image_dir": "So2Sat_POP_Part1",
        "file_list_name": "FileList.csv"
    }
    
    print("\nLoading test dataset...")
    test_dataset = dataset_class(root=data_dir, split="test", **kwargs)
    print(f"✓ Test samples: {len(test_dataset)}")
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    
    # Run evaluation
    print("\nRunning test evaluation...")
    yhat_list = []
    y_list = []
    samp_fq = 5
    
    with torch.no_grad():
        for X, outcome in tqdm.tqdm(test_loader, desc="Evaluating"):
            X = X.to(device)
            y_list.append(outcome.numpy())
            
            # Monte Carlo sampling
            mean_preds = []
            for _ in range(samp_fq):
                mean_0, _ = model(X)
                mean_1, _ = model_1(X)
                mean_preds.append((mean_0 + mean_1) / 2)
            
            # Average predictions
            mean_pred = torch.stack(mean_preds, dim=0).mean(dim=0)
            
            # Denormalize predictions
            yhat = mean_pred.view(-1).cpu().numpy() * y_std + y_mean
            yhat_list.append(yhat)
    
    # Concatenate results
    yhat = np.concatenate(yhat_list)
    y = np.concatenate(y_list)
    
    # Calculate metrics (CORRECTED - no double denormalization!)
    r2 = r2_score(y, yhat)
    mae = mean_absolute_error(y, yhat)
    rmse = mean_squared_error(y, yhat) ** 0.5
    
    print(f"\n{'='*70}")
    print(f"CORRECTED TEST RESULTS")
    print(f"{'='*70}")
    print(f"Samples:          {len(y)}")
    print(f"Target mean:      {y.mean():.2f}")
    print(f"Target std:       {y.std():.2f}")
    print(f"Prediction mean:  {yhat.mean():.2f}")
    print(f"Prediction std:   {yhat.std():.2f}")
    print(f"\nTest R²:          {r2:.4f}")
    print(f"Test MAE:         {mae:.2f}")
    print(f"Test RMSE:        {rmse:.2f}")
    print(f"{'='*70}")
    
    # Show some examples
    print("\nSample predictions (first 15):")
    print(f"{'Predicted':>10} | {'Actual':>8} | {'Error':>8} | {'% Error':>8}")
    print("-" * 50)
    for i in range(min(15, len(y))):
        error = abs(yhat[i] - y[i])
        pct_error = (error / y[i] * 100) if y[i] > 0 else 0
        print(f"{yhat[i]:10.1f} | {y[i]:8.1f} | {error:8.1f} | {pct_error:7.1f}%")
    
    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'n_samples': len(y)
    }


if __name__ == '__main__':
    # Configuration for each experiment
    experiments = {
        '10% Labeled': {
            'y_mean': 1872.5640,
            'y_std': 3398.8516,
            'data_dir': '/workspace/data/So2Sat_POP',
            'batch_size': 64,
            'output_dir': '/workspace/output/so2sat_pop_efficientnetb0_10percent_fixed'
        },
        '5% Labeled': {
            'y_mean': 1872.5640,
            'y_std': 3398.8516,
            'data_dir': '/workspace/data/So2Sat_POP',
            'batch_size': 64,
            'output_dir': '/workspace/output/so2sat_pop_efficientnetb0_5percent_fixed'
        },
        '20% Labeled': {
            'y_mean': 1872.5640,
            'y_std': 3398.8516,
            'data_dir': '/workspace/data/So2Sat_POP',
            'batch_size': 64,
            'output_dir': '/workspace/output/so2sat_pop_efficientnetb0_20percent_fixed'
        }
    }
    
    # Calculate for all experiments
    results = {}
    for exp_name, config in experiments.items():
        try:
            results[exp_name] = calculate_test_metrics(exp_name, config)
        except Exception as e:
            print(f"\nError processing {exp_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY OF CORRECTED TEST RESULTS")
    print(f"{'='*70}")
    for exp_name, metrics in results.items():
        print(f"\n{exp_name}:")
        print(f"  R²:   {metrics['r2']:.4f}")
        print(f"  MAE:  {metrics['mae']:.2f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
