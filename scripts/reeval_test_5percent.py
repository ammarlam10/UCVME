#!/usr/bin/env python3
"""
Re-evaluate test MAE and R² for a single run using best.pt.
Uses same config/dataset as training. Run from repo root.
Usage:
  python scripts/reeval_test_5percent.py
  # Or with custom paths:
  DATA_DIR=/path/to/So2Sat_POP OUTPUT_DIR=output/so2sat_pop_efficientnetb0_5percent_fixed python scripts/reeval_test_5percent.py
"""
import os
import sys
import torch
import numpy as np
import yaml
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Run from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import datasets
import models
import utils


def main():
    output_dir = os.environ.get("OUTPUT_DIR", os.path.join(REPO_ROOT, "output", "so2sat_pop_efficientnetb0_5percent_fixed"))
    data_dir = os.environ.get("DATA_DIR", "/work/ammar/sslrp/data/So2Sat_POP")
    config_path = os.path.join(REPO_ROOT, "configs", "so2sat_pop_efficientnetb0_5percent_fixed.yaml")

    if not os.path.isdir(data_dir):
        raise SystemExit(f"Data dir not found: {data_dir}. Set DATA_DIR.")
    if not os.path.isdir(output_dir):
        raise SystemExit(f"Output dir not found: {output_dir}. Set OUTPUT_DIR.")
    best_pt = os.path.join(output_dir, "best.pt")
    if not os.path.isfile(best_pt):
        raise SystemExit(f"best.pt not found: {best_pt}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    y_mean = float(cfg["target"]["y_mean"])
    y_std = float(cfg["target"]["y_std"])
    batch_size = int(cfg["training"].get("batch_size", 80))
    target_column = cfg["data"]["target_column"]
    image_dir = cfg["data"].get("image_dir", "So2Sat_POP_Part1")
    file_list_name = cfg["data"].get("file_list_name", "FileList.csv")
    dataset_name = cfg["data"]["dataset_name"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_class = datasets.get_dataset(dataset_name)

    # Same kwargs as in ucvme_age.py (without preload for speed)
    dataset_kwargs_base = {
        "target_type": [target_column],
        "mean": 0.0,
        "std": 1.0,
        "image_dir": image_dir,
        "file_list_name": file_list_name,
    }
    if dataset_name == "so2sat_pop_custom":
        dataset_kwargs_base["normalize_mean"] = y_mean
        dataset_kwargs_base["normalize_std"] = y_std

    print("Computing image mean/std from train set...")
    mean, std = utils.get_mean_and_std(dataset_class(root=data_dir, split="train", **dataset_kwargs_base))
    kwargs_test = {
        "target_type": [target_column],
        "mean": mean,
        "std": std,
        "image_dir": image_dir,
        "file_list_name": file_list_name,
    }
    if dataset_name == "so2sat_pop_custom":
        kwargs_test["normalize_mean"] = y_mean
        kwargs_test["normalize_std"] = y_std

    print(f"Loading best model from {best_pt} ...")
    checkpoint = torch.load(best_pt, map_location=device)
    model = models.efficientnetb0_unc(pretrained=False, drp_p=float(cfg["model"].get("drp_p", 0.05)))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(device)
    model.eval()

    model_1 = models.efficientnetb0_unc(pretrained=False, drp_p=float(cfg["model"].get("drp_p", 0.05)))
    model_1 = torch.nn.DataParallel(model_1)
    model_1.load_state_dict(checkpoint["state_dict_1"], strict=False)
    model_1.to(device)
    model_1.eval()

    test_dataset = dataset_class(root=data_dir, split="test", **kwargs_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda")
    )

    samp_fq = int(cfg.get("ssl", {}).get("samp_fq", 5))
    yhat_list = []
    y_list = []

    with torch.no_grad():
        for X, outcome in test_loader:
            X = X.to(device)
            y_list.append(outcome.numpy() if isinstance(outcome, np.ndarray) else outcome.cpu().numpy())
            mean_preds = []
            for _ in range(samp_fq):
                mean_0, _ = model(X)
                mean_1, _ = model_1(X)
                mean_preds.append((mean_0 + mean_1) / 2)
            mean_pred = torch.stack(mean_preds, dim=0).mean(dim=0)
            yhat = mean_pred.view(-1).cpu().numpy() * y_std + y_mean
            yhat_list.append(yhat)

    yhat = np.concatenate(yhat_list)
    y = np.concatenate([yy.flatten() if hasattr(yy, "flatten") else np.atleast_1d(yy) for yy in y_list])

    r2 = r2_score(y, yhat)
    mae = mean_absolute_error(y, yhat)
    rmse = np.sqrt(mean_squared_error(y, yhat))

    print()
    print("=" * 60)
    print("CORRECTED TEST METRICS (best.pt)")
    print("=" * 60)
    print(f"Run: {output_dir}")
    print(f"Test samples: {len(y)}")
    print(f"Test R²:      {r2:.4f}")
    print(f"Test MAE:     {mae:.2f}")
    print(f"Test RMSE:    {rmse:.2f}")
    print("=" * 60)

    # Append to log.csv so the corrected values are recorded
    log_path = os.path.join(output_dir, "log.csv")
    with open(log_path, "a") as f:
        f.write("\n# Corrected test metrics (reeval_test_5percent.py):\n")
        f.write(f"# Test R2: {r2:.4f}, Test MAE: {mae:.2f}, Test RMSE: {rmse:.2f}\n")
    print(f"Appended corrected metrics to {log_path}")


if __name__ == "__main__":
    main()
