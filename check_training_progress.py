#!/usr/bin/env python3
import sys
import os

log_file = '/workspace/output/so2sat_pop_effnetb0_20percent/log.csv'

if not os.path.exists(log_file):
    print(f"Error: Log file not found: {log_file}")
    sys.exit(1)

train_epochs = {}
val_epochs = {}

with open(log_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('Run timestamp') or line.startswith('Starting run') or line.startswith('Resuming'):
            continue
        
        parts = line.split(',')
        if len(parts) < 3:
            continue
        
        try:
            epoch = int(parts[0])
            phase = parts[1]
            
            if phase == 'train' and len(parts) >= 12:
                loss = float(parts[2])
                r2_0 = float(parts[3])
                r2_1 = float(parts[4])
                train_epochs[epoch] = {'loss': loss, 'r2_0': r2_0, 'r2_1': r2_1}
            elif phase == 'val' and len(parts) >= 4:
                loss = float(parts[2])
                r2 = float(parts[3])
                val_epochs[epoch] = {'loss': loss, 'r2': r2}
        except (ValueError, IndexError):
            continue

if not train_epochs:
    print("No training epochs found in log file")
    sys.exit(1)

max_epoch = max(train_epochs.keys())
latest_train = train_epochs[max_epoch]
latest_val = val_epochs.get(max_epoch)

print("="*60)
print(f"TRAINING PROGRESS SUMMARY")
print("="*60)
print(f"\nTotal epochs completed: {max_epoch + 1} out of 30")
print(f"\n=== LATEST EPOCH {max_epoch} ===")
print(f"Train - Loss: {latest_train['loss']:.4f}")
print(f"         R² (model 0): {latest_train['r2_0']:.4f}")
print(f"         R² (model 1): {latest_train['r2_1']:.4f}")

if latest_val:
    print(f"\nVal   - Loss: {latest_val['loss']:.4e}")
    print(f"         R²: {latest_val['r2']:.4f}")
else:
    print(f"\nVal   - Not yet logged for epoch {max_epoch}")

print(f"\n=== ALL EPOCHS SUMMARY ===")
print(f"{'Epoch':<8} {'Train Loss':<12} {'Train R²_0':<12} {'Train R²_1':<12} {'Val Loss':<15} {'Val R²':<10}")
print("-" * 80)

for epoch in sorted(train_epochs.keys()):
    train = train_epochs[epoch]
    val = val_epochs.get(epoch, {})
    val_loss_str = f"{val.get('loss', 0):.4e}" if val else "N/A"
    val_r2_str = f"{val.get('r2', 0):.4f}" if val else "N/A"
    print(f"{epoch:<8} {train['loss']:<12.4f} {train['r2_0']:<12.4f} {train['r2_1']:<12.4f} {val_loss_str:<15} {val_r2_str:<10}")
