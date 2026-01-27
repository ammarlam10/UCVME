# Training Speed Optimization Recommendations

## Current Setup Analysis

### Available Resources
- **GPUs**: 6 total
  - GPUs 0-1: P100 (16GB each) - Available
  - GPU 2: V100S (32GB) - Available
  - GPU 3: V100S (32GB) - Currently in use
  - GPU 4: P100 (16GB) - Available
  - GPU 5: V100 (32GB) - Currently in use
- **RAM**: 377GB total, 366GB available
- **CPU**: 48 cores available

### Current Configuration
- **GPUs used**: 2 (GPUs 3 and 5)
- **Batch size**: 128 (64 per GPU)
- **num_workers**: 0 (disabled due to storage errors)
- **GPU utilization**: 15-18% (very low!)

### Bottlenecks Identified
1. **Data loading bottleneck**: `num_workers=0` means single-threaded data loading
2. **Low GPU utilization**: Only 15-18% suggests GPU is waiting for data
3. **Limited parallelism**: Only using 2 out of 4 available high-memory GPUs
4. **DataParallel overhead**: Using DataParallel has some overhead, but acceptable for 2-4 GPUs

## Optimization Recommendations

### Option 1: Use 4 V100 GPUs (RECOMMENDED)
**GPUs**: 2, 3, 4, 5 (all V100/V100S 32GB)
**Configuration**:
- `batch_size`: 256 (64 per GPU)
- `num_workers`: 4-8 (try with increased shared memory)
- `--gpus '"device=2,3,4,5"'`
- `--shm-size`: 32g (increased shared memory for workers)

**Expected speedup**: ~2-3x faster
- 2x from more GPUs
- Additional speedup from parallel data loading

### Option 2: Use 4 V100 GPUs with Larger Batch
**GPUs**: 2, 3, 4, 5
**Configuration**:
- `batch_size`: 512 (128 per GPU - if memory allows)
- `num_workers`: 4-8
- `--gpus '"device=2,3,4,5"'`
- `--shm-size`: 32g

**Expected speedup**: ~2-3x faster, better GPU utilization

### Option 3: Use All 6 GPUs (Mixed - Less Recommended)
**GPUs**: 0, 1, 2, 3, 4, 5
**Note**: Mixed GPU types (P100 16GB + V100 32GB) will be limited by slower GPUs
**Expected speedup**: ~2-2.5x (limited by P100 performance)

## Recommended Configuration

I recommend **Option 1**: Use GPUs 2, 3, 4, 5 with:
- `batch_size`: 256
- `num_workers`: 4-8
- Docker command with `--gpus '"device=2,3,4,5"'` and `--shm-size 32g`

### Why This Works:
1. **More GPUs**: 4x V100 GPUs provide better parallelization
2. **Parallel data loading**: `num_workers=4-8` allows CPU to prefetch data while GPU computes
3. **Larger batch size**: Better GPU utilization, more stable gradients
4. **Increased shared memory**: Needed for DataLoader workers

### Potential Issues:
- If `num_workers > 0` still causes storage errors, we may need to:
  - Keep `num_workers=0` but use more GPUs (still 2x speedup)
  - Try `num_workers=2` first, then increase if stable
  - Ensure `/dev/shm` is large enough (32GB recommended)

## Docker Command for Optimized Training

```bash
docker run --rm \
  -v /work/ammar/sslrp/data/So2Sat_POP:/workspace/data/So2Sat_POP \
  -v /work/ammar/sslrp/UCVME:/workspace \
  -v /work/ammar/sslrp/UCVME/output:/workspace/output \
  -w /workspace \
  --gpus '"device=2,3,4,5"' \
  --shm-size=32g \
  ucvme:latest \
  python3 ucvme_age.py \
    --config=/workspace/configs/so2sat_pop_efficientnetb0_20percent_optimized.yaml \
    --output=/workspace/output/so2sat_pop_effnetb0_20percent_optimized
```

## Expected Performance Improvement

**Current**:
- 2 GPUs, batch_size=128, num_workers=0
- ~1497 iterations per epoch
- Low GPU utilization (15-18%)
- Estimated: ~30-45 minutes per epoch

**Optimized**:
- 4 GPUs, batch_size=256, num_workers=4
- ~748 iterations per epoch (fewer iterations due to larger batch)
- Higher GPU utilization (60-80% expected)
- Estimated: **~10-15 minutes per epoch** (2-3x speedup)

Total training time (30 epochs):
- Current: ~15-22 hours
- Optimized: **~5-7.5 hours** (3x faster)
