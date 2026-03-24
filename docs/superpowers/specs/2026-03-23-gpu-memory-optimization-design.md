# GPU Memory Optimization: Semaphore Pool + Gradient Checkpointing

**Date:** 2026-03-23
**Status:** Approved

## Problem

Running 10-15 parallel clients with large models (ResNet-18, DenseNet-121) on an 8 GB GPU causes CUDA OOM. The current parallel path (`_run_clients_parallel`) deepcopies one full model per thread onto GPU simultaneously. ResNet-18 needs ~500 MB per training client (weights + gradients + optimizer state + activations), so 14 concurrent clients require ~7 GB — exceeding available VRAM.

## Solution

Two complementary changes:

### 1. GPU Semaphore Pool

Decouple thread count from GPU concurrency. All threads remain alive for CPU parallelism (data loading, numpy work), but a `threading.Semaphore` gates how many can hold a GPU model simultaneously.

**Thread lifecycle:**
1. Thread starts, does CPU prep
2. `semaphore.acquire()` — blocks if GPU is full
3. Deepcopy model, move to GPU, train, extract params to CPU (numpy)
4. `model.cpu()`, `del model`, `torch.cuda.empty_cache()`
5. `semaphore.release()` — next waiting thread proceeds
6. Return results

**Auto-concurrency calculation:**
- `_estimate_per_client_gpu_bytes(model, use_amp, use_gradient_checkpointing)` estimates per-client GPU cost:
  - Weights: `sum(p.numel() * p.element_size())`
  - Gradients: same as weights
  - Optimizer state: 1x weights (SGD+momentum)
  - Activations: 6x weights (1.5x with checkpointing, further 0.6x with AMP)
  - Buffers: BatchNorm running stats
- `_max_gpu_concurrent(model, device, use_amp, use_gradient_checkpointing)`:
  - Queries `torch.cuda.get_device_properties` for total VRAM
  - Subtracts `torch.cuda.memory_allocated(device)`
  - Reserves 15% for fragmentation
  - Returns `max(1, available // per_client)`

### 2. Gradient Checkpointing

New `gradient_checkpointing: bool = False` field on `SimulationConfig`.

Function `_enable_gradient_checkpointing(model)` wraps compute-heavy blocks with `torch.utils.checkpoint.checkpoint(use_reentrant=False)`:

- **ResNet-18:** `model.resnet.layer1` through `layer4`
- **DenseNet-121:** Each `denseblock` in `model.densenet.features`
- **CNN/MLP:** Skipped (too small to benefit)
- **Custom models:** Calls `model.enable_gradient_checkpointing()` if the method exists

Applied in `_run_client_round` after model setup, before `_train_client`.

Tradeoff: ~30% more compute, ~3-4x less activation memory. Shifts ResNet-18 per-client cost from ~500 MB to ~180 MB, allowing ~10-12 concurrent clients on 8 GB GPU.

## Files Modified

| File | Change |
|------|--------|
| `simulation/runner.py` | Add estimation functions, semaphore logic, gradient checkpointing, `SimulationConfig.gradient_checkpointing` |
| `app.py` | Add checkbox in advanced GPU section |
| `custom/models/*.py` | Document `enable_gradient_checkpointing()` hook in templates |

## Non-Goals

- CPU training fallback (too slow to be useful)
- Model sharding / pipeline parallelism (overkill for this scale)
- Changes to sequential path (already memory-efficient)
