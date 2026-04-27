# CUDA Batch Signal Smoother

GPU-accelerated Gaussian smoothing over a large batch of 1-D signals.  
A custom CUDA kernel processes **200 noisy signals × 1 024 samples** in a
single batched launch, writing back smoothed float32 binary files.

## Assignment Requirement Coverage

| Requirement | Where |
|---|---|
| GPU computation | Custom CUDA kernel (`processKernel` → `gaussianSmoothKernel`) in `src/batchSignalSmoother.cu` |
| Large dataset | Auto-generated synthetic dataset of **200 signals** (each 1 024 float32 samples) |
| Proof of execution | Smoothed binaries in `data/output/` and execution log in `data/output/run_log.txt` |

## Project Structure

```
cuda-signal-smoother/
├── src/
│   ├── batchSignalSmoother.cu   # CUDA kernel + host driver
│   └── generate_dataset.py      # synthetic dataset generator
├── data/
│   ├── input/                   # signal_0000.bin … signal_0199.bin  (generated)
│   └── output/                  # smoothed_0000.bin … + run_log.txt  (generated)
├── bin/                         # compiled executable (generated)
├── Makefile
├── run.sh
├── INSTALL
└── README.md
```

## Quick Start

```bash
chmod +x run.sh
./run.sh
```

`run.sh` does three things in order:

1. **Generate** — runs `generate_dataset.py` to create 200 noisy `.bin` files in `data/input/`
2. **Build** — compiles `batchSignalSmoother.cu` with `nvcc`
3. **Process** — runs the binary, smoothing all signals on the GPU

## How It Works

### Dataset (`generate_dataset.py`)

Each synthetic signal is built from 2–3 superimposed sine waves (random
frequency and phase) with additive white Gaussian noise (σ = 0.25).  
Signals are saved as flat `float32` binary files — no external Python
dependencies required.

### CUDA Kernel (`batchSignalSmoother.cu`)

```
gridDim  = (⌈signalLen / 256⌉, numSignals)
blockDim = (256, 1)
```

Each CUDA thread is responsible for exactly one output sample.  
It applies a **Gaussian convolution window** (radius = 12, σ = 4 samples)
whose coefficients are stored in `__constant__` memory for fast broadcast.
Boundary samples are handled with mirror-padding so no output is clamped.

All signals are packed into a single contiguous device allocation to
maximise memory-transfer efficiency (one `cudaMemcpy` in, one out).

### Output

| File | Description |
|---|---|
| `data/output/smoothed_NNNN.bin` | Smoothed float32 signal (same length as input) |
| `data/output/run_log.txt` | Timestamp, dimensions, kernel time, throughput |

### Example `run_log.txt`

```
=== CUDA Batch Signal Smoother — Run Log ===
Timestamp             : Mon Apr 28 10:42:17 2025
Input directory       : data/input
Output directory      : data/output
Signals processed     : 200
Samples per signal    : 1024
Total samples         : 204800
Gaussian radius       : 12 samples
Gaussian sigma        : 4.0 samples
GPU kernel time (ms)  : 0.847
Throughput (Msamples/s): 241.80
```

## Manual Commands

```bash
# 1. Generate dataset
python3 src/generate_dataset.py --output data/input --count 200 --length 1024 --noise 0.25

# 2. Build
make

# 3. Run
./bin/batch_signal_smoother data/input data/output
```

## Tuning

Edit the constants at the top of `src/batchSignalSmoother.cu`:

| Constant | Default | Effect |
|---|---|---|
| `KERNEL_RADIUS` | 12 | Half-width of Gaussian window (larger = smoother) |
| `GAUSS_SIGMA` | 4.0 | Standard deviation of the Gaussian (samples) |
| `BLOCK_SIZE` | 256 | CUDA threads per block |
