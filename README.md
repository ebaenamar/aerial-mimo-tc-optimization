# Aerial MIMO Tensor Core Optimization

Tensor Core acceleration of Massive MIMO channel equalization in NVIDIA Aerial cuPHY.

## Overview

This repository demonstrates the identification and optimization of the dominant performance bottleneck in the PUSCH (Physical Uplink Shared Channel) receiver pipeline of NVIDIA Aerial CUDA-Accelerated RAN [1]. Through systematic profiling with Nsight Compute, we discovered that the **coefficient application kernel** (`eqMmseSoftDemapKernel_v4`) — a hidden batched complex GEMM of shape `[8×64] × [64×132]` — consumes **81% of the equalization time**, despite the LLR soft-demapper being the perceived bottleneck.

By replacing the scalar FP32 `cuCma` loop (K=64 serial FMAs per thread) with WMMA FP16 Tensor Core operations via `nvcuda::wmma` [2], we achieve:

| Metric | V1 Baseline | V2 WMMA (this work) |
|---|---|---|
| Latency (median) | 0.146 ms | **0.027 ms** |
| Throughput | 487 GFLOP/s | **2,681 GFLOP/s** |
| Speedup | 1.0× | **5.5×** |
| Cosine similarity | — | 0.99999997 |
| Max abs error | — | 0.001289 |
| HMMA TC active | 0.02% | 2.44% |

### Hardware

- GPU: NVIDIA A100 80GB PCIe (108 SMs, SM 8.0) [3]
- CUDA 12.9
- FP16 Tensor Core peak: 312 TFLOPS [3]

### Workload Configuration

| Parameter | Value |
|---|---|
| PRBs | 132 |
| Subcarriers per PRB | 12 |
| MIMO layers | 8 |
| Data symbols per slot | 11 |
| Base station antennas | 64 |
| Total subcarriers | 1,584 |

## Methodology

### Bottleneck Identification

Profiling the three main kernels in the PUSCH RX equalization pipeline:

| Kernel | Latency | % of pipeline | Arithmetic Intensity | Limiter |
|---|---|---|---|---|
| Gram matrix compute | 0.035 ms | 19% | 3.76 FLOP/B | Memory |
| **Coefficient apply** | **0.146 ms** | **81%** | **4.32 FLOP/B** | **Compute (scalar)** |
| LLR soft demapper | ~0.03 ms | ~10% | low | Memory |

The coefficient application kernel uses 132 threads per block (one per SC×symbol pair), each performing 64 serial complex FMAs. This scalar loop structure leaves Tensor Cores completely idle (0.02% HMMA utilization).

### WMMA FP16 Optimization

The coefficient application is reframed as a batched complex GEMM:

```
Y_eq[sc, layer, sym] = sum_ant C[sc, layer, ant] * Y_rx[sc, sym, ant]
```

Mapped to WMMA fragments (m16×n16×k16):
- **A**: Coefficient matrix `[N_LAYERS=8 → pad 16, K=64]` per subcarrier
- **B**: Received signal `[K=64, N_DATA_SYMS=11 → pad 16]` per subcarrier
- **C**: Equalized output `[16, 16]` → extract `[8, 11]`

Complex multiplication is decomposed into 4 real WMMA calls:
```
Re(Y_eq) = Re(C)@Re(Y) - Im(C)@Im(Y)
Im(Y_eq) = Re(C)@Im(Y) + Im(C)@Re(Y)
```

Each warp processes one subcarrier: 4 k-steps × 4 real MMAs = 16 MMA calls. With 1,584 subcarriers mapped to 1,584 blocks (1 warp each), the kernel achieves full GPU utilization.

### Why Not Multi-Warp (V3)?

A V3 variant groups 4 subcarriers per block (4 warps) to increase occupancy. However, V3 matches V2 performance exactly (0.027 ms) because the kernel is **memory-bound** at AI=4.32 FLOP/byte — adding warps does not relieve the L1/L2 bandwidth pressure.

## Files

| File | Description |
|---|---|
| `src/aerial_coef_wmma.cu` | Main benchmark: V1 scalar baseline, V2 WMMA, V3 multi-warp |
| `src/aerial_bottleneck.cu` | Microbenchmark comparing Gram compute vs coefficient apply |

### Build & Run

```bash
nvcc -O3 -arch=sm_80 -o aerial_coef_wmma src/aerial_coef_wmma.cu
./aerial_coef_wmma
```

### Profiling with Nsight Compute

```bash
ncu --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed \
--kernel-name 'regex:coef_apply' \
./aerial_coef_wmma
```

## Results

```
==========================================================
 Aerial coef_apply: Scalar FP32 vs WMMA FP16 Tensor Core
==========================================================
 GPU: NVIDIA A100 80GB PCIe (108 SMs, SM 8.0)
 Config: 132 PRB × 12 SC × 8 layers × 11 syms × 64 ants

Correctness (V2 WMMA vs V1 scalar FP32):
  Cosine similarity : 0.99999997  (threshold >0.999)
  Max abs error     : 0.001289

Performance (median over 500 alternating iterations):
  V1 scalar FP32   [Aerial baseline]  :   0.146 ms   487.4 GFLOP/s
  V2 WMMA 1w/SC    [TC, 1584 blocks]  :   0.027 ms  2680.6 GFLOP/s  speedup 5.50x
  V3 WMMA 4w/blk   [TC,  396 blocks]  :   0.027 ms  2680.6 GFLOP/s  speedup 5.50x

  A100 FP16 TC peak 312 TFLOP/s
  V2 TC utilization: 0.86%  |  V3 TC utilization: 0.86%
```

### NCU Profiling Comparison

| Metric | V1 Scalar | V2 WMMA |
|---|---|---|
| HMMA TC active (%) | 0.02 | 2.44 |
| Warps active (%) | 57.20 | 13.57 |
| L1tex throughput (%) | 93.64 | 55.04 |
| DRAM throughput (%) | 4.42 | 21.18 |

V1 is L1-bound (94%) from shared memory coefficient cache lookups in the scalar loop. V2 shifts the bottleneck to DRAM (21%) by converting compute to Tensor Cores, but remains memory-bound due to the FP32→FP16 conversion overhead.

## Key Insights

1. **The perceived bottleneck was not the real one.** The LLR soft-demapper with its constant-memory LUT stalls appeared to be the bottleneck, but profiling revealed the coefficient application kernel consumes 81% of equalization time.

2. **Not all kernels benefit from Tensor Cores.** The Gram matrix `[8×64]@[64×8]` has M=N=8, below the WMMA minimum of m=16. The LLR computation is intrinsically scalar (1 thread/symbol, LUT chain). Only the coefficient application satisfies both conditions: (1) underlying GEMM structure, (2) dimensions ≥ WMMA fragment minimum.

3. **Memory-bound regime limits TC utilization.** At AI=4.32 FLOP/byte, the kernel cannot saturate the 312 TFLOPS TC peak. In production Aerial with native FP16 data paths (no FP32→FP16 conversion), the arithmetic intensity improves and higher TC utilization is expected.

## Future Work

- **Channel estimation filter** (`channel_est.cu`): Identified a second hidden GEMM `[64×96] @ [96×4]` in the frequency interpolation kernel, batched over 14 OFDM symbols and multiple UEs. WMMA FP16 optimization is viable (M=64 ≥ 16, K=96 ≥ 16, N=4 pad→16).
- **FP16 native data path**: Eliminate FP32→FP16 conversion by leveraging Aerial's `TCompute=__half` template path.
- **`cp.async` pipelining**: Overlap shared memory loads with computation to reduce L1 pressure (currently 55%).

## References

1. NVIDIA, "cuPHY Developer Guide — Aerial CUDA-Accelerated RAN," [Online]. Available: https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/cubb/cuphy_developer_guide/index.html

2. NVIDIA, "Programming Tensor Cores in CUDA 9," [Online]. Available: https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/

3. NVIDIA, "NVIDIA A100 Tensor Core GPU Architecture," Whitepaper, 2020. [Online]. Available: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf

4. 3GPP, "TS 38.211: NR; Physical channels and modulation," Release 17, 2023. [Online]. Available: https://www.etsi.org/deliver/etsi_ts/138200_138299/138211/17.10.00_60/ts_138211v171000p.pdf

5. O-RAN Alliance, "WG4 CUS Plane Specification," v16.01. [Online]. Available: https://specifications.o-ran.org/download?id=738

6. NVIDIA, "NVIDIA Ampere Architecture In-Depth," [Online]. Available: https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/

7. J. Wang et al., "Dissecting Tensor Cores via Microbenchmarks: Latency, Throughput and Numeric Behaviors," arXiv:2206.02874, 2022. [Online]. Available: https://arxiv.org/abs/2206.02874

8. GPU MODE & NVIDIA, "Blackwell NVFP4 Kernel Hackathon," 2025–2026. [Online]. Available: https://luma.com/9n27uem4

9. MIT HAN Lab, "KernelWiki — Research Contests," [Online]. Available: https://github.com/mit-han-lab/KernelWiki/blob/master/research-contests.md

## License

This project uses NVIDIA Aerial cuPHY as a reference framework. The optimization code in this repository is provided for research and benchmarking purposes.
