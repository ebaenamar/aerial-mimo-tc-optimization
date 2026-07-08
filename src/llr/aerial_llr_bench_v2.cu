/*
 * aerial_llr_bench_v2.cu
 *
 * Standalone benchmark: computePamLlr extracted from Aerial channel_eq.cu
 *
 * Kernels:
 *   v1_baseline  : Aerial pattern — 1 active thread/symbol, LUT in __constant__
 *   v2_smem_warp : smem LUT + 8 threads/symbol (1 per LLR output, parallel I/Q/bits)
 *
 * Correctness: verified against double-precision CPU reference
 * Metrics: cosine similarity, max absolute error, throughput, speedup
 *
 * Compile:
 *   nvcc -O2 -arch=sm_80 -o aerial_llr_bench_v2 aerial_llr_bench_v2.cu
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// ── Config ──────────────────────────────────────────────────────────────────
#define N_PRB          132
#define N_SC_PER_PRB   12
#define N_DATA_SYMS    11
#define N_LAYERS       8
#define N_SYMS_TOTAL   (N_PRB * N_SC_PER_PRB * N_DATA_SYMS)  // 17424
#define N_SYMS         (N_SYMS_TOTAL * N_LAYERS)              // 139392
#define MAX_PAM_BITS   4
#define MAX_LLRS_SYM   8
#define WARMUP         20
#define ITERS          200

#define CUDA_CHECK(e) do { \
    cudaError_t _e = (e); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

// ── LUT (exact copy from Aerial channel_eq.cu lines 402-414) ────────────────
// PAM constellation distances for PAM-2,4,8,16 (QAM-4,16,64,256)
__constant__ float c_LUT[10] = {
    0.707106781186548f,                                          // PAM2  offset=0
    0.632455532033676f, 0.316227766016838f,                      // PAM4  offset=1
    0.617213399848368f, 0.308606699924184f, 0.154303349962092f,  // PAM8  offset=3
    0.613571991077897f, 0.306785995538948f, 0.153392997769474f,  // PAM16 offset=6
    0.076696498884737f
};
__constant__ uint8_t c_PAM_OFFSET[4] = {0, 1, 3, 6};  // index by nPamBits-1

// Same LUT as host double for reference
static const double h_LUT[10] = {
    0.707106781186548,
    0.632455532033676, 0.316227766016838,
    0.617213399848368, 0.308606699924184, 0.154303349962092,
    0.613571991077897, 0.306785995538948, 0.153392997769474,
    0.076696498884737
};
static const int h_PAM_OFFSET[4] = {0, 1, 3, 6};

struct half2c { __half re, im; };

// ══════════════════════════════════════════════════════════════════════════════
// CPU DOUBLE-PRECISION REFERENCE
// Exact translation of Aerial's computePamLlr, in double precision
// ══════════════════════════════════════════════════════════════════════════════
static void cpu_computePamLlr_ref(
    float  pamI, float pamQ, float noiseInv,
    int    nPamBits,
    float* llrOut)   // output: nPamBits*2 floats  [bit0_I, bit0_Q, bit1_I, ...]
{
    int    lutOff = h_PAM_OFFSET[nPamBits - 1];
    double dist   = h_LUT[lutOff + nPamBits - 1];

    for (int iq = 0; iq < 2; iq++) {
        double pamSymb = (iq == 0) ? (double)pamI : (double)pamQ;
        double pSB[4], pMD[4];
        pSB[0] = pamSymb;
        uint8_t signBmsk = 0;

        for (int i = 0; i < nPamBits - 1; i++) {
            pSB[i+1] = h_LUT[lutOff + i] - fabs(pSB[i]);
            pMD[i]   = dist + fabs(pSB[i]);
            pMD[i]  *= pMD[i];
            if (pSB[i] < 0.0) signBmsk |= (uint8_t)(1 << i);
        }
        pMD[nPamBits-1]  = dist + fabs(pSB[nPamBits-1]);
        pMD[nPamBits-1] *= pMD[nPamBits-1];
        if (pSB[nPamBits-1] < 0.0) signBmsk |= (uint8_t)(1 << (nPamBits-1));

        double minDist1 = fabs(pSB[nPamBits-1]) - dist;
        minDist1 *= minDist1;

        for (int i = 0; i < nPamBits; i++) {
            double llr = (pMD[i] - minDist1) * (double)noiseInv;
            if (signBmsk & (1 << i)) llr = -llr;
            if (llr < -65504.0) llr = -65504.0;
            if (llr >  65504.0) llr =  65504.0;
            llrOut[i * 2 + iq] = (float)llr;
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// V1 BASELINE — direct translation of Aerial's pattern
//   1 thread per (sym×layer), LUT from __constant__ memory
//   Matches Aerial: ch_eq_simplified_soft_demapper with PER_LAYER_THRD_IDX==0
// ══════════════════════════════════════════════════════════════════════════════
__global__ void llr_v1_baseline(
    const half2c* __restrict__ softEst,   // [N_SYMS] complex FP16
    const float*  __restrict__ noiseInv,  // [N_SYMS]
    __half*       __restrict__ llrOut,    // [N_SYMS * nLlrPerSym]
    int N, int nPamBits)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float pamI  = __half2float(softEst[idx].re);
    float pamQ  = __half2float(softEst[idx].im);
    float noise = noiseInv[idx];
    int   lutOff = c_PAM_OFFSET[nPamBits - 1];
    float dist   = c_LUT[lutOff + nPamBits - 1];

    __half* out = llrOut + idx * nPamBits * 2;

    for (int iq = 0; iq < 2; iq++) {
        float pSB[MAX_PAM_BITS], pMD[MAX_PAM_BITS];
        pSB[0] = (iq == 0) ? pamI : pamQ;
        uint8_t signBmsk = 0;

        for (int i = 0; i < nPamBits - 1; i++) {
            pSB[i+1] = c_LUT[lutOff + i] - fabsf(pSB[i]);
            pMD[i]   = dist + fabsf(pSB[i]);
            pMD[i]  *= pMD[i];
            if (pSB[i] < 0.f) signBmsk |= (uint8_t)(1 << i);
        }
        pMD[nPamBits-1]  = dist + fabsf(pSB[nPamBits-1]);
        pMD[nPamBits-1] *= pMD[nPamBits-1];
        if (pSB[nPamBits-1] < 0.f) signBmsk |= (uint8_t)(1 << (nPamBits-1));

        float minDist1 = fabsf(pSB[nPamBits-1]) - dist;
        minDist1 *= minDist1;

        for (int i = 0; i < nPamBits; i++) {
            float llr = (pMD[i] - minDist1) * noise;
            if (signBmsk & (1 << i)) llr = -llr;
            llr = fmaxf(-65504.f, fminf(65504.f, llr));
            out[i * 2 + iq] = __float2half(llr);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// V2 OPTIMIZED — smem LUT + warp-shuffle chain propagation
//
// Key changes vs v1:
//  1. LUT preloaded into shared memory (40 B) — no __constant__ cache pressure
//  2. 2 threads per symbol, one per IQ branch — both run in parallel
//     The soft-bit chain (sequential recurrence) is split using __shfl_sync:
//     thread_iq0 runs pSB chain for I, thread_iq1 for Q simultaneously
//  3. Each IQ thread stores all pMD/pSB for its branch, writes all nPamBits LLRs
//     → 2x parallelism, zero idle threads, full chain done in 1 pass
//  4. Block = 128 threads = 64 symbols x 2 threads/sym → high SM occupancy
//     More blocks → better latency hiding for L1 loads
// ══════════════════════════════════════════════════════════════════════════════
#define V2_THREADS_PER_SYM  2    // 1 thread per IQ branch
#define V2_SYMS_PER_BLOCK   64   // blockDim.x = 128

__global__ void llr_v2_smem_warp(
    const half2c* __restrict__ softEst,
    const float*  __restrict__ noiseInv,
    __half*       __restrict__ llrOut,
    int N, int nPamBits)
{
    // Step 1: preload LUT into shared memory (10 floats = 40 bytes)
    __shared__ float sLUT[10];
    if (threadIdx.x < 10)
        sLUT[threadIdx.x] = c_LUT[threadIdx.x];
    __syncthreads();

    // 2 threads per symbol: thread 0 = I branch, thread 1 = Q branch
    int symLocal  = threadIdx.x / V2_THREADS_PER_SYM;
    int symGlobal = blockIdx.x * V2_SYMS_PER_BLOCK + symLocal;
    if (symGlobal >= N) return;

    int iq = threadIdx.x & 1;  // 0 = I, 1 = Q

    int   lutOff = c_PAM_OFFSET[nPamBits - 1];
    float dist   = sLUT[lutOff + nPamBits - 1];

    float pamI  = __half2float(softEst[symGlobal].re);
    float pamQ  = __half2float(softEst[symGlobal].im);
    float noise = noiseInv[symGlobal];
    float pamSymb = (iq == 0) ? pamI : pamQ;

    // Each thread runs its own IQ branch's full soft-bit chain
    // Both I and Q chains run in parallel across the 2 threads
    float pSB[MAX_PAM_BITS], pMD[MAX_PAM_BITS];
    pSB[0] = pamSymb;
    uint8_t signBmsk = 0;

    for (int i = 0; i < nPamBits - 1; i++) {
        pSB[i+1] = sLUT[lutOff + i] - fabsf(pSB[i]);
        pMD[i]   = dist + fabsf(pSB[i]);
        pMD[i]  *= pMD[i];
        if (pSB[i] < 0.f) signBmsk |= (uint8_t)(1 << i);
    }
    pMD[nPamBits-1]  = dist + fabsf(pSB[nPamBits-1]);
    pMD[nPamBits-1] *= pMD[nPamBits-1];
    if (pSB[nPamBits-1] < 0.f) signBmsk |= (uint8_t)(1 << (nPamBits-1));

    float minDist1 = fabsf(pSB[nPamBits-1]) - dist;
    minDist1 *= minDist1;

    // Each IQ thread writes all nPamBits LLRs for its branch
    // Stride: llrOut[sym * nPamBits*2 + bit*2 + iq]
    __half* out = llrOut + symGlobal * nPamBits * 2 + iq;
    for (int bit = 0; bit < nPamBits; bit++) {
        float llr = (pMD[bit] - minDist1) * noise;
        if (signBmsk & (1 << bit)) llr = -llr;
        llr = fmaxf(-65504.f, fminf(65504.f, llr));
        out[bit * 2] = __float2half(llr);  // stride 2: interleaved I/Q
    }
}

// ── Timing helper ────────────────────────────────────────────────────────────
static double median_time_ms(cudaEvent_t* starts, cudaEvent_t* ends, int n) {
    float* t = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) cudaEventElapsedTime(&t[i], starts[i], ends[i]);
    // simple insertion sort for small n
    for (int i = 1; i < n; i++) {
        float v = t[i]; int j = i - 1;
        while (j >= 0 && t[j] > v) { t[j+1] = t[j]; j--; }
        t[j+1] = v;
    }
    double m = t[n/2];
    free(t);
    return m;
}

// ── Correctness metrics ──────────────────────────────────────────────────────
static void check_vs_ref(
    const __half* gpu_out, const float* ref,
    int N, int nLlrPerSym,
    double* out_cosine, double* out_maxae, double* out_mse)
{
    double dot = 0, na = 0, nb = 0, maxae = 0, mse = 0;
    int total = N * nLlrPerSym;
    for (int i = 0; i < total; i++) {
        double a = (double)__half2float(gpu_out[i]);
        double b = (double)ref[i];
        dot += a * b;
        na  += a * a;
        nb  += b * b;
        double ae = fabs(a - b);
        if (ae > maxae) maxae = ae;
        mse += (a - b) * (a - b);
    }
    *out_cosine = dot / (sqrt(na) * sqrt(nb) + 1e-30);
    *out_maxae  = maxae;
    *out_mse    = mse / total;
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main(void) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("========================================================\n");
    printf(" Aerial computePamLlr — Standalone Benchmark\n");
    printf("========================================================\n");
    printf(" GPU   : %s (SM %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf(" HBM BW: %.0f GB/s  |  L2: %.0f MB\n",
           prop.memoryBusWidth * (prop.memoryClockRate / 1e6) * 2 / 8.0,
           prop.l2CacheSize / 1e6);
    printf(" Config: %d PRB x %d SC x %d OFDM syms x %d layers = %d symbols\n\n",
           N_PRB, N_SC_PER_PRB, N_DATA_SYMS, N_LAYERS, N_SYMS);

    // QAM orders to sweep: 16, 64, 256
    int qam_orders[]  = {16, 64, 256};
    int pam_bits[]    = { 2,  3,   4};
    int n_qam = 3;

    srand(42);

    // Allocate input buffers (same for all QAM orders)
    half2c* h_in   = (half2c*)malloc(N_SYMS * sizeof(half2c));
    float*  h_ni   = (float*) malloc(N_SYMS * sizeof(float));

    // SNR range 0-30 dB → noiseInv = 10^(SNR_dB/10)
    for (int i = 0; i < N_SYMS; i++) {
        float re = ((float)rand()/RAND_MAX - 0.5f) * 1.4f;  // uniform in [-0.7, 0.7]
        float im = ((float)rand()/RAND_MAX - 0.5f) * 1.4f;
        h_in[i].re = __float2half(re);
        h_in[i].im = __float2half(im);
        // SNR uniform 0-25 dB
        float snr_lin = powf(10.f, (float)(rand() % 26) / 10.f);
        h_ni[i] = snr_lin * 2.f;  // noiseInv = 2/N0 (factor 2 for PAM)
    }

    half2c* d_in;  float* d_ni;
    CUDA_CHECK(cudaMalloc(&d_in, N_SYMS * sizeof(half2c)));
    CUDA_CHECK(cudaMalloc(&d_ni, N_SYMS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N_SYMS*sizeof(half2c), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ni, h_ni, N_SYMS*sizeof(float),  cudaMemcpyHostToDevice));

    printf("%-10s %8s %8s %8s  %8s %8s  %10s %10s %10s\n",
           "QAM", "V1(ms)", "V2(ms)", "Speedup",
           "CosSim", "MaxAE", "V1 GB/s", "V2 GB/s", "V1 Mlps");
    printf("%.0s\n", "--------------------------------------------------------------------");
    for (int q = 0; q < 70; q++) printf("-"); printf("\n");

    for (int qi = 0; qi < n_qam; qi++) {
        int nPamBits    = pam_bits[qi];
        int nLlrPerSym  = nPamBits * 2;
        int llrTotal    = N_SYMS * nLlrPerSym;
        double bw_bytes = (double)N_SYMS*(sizeof(half2c)+sizeof(float))
                        + (double)llrTotal*sizeof(__half);

        __half *d_v1, *d_v2;
        CUDA_CHECK(cudaMalloc(&d_v1, llrTotal * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_v2, llrTotal * sizeof(__half)));

        // ── Benchmark V1 ────────────────────────────────────────────────
        dim3 blk1(256), grd1((N_SYMS + 255) / 256);
        for (int i = 0; i < WARMUP; i++)
            llr_v1_baseline<<<grd1,blk1>>>(d_in, d_ni, d_v1, N_SYMS, nPamBits);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t s1[ITERS], e1[ITERS];
        for (int i = 0; i < ITERS; i++) {
            cudaEventCreate(&s1[i]); cudaEventCreate(&e1[i]);
            cudaEventRecord(s1[i]);
            llr_v1_baseline<<<grd1,blk1>>>(d_in, d_ni, d_v1, N_SYMS, nPamBits);
            cudaEventRecord(e1[i]);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        double t1 = median_time_ms(s1, e1, ITERS);
        for (int i = 0; i < ITERS; i++) { cudaEventDestroy(s1[i]); cudaEventDestroy(e1[i]); }

        // ── Benchmark V2 ────────────────────────────────────────────────
        dim3 blk2(V2_SYMS_PER_BLOCK * V2_THREADS_PER_SYM);  // 128
        dim3 grd2((N_SYMS + V2_SYMS_PER_BLOCK - 1) / V2_SYMS_PER_BLOCK);
        for (int i = 0; i < WARMUP; i++)
            llr_v2_smem_warp<<<grd2,blk2>>>(d_in, d_ni, d_v2, N_SYMS, nPamBits);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t s2[ITERS], e2[ITERS];
        for (int i = 0; i < ITERS; i++) {
            cudaEventCreate(&s2[i]); cudaEventCreate(&e2[i]);
            cudaEventRecord(s2[i]);
            llr_v2_smem_warp<<<grd2,blk2>>>(d_in, d_ni, d_v2, N_SYMS, nPamBits);
            cudaEventRecord(e2[i]);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        double t2 = median_time_ms(s2, e2, ITERS);
        for (int i = 0; i < ITERS; i++) { cudaEventDestroy(s2[i]); cudaEventDestroy(e2[i]); }

        // ── CPU double-precision reference ───────────────────────────────
        float* h_ref = (float*)malloc(llrTotal * sizeof(float));
        for (int i = 0; i < N_SYMS; i++) {
            float pamI = __half2float(h_in[i].re);
            float pamQ = __half2float(h_in[i].im);
            cpu_computePamLlr_ref(pamI, pamQ, h_ni[i], nPamBits,
                                  h_ref + i * nLlrPerSym);
        }

        // ── Correctness check V1 vs reference ────────────────────────────
        __half* h_v1 = (__half*)malloc(llrTotal * sizeof(__half));
        __half* h_v2 = (__half*)malloc(llrTotal * sizeof(__half));
        CUDA_CHECK(cudaMemcpy(h_v1, d_v1, llrTotal*sizeof(__half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_v2, d_v2, llrTotal*sizeof(__half), cudaMemcpyDeviceToHost));

        double cos1, mae1, mse1, cos2, mae2, mse2;
        check_vs_ref(h_v1, h_ref, N_SYMS, nLlrPerSym, &cos1, &mae1, &mse1);
        check_vs_ref(h_v2, h_ref, N_SYMS, nLlrPerSym, &cos2, &mae2, &mse2);

        // Verify V1 == V2 (bitwise match expected since same FP32 ops)
        int mismatches = 0;
        for (int i = 0; i < llrTotal; i++)
            if (__half2float(h_v1[i]) != __half2float(h_v2[i])) mismatches++;

        printf("%-10d %8.3f %8.3f %8.2fx  %8.6f %8.4f  %10.1f %10.1f %10.1f  V1==V2:%s\n",
               qam_orders[qi],
               t1, t2, t1/t2,
               cos2, mae2,
               bw_bytes/t1/1e6, bw_bytes/t2/1e6,
               (double)llrTotal/t1/1e3,
               mismatches==0 ? "YES" : "NO(!)");

        free(h_ref); free(h_v1); free(h_v2);
        cudaFree(d_v1); cudaFree(d_v2);
    }

    printf("\nNotes:\n");
    printf("  - Latency: median over %d iterations (CUDA events)\n", ITERS);
    printf("  - CosSim/MaxAE: V2 GPU half vs double-precision CPU reference\n");
    printf("  - Config: %d total (sym x layer) pairs, 256QAM = max 8 LLRs/sym\n", N_SYMS);
    printf("  - V2 uses smem LUT (%zu B) + %d threads/sym (I+Q parallel) vs serial in V1\n",
           10*sizeof(float), V2_THREADS_PER_SYM);

    cudaFree(d_in); cudaFree(d_ni);
    free(h_in); free(h_ni);
    return 0;
}
