/*
 * aerial_bottleneck.cu
 *
 * Microbenchmark: which dominates in cuPHY PUSCH RX?
 *   A) Gram matrix compute:       G[8x8]   = M[8x64]  @ H[64x8]   (coef compute)
 *   B) Coefficient application:   Y_eq[8]  = C[8x64]  @ Y_rx[64]  (soft demap)
 *
 * Aerial config: 132 PRBs, 12 SC/PRB, 11 data symbols, 8 layers
 * Both executed with the same scalar FP32 loop pattern as in channel_eq.cu
 *
 * Compile: nvcc -O3 -arch=sm_80 -o aerial_bottleneck aerial_bottleneck.cu
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ── Config matching Aerial PUSCH RX ──────────────────────────────────────────
#define N_BS_ANTS     64
#define N_LAYERS      8
#define N_PRB         132
#define N_SC_PRB      12       // CUPHY_N_TONES_PER_PRB
#define N_DATA_SYMS   11
#define N_SUBCARRIERS (N_PRB * N_SC_PRB)   // 1584

// Gram compute: called once per subcarrier during coef compute phase
// Grid: (N_SC_PRB, N_PRB, 1), Block: (N_BS_ANTS, 1, 1)  → 1 thread per antenna
// Each block computes G[8×8] for 1 subcarrier via outer-product accumulation
#define GRAM_ITERS    1        // 1 per subcarrier per OFDM symbol group

// Coef apply: called per (subcarrier, data_symbol, layer)
// Grid: (N_PRB, N_LAYERS, 1), Block: (N_SC_PRB, N_DATA_SYMS, 1)
// Each thread computes 1 softEst = dot(C[64], Y_rx[64])

#define WARMUP 50
#define ITERS  500

#define CUDA_CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)

// Complex FP32
struct cf32 { float re, im; };
__device__ __forceinline__ cf32 cma(cf32 a, cf32 b, cf32 acc) {
    return {acc.re + a.re*b.re - a.im*b.im,
            acc.im + a.re*b.im + a.im*b.re};
}

// ══════════════════════════════════════════════════════════════════════════════
// KERNEL A: Gram matrix compute — G[L×L] = M[L×A] @ H[A×L]
// Models eqMmseCoefCompHighMimoKernel inner loop (line 759 channel_eq.cu)
// Grid: (N_SC_PRB, N_PRB)  Block: (N_LAYERS, N_LAYERS) = (8,8)
// Each thread computes 1 element of G
// ══════════════════════════════════════════════════════════════════════════════
__global__ void gram_scalar(
    const cf32* __restrict__ M,   // [N_SUBCARRIERS, N_LAYERS, N_BS_ANTS]
    const cf32* __restrict__ H,   // [N_SUBCARRIERS, N_BS_ANTS, N_LAYERS]
    cf32*       __restrict__ G,   // [N_SUBCARRIERS, N_LAYERS, N_LAYERS]
    int nSC)
{
    int sc = blockIdx.x + blockIdx.y * N_SC_PRB;
    if (sc >= nSC) return;

    int row = threadIdx.x;  // layer index (0..N_LAYERS-1)
    int col = threadIdx.y;

    // M[sc, row, k] @ H[sc, k, col]  for k=0..N_BS_ANTS-1
    cf32 acc = {0.f, 0.f};
    const cf32* Mrow = M + sc * N_LAYERS * N_BS_ANTS + row * N_BS_ANTS;
    const cf32* Hcol = H + sc * N_BS_ANTS * N_LAYERS + col;

    #pragma unroll 4
    for (int k = 0; k < N_BS_ANTS; k++) {
        cf32 m = Mrow[k];
        cf32 h = Hcol[k * N_LAYERS];
        acc = cma(m, h, acc);
    }
    G[sc * N_LAYERS * N_LAYERS + row * N_LAYERS + col] = acc;
}

// ══════════════════════════════════════════════════════════════════════════════
// KERNEL B: Coefficient application — softEst = C[A] · Y_rx[A]
// Models eqMmseSoftDemapKernel_v4 inner loop (lines 3800-3825 channel_eq.cu)
// Grid: (N_PRB, N_LAYERS)  Block: (N_SC_PRB, N_DATA_SYMS) = (12, 11)
// Each thread: 1 subcarrier × 1 symbol → 1 dot product over N_BS_ANTS=64
// ══════════════════════════════════════════════════════════════════════════════
__global__ void coef_apply_scalar(
    const cf32* __restrict__ C,      // [N_SUBCARRIERS, N_LAYERS, N_BS_ANTS]
    const cf32* __restrict__ Y_rx,   // [N_SUBCARRIERS, N_DATA_SYMS, N_BS_ANTS]
    cf32*       __restrict__ Y_eq,   // [N_SUBCARRIERS, N_LAYERS, N_DATA_SYMS]
    int nSC, int nSym)
{
    int prb   = blockIdx.x;
    int layer = blockIdx.y;
    int sc_local = threadIdx.x;   // 0..11
    int sym      = threadIdx.y;   // 0..10

    int sc = prb * N_SC_PRB + sc_local;
    if (sc >= nSC || sym >= nSym) return;

    // Load coef vector C[sc, layer, 0..63] — smem cache (mirrors Aerial sC)
    __shared__ cf32 sC[N_BS_ANTS][N_SC_PRB];
    if (sym == 0) {
        for (int ant = 0; ant < N_BS_ANTS; ant++)
            sC[ant][sc_local] = C[sc * N_LAYERS * N_BS_ANTS + layer * N_BS_ANTS + ant];
    }
    __syncthreads();

    // dot product: softEst += C[ant] * Y_rx[ant]  (same as cuCma loop in Aerial)
    cf32 softEst = {0.f, 0.f};
    cf32 Cnext = sC[0][sc_local];
    cf32 Ynext = Y_rx[sc * nSym * N_BS_ANTS + sym * N_BS_ANTS + 0];

    #pragma unroll 4
    for (int ant = 0; ant + 1 < N_BS_ANTS; ant++) {
        softEst = cma(Cnext, Ynext, softEst);
        Cnext = sC[ant+1][sc_local];
        Ynext = Y_rx[sc * nSym * N_BS_ANTS + sym * N_BS_ANTS + ant + 1];
    }
    softEst = cma(Cnext, Ynext, softEst);

    Y_eq[sc * N_LAYERS * nSym + layer * nSym + sym] = softEst;
}

// ── helpers ──────────────────────────────────────────────────────────────────
static double median_ms(float* t, int n) {
    for (int i=1;i<n;i++){float v=t[i];int j=i-1;while(j>=0&&t[j]>v){t[j+1]=t[j];j--;}t[j+1]=v;}
    return t[n/2];
}

int main(void) {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("==========================================================\n");
    printf(" Aerial PUSCH RX Bottleneck Analysis\n");
    printf("==========================================================\n");
    printf(" GPU: %s | %d SMs | %.0f GB/s HBM\n\n", p.name,
           p.multiProcessorCount,
           p.memoryBusWidth*(p.memoryClockRate/1e6)*2/8.0);
    printf(" Config: %d PRB × %d SC × %d layers × %d data syms\n",
           N_PRB, N_SC_PRB, N_LAYERS, N_DATA_SYMS);
    printf(" N_BS_ANTS=%d  N_SUBCARRIERS=%d\n\n", N_BS_ANTS, N_SUBCARRIERS);

    srand(42);
    int nSC  = N_SUBCARRIERS;
    int nSym = N_DATA_SYMS;

    // ── Allocate ─────────────────────────────────────────────────────────────
    // Gram: M[nSC, L, A], H[nSC, A, L], G[nSC, L, L]
    size_t sz_M = (size_t)nSC * N_LAYERS  * N_BS_ANTS * sizeof(cf32);
    size_t sz_H = (size_t)nSC * N_BS_ANTS * N_LAYERS  * sizeof(cf32);
    size_t sz_G = (size_t)nSC * N_LAYERS  * N_LAYERS  * sizeof(cf32);
    // Coef apply: C[nSC, L, A], Y_rx[nSC, nSym, A], Y_eq[nSC, L, nSym]
    size_t sz_C    = sz_M;
    size_t sz_Yrx  = (size_t)nSC * nSym * N_BS_ANTS * sizeof(cf32);
    size_t sz_Yeq  = (size_t)nSC * N_LAYERS * nSym  * sizeof(cf32);

    cf32 *d_M, *d_H, *d_G, *d_C, *d_Yrx, *d_Yeq;
    CUDA_CHECK(cudaMalloc(&d_M,   sz_M));
    CUDA_CHECK(cudaMalloc(&d_H,   sz_H));
    CUDA_CHECK(cudaMalloc(&d_G,   sz_G));
    CUDA_CHECK(cudaMalloc(&d_C,   sz_C));
    CUDA_CHECK(cudaMalloc(&d_Yrx, sz_Yrx));
    CUDA_CHECK(cudaMalloc(&d_Yeq, sz_Yeq));

    // Random init on device via memset pattern (sufficient for timing)
    CUDA_CHECK(cudaMemset(d_M,   1, sz_M));
    CUDA_CHECK(cudaMemset(d_H,   1, sz_H));
    CUDA_CHECK(cudaMemset(d_C,   1, sz_C));
    CUDA_CHECK(cudaMemset(d_Yrx, 1, sz_Yrx));

    // ── Gram kernel config ───────────────────────────────────────────────────
    // 1 block per subcarrier, (N_LAYERS × N_LAYERS) threads per block
    dim3 gram_blk(N_LAYERS, N_LAYERS);           // 8×8 = 64 threads
    dim3 gram_grd(N_SC_PRB, N_PRB);              // 12×132 = 1584 blocks

    // ── Coef apply config ────────────────────────────────────────────────────
    dim3 coef_blk(N_SC_PRB, N_DATA_SYMS);        // 12×11 = 132 threads
    dim3 coef_grd(N_PRB, N_LAYERS);              // 132×8 = 1056 blocks

    // ── Warmup ───────────────────────────────────────────────────────────────
    for (int i=0;i<WARMUP;i++) {
        gram_scalar<<<gram_grd,gram_blk>>>(d_M,d_H,d_G,nSC);
        coef_apply_scalar<<<coef_grd,coef_blk>>>(d_C,d_Yrx,d_Yeq,nSC,nSym);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Benchmark — alternating to remove order bias ─────────────────────────
    float *tA=(float*)malloc(ITERS*sizeof(float)), *tB=(float*)malloc(ITERS*sizeof(float));
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    float ms;

    for (int i=0;i<ITERS;i++) {
        cudaEventRecord(s);
        gram_scalar<<<gram_grd,gram_blk>>>(d_M,d_H,d_G,nSC);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms,s,e); tA[i]=ms;

        cudaEventRecord(s);
        coef_apply_scalar<<<coef_grd,coef_blk>>>(d_C,d_Yrx,d_Yeq,nSC,nSym);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms,s,e); tB[i]=ms;
    }

    double mA = median_ms(tA, ITERS);
    double mB = median_ms(tB, ITERS);

    // ── Arithmetic intensity ─────────────────────────────────────────────────
    // Gram: nSC × (L×L) outputs, each costs 2×A FMAs complex = 8*A real FMAs
    double flops_gram = (double)nSC * N_LAYERS * N_LAYERS * 8.0 * N_BS_ANTS;
    double bytes_gram = (double)(sz_M + sz_H + sz_G);

    // Coef: nSC × nSym × L outputs, each costs 2×A FMAs complex
    double flops_coef = (double)nSC * nSym * N_LAYERS * 8.0 * N_BS_ANTS;
    double bytes_coef = (double)(sz_C + sz_Yrx + sz_Yeq);

    printf("┌─────────────────────────────────────────────────────┐\n");
    printf("│ Kernel A: Gram compute  G[8×8] = M[8×64] @ H[64×8] │\n");
    printf("│  per subcarrier (1584 subcarriers)                  │\n");
    printf("├─────────────────────────────────────────────────────┤\n");
    printf("│  Latency  : %6.3f ms (median %d iters)            │\n", mA, ITERS);
    printf("│  Throughput: %6.1f GFLOP/s                         │\n", flops_gram/mA/1e6);
    printf("│  BW used  : %6.1f GB/s                             │\n", bytes_gram/mA/1e6);
    printf("│  Arith Int: %6.2f FLOP/byte                        │\n", flops_gram/bytes_gram);
    printf("├─────────────────────────────────────────────────────┤\n");
    printf("│ Kernel B: Coef apply  Y_eq[8] = C[8×64] · Y_rx[64] │\n");
    printf("│  per subcarrier × 11 data symbols                   │\n");
    printf("├─────────────────────────────────────────────────────┤\n");
    printf("│  Latency  : %6.3f ms (median %d iters)            │\n", mB, ITERS);
    printf("│  Throughput: %6.1f GFLOP/s                         │\n", flops_coef/mB/1e6);
    printf("│  BW used  : %6.1f GB/s                             │\n", bytes_coef/mB/1e6);
    printf("│  Arith Int: %6.2f FLOP/byte                        │\n", flops_coef/bytes_coef);
    printf("└─────────────────────────────────────────────────────┘\n\n");
    printf(" Ratio B/A (coef_apply / gram): %.2fx slower\n", mB/mA);
    printf(" → Optimize the %.0f%% dominant kernel first\n",
           100.0*fmax(mA,mB)/(mA+mB));

    // Cleanup
    cudaFree(d_M); cudaFree(d_H); cudaFree(d_G);
    cudaFree(d_C); cudaFree(d_Yrx); cudaFree(d_Yeq);
    free(tA); free(tB);
    return 0;
}
