/*
 * aerial_coef_wmma.cu
 *
 * Benchmark: eqMmseSoftDemapKernel_v4 inner GEMM
 *   Y_eq[N_LAYERS, N_SC] = C[N_LAYERS, N_BS_ANTS] @ Y_rx[N_BS_ANTS, N_SC]
 *   (complex FP16 inputs, FP32 accumulation)
 *
 * Baseline V1: scalar FP32 cuCma loop — exact Aerial pattern
 * Optimized V2: WMMA FP16 tensor cores — hackathon technique applied to Aerial
 *
 * Key idea from sundai-hackathon-track2:
 *   Replace serial dot-product loop (1 thread/output, K=64 FMAs) with
 *   warp-level MMA using nvcuda::wmma — same philosophy as mma_s4 in the hack,
 *   but FP16 complex here since channel data is complex-valued.
 *
 * Complex WMMA trick: treat complex GEMM as 2x2 block of real GEMMs:
 *   Re(C) = Re(A)@Re(B) - Im(A)@Im(B)
 *   Im(C) = Re(A)@Im(B) + Im(A)@Re(B)
 *   → 4 real WMMA calls per complex tile
 *
 * Tile mapping:
 *   WMMA fragment: m=16, n=16, k=16 (FP16 accumulate FP32)
 *   A: [N_LAYERS=8 → padded 16, N_BS_ANTS=64] → 4 k-steps of 16
 *   B: [N_BS_ANTS=64, N_SC_PRB=12 → padded 16]
 *   1 warp per (PRB, data_symbol) pair
 *
 * Compile:
 *   nvcc -O3 -arch=sm_80 -o aerial_coef_wmma aerial_coef_wmma.cu
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

using namespace nvcuda;

// ── Aerial config ─────────────────────────────────────────────────────────────
#define N_BS_ANTS   64
#define N_LAYERS    8
#define N_PRB       132
#define N_SC_PRB    12
#define N_DATA_SYMS 11
#define N_SC        (N_PRB * N_SC_PRB)   // 1584

// WMMA tile dimensions
#define WM  16   // m: rows of A and C (pad N_LAYERS=8 → 16)
#define WN  16   // n: cols of B and C (pad N_SC_PRB=12 → 16)
#define WK  16   // k: inner dimension steps

#define WARMUP 50
#define ITERS  500

#define CUDA_CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)

struct cf32 { float re, im; };
__device__ __forceinline__ cf32 cma(cf32 a, cf32 b, cf32 acc) {
    return {acc.re + a.re*b.re - a.im*b.im,
            acc.im + a.re*b.im + a.im*b.re};
}

// ══════════════════════════════════════════════════════════════════════════════
// V1 BASELINE — scalar FP32, exact Aerial eqMmseSoftDemapKernel_v4 pattern
// Grid: (N_PRB, N_LAYERS)  Block: (N_SC_PRB, N_DATA_SYMS)
// Each thread: 1 dot product of length N_BS_ANTS=64
// ══════════════════════════════════════════════════════════════════════════════
__global__ void coef_apply_v1_scalar(
    const cf32* __restrict__ C,      // [N_SC, N_LAYERS, N_BS_ANTS]
    const cf32* __restrict__ Y_rx,   // [N_SC, N_DATA_SYMS, N_BS_ANTS]
    cf32*       __restrict__ Y_eq,   // [N_SC, N_LAYERS, N_DATA_SYMS]
    int nSC, int nSym)
{
    int prb   = blockIdx.x;
    int layer = blockIdx.y;
    int sc_l  = threadIdx.x;
    int sym   = threadIdx.y;
    int sc    = prb * N_SC_PRB + sc_l;
    if (sc >= nSC || sym >= nSym) return;

    // Cache C into smem (mirrors Aerial sC)
    __shared__ cf32 sC[N_BS_ANTS][N_SC_PRB];
    if (sym == 0) {
        for (int a = 0; a < N_BS_ANTS; a++)
            sC[a][sc_l] = C[sc * N_LAYERS * N_BS_ANTS + layer * N_BS_ANTS + a];
    }
    __syncthreads();

    cf32 softEst = {0.f, 0.f};
    cf32 Cn = sC[0][sc_l];
    cf32 Yn = Y_rx[sc * nSym * N_BS_ANTS + sym * N_BS_ANTS];

    #pragma unroll 4
    for (int a = 0; a + 1 < N_BS_ANTS; a++) {
        softEst = cma(Cn, Yn, softEst);
        Cn = sC[a+1][sc_l];
        Yn = Y_rx[sc * nSym * N_BS_ANTS + sym * N_BS_ANTS + a + 1];
    }
    softEst = cma(Cn, Yn, softEst);
    Y_eq[sc * N_LAYERS * nSym + layer * nSym + sym] = softEst;
}

// ══════════════════════════════════════════════════════════════════════════════
// V2 WMMA — Tensor Core FP16 complex GEMM
//
// Correct tiling respecting per-SC coef structure:
//   1 block per PRB, N_SC_PRB=12 warps per block
//   Each warp w owns SC = prb*N_SC_PRB + w
//
//   Per warp GEMM:
//     A[w]: [WM=16, K=64]  ← C[sc, layers(pad8→16), ants]
//     B[w]: [K=64,  WN=16] ← Y_rx[sc, ants, syms(pad11→16)]
//     Out : [WM=16, WN=16] ← Y_eq[sc, layers, syms], extract [8,11]
//
//   Complex trick: 4 real WMMAs per warp
//     Re(Y_eq) = Re(C)@Re(Y) - Im(C)@Im(Y)
//     Im(Y_eq) = Re(C)@Im(Y) + Im(C)@Re(Y)
//
//   Each warp: 4 k-steps × 4 real MMAs = 16 MMA calls total
//   12 warps in flight → 192 MMA calls per block → full TC pipeline
//
// Grid: (N_PRB,)   Block: (N_SC_PRB * 32,) = 384 threads
// ══════════════════════════════════════════════════════════════════════════════

#define N_WARPS_V2  N_SC_PRB    // 12 warps, 1 per SC
// Per-warp smem: A[16][64] + B[64][16] × 4 components (re/im × A/B) × half
// = 4 × (16×64 + 64×16) × 2B = 4 × 2048 × 2 = 16KB per warp → too much
// Solution: reuse smem sequentially — load A then B, compute, store
// Use single shared tile per warp: A_re[16][64], A_im[16][64], B_re[64][16], B_im[64][16]
// Total per block: 12 warps × (2×16×64 + 2×64×16) × 2B = 12 × 4096 × 2 = 96KB > 48KB
//
// Fix: use registers for A (small, 16×64 half → 512 halfs = 1KB per warp but
// registers are limited). Better: share A across warps since different SC have
// different A. Use a ping-pong: load 1 warp's data at a time.
//
// Practical solution: 1 warp per block, loop over 12 SC.
// Smem: A_re[16][64], A_im[16][64], B_re[64][16], B_im[64][16] = 4×1024×2B = 8KB
// 1 warp does 12 × (4 k-steps × 4 MMAs) = 192 MMAs per block
// Grid: (N_PRB × N_SC_PRB,)  Block: (32,) = 1 warp
// This gives 1584 blocks × 1 warp = 1584 warps → fills 108 SMs × 64 warps
// The key vs V1: TC instead of scalar FP32, plus FP16 compute

__global__ void coef_apply_v2_wmma(
    const cf32* __restrict__ C,      // [N_SC, N_LAYERS, N_BS_ANTS]
    const cf32* __restrict__ Y_rx,   // [N_SC, N_DATA_SYMS, N_BS_ANTS]
    cf32*       __restrict__ Y_eq,   // [N_SC, N_LAYERS, N_DATA_SYMS]
    int nSC, int nSym)
{
    // 1 warp per SC
    int sc  = blockIdx.x;
    int lane = threadIdx.x;   // 0..31
    if (sc >= nSC) return;

    // Per-warp smem: A[16×64] + B[64×16] half × 4 matrices = 8KB
    __shared__ __half smA_re[WM][N_BS_ANTS];   // [16][64]
    __shared__ __half smA_im[WM][N_BS_ANTS];
    __shared__ __half smB_re[N_BS_ANTS][WN];   // [64][16]
    __shared__ __half smB_im[N_BS_ANTS][WN];

    // ── Load A: C[sc, layer, ant] → smA[layer_pad][ant] ──────────────────────
    // 16×64 = 1024 elements, 32 threads → 32 iters
    for (int idx = lane; idx < WM * N_BS_ANTS; idx += 32) {
        int row = idx / N_BS_ANTS;   // 0..15
        int col = idx % N_BS_ANTS;   // 0..63
        if (row < N_LAYERS) {
            cf32 c = C[sc * N_LAYERS * N_BS_ANTS + row * N_BS_ANTS + col];
            smA_re[row][col] = __float2half(c.re);
            smA_im[row][col] = __float2half(c.im);
        } else {
            smA_re[row][col] = __float2half(0.f);
            smA_im[row][col] = __float2half(0.f);
        }
    }

    // ── Load B: Y_rx[sc, sym, ant] → smB[ant][sym_pad] ───────────────────────
    // B[ant][sym]: ant=0..63, sym=0..10 (pad 11..15 with 0)
    // 64×16 = 1024 elements, 32 threads → 32 iters
    for (int idx = lane; idx < N_BS_ANTS * WN; idx += 32) {
        int ant = idx / WN;   // 0..63
        int sym = idx % WN;   // 0..15
        if (sym < nSym) {
            cf32 y = Y_rx[sc * nSym * N_BS_ANTS + sym * N_BS_ANTS + ant];
            smB_re[ant][sym] = __float2half(y.re);
            smB_im[ant][sym] = __float2half(y.im);
        } else {
            smB_re[ant][sym] = __float2half(0.f);
            smB_im[ant][sym] = __float2half(0.f);
        }
    }
    __syncwarp();

    // ── WMMA: 4 real GEMMs for complex result ─────────────────────────────────
    wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc_rr, acc_ri, acc_ir, acc_ii;

    wmma::fill_fragment(acc_rr, 0.f);
    wmma::fill_fragment(acc_ri, 0.f);
    wmma::fill_fragment(acc_ir, 0.f);
    wmma::fill_fragment(acc_ii, 0.f);

    #pragma unroll
    for (int k = 0; k < N_BS_ANTS; k += WK) {
        wmma::load_matrix_sync(a_frag, &smA_re[0][k], N_BS_ANTS);
        wmma::load_matrix_sync(b_frag, &smB_re[k][0], WN);
        wmma::mma_sync(acc_rr, a_frag, b_frag, acc_rr);  // Re*Re

        wmma::load_matrix_sync(b_frag, &smB_im[k][0], WN);
        wmma::mma_sync(acc_ri, a_frag, b_frag, acc_ri);  // Re*Im

        wmma::load_matrix_sync(a_frag, &smA_im[0][k], N_BS_ANTS);
        wmma::mma_sync(acc_ir, a_frag, b_frag, acc_ir);  // Im*Im

        wmma::load_matrix_sync(b_frag, &smB_re[k][0], WN);
        wmma::mma_sync(acc_ii, a_frag, b_frag, acc_ii);  // Im*Re
    }

    // ── Store result ──────────────────────────────────────────────────────────
    __shared__ float tmp_rr[WM][WN], tmp_ri[WM][WN];
    __shared__ float tmp_ir[WM][WN], tmp_ii[WM][WN];
    wmma::store_matrix_sync(&tmp_rr[0][0], acc_rr, WN, wmma::mem_row_major);
    wmma::store_matrix_sync(&tmp_ri[0][0], acc_ri, WN, wmma::mem_row_major);
    wmma::store_matrix_sync(&tmp_ir[0][0], acc_ir, WN, wmma::mem_row_major);
    wmma::store_matrix_sync(&tmp_ii[0][0], acc_ii, WN, wmma::mem_row_major);
    __syncwarp();

    // Write valid [N_LAYERS × nSym] outputs to global
    for (int idx = lane; idx < WM * WN; idx += 32) {
        int layer = idx / WN;
        int sym   = idx % WN;
        if (layer >= N_LAYERS || sym >= nSym) continue;
        float re = tmp_rr[layer][sym] - tmp_ir[layer][sym];
        float im = tmp_ri[layer][sym] + tmp_ii[layer][sym];
        Y_eq[sc * N_LAYERS * nSym + layer * nSym + sym] = {re, im};
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// V3 WMMA multi-warp — same as V2 but N_SC_PER_BLOCK subcarriers per block
// Increases occupancy: fewer blocks but more warps per SM for better hiding
// Grid: (ceil(N_SC / SC_PER_BLK),)   Block: (SC_PER_BLK * 32,)
// ══════════════════════════════════════════════════════════════════════════════

#define SC_PER_BLK  4    // warps per block — 4 keeps smem under 48KB
// smem: 4×(2×16×64 + 2×64×16)×2B + 4×4×16×16×4B = 32KB + 16KB = 48KB

__global__ void coef_apply_v3_wmma_multiw(
    const cf32* __restrict__ C,
    const cf32* __restrict__ Y_rx,
    cf32*       __restrict__ Y_eq,
    int nSC, int nSym)
{
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x % 32;
    int sc      = blockIdx.x * SC_PER_BLK + warp_id;
    if (sc >= nSC) return;

    // Per-warp smem slice — each warp uses its own region
    __shared__ __half smA_re[SC_PER_BLK][WM][N_BS_ANTS];
    __shared__ __half smA_im[SC_PER_BLK][WM][N_BS_ANTS];
    __shared__ __half smB_re[SC_PER_BLK][N_BS_ANTS][WN];
    __shared__ __half smB_im[SC_PER_BLK][N_BS_ANTS][WN];

    // Load A
    for (int idx = lane; idx < WM * N_BS_ANTS; idx += 32) {
        int row = idx / N_BS_ANTS;
        int col = idx % N_BS_ANTS;
        if (row < N_LAYERS) {
            cf32 c = C[sc * N_LAYERS * N_BS_ANTS + row * N_BS_ANTS + col];
            smA_re[warp_id][row][col] = __float2half(c.re);
            smA_im[warp_id][row][col] = __float2half(c.im);
        } else {
            smA_re[warp_id][row][col] = __float2half(0.f);
            smA_im[warp_id][row][col] = __float2half(0.f);
        }
    }

    // Load B
    for (int idx = lane; idx < N_BS_ANTS * WN; idx += 32) {
        int ant = idx / WN;
        int sym = idx % WN;
        if (sym < nSym) {
            cf32 y = Y_rx[sc * nSym * N_BS_ANTS + sym * N_BS_ANTS + ant];
            smB_re[warp_id][ant][sym] = __float2half(y.re);
            smB_im[warp_id][ant][sym] = __float2half(y.im);
        } else {
            smB_re[warp_id][ant][sym] = __float2half(0.f);
            smB_im[warp_id][ant][sym] = __float2half(0.f);
        }
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc_rr, acc_ri, acc_ir, acc_ii;

    wmma::fill_fragment(acc_rr, 0.f);
    wmma::fill_fragment(acc_ri, 0.f);
    wmma::fill_fragment(acc_ir, 0.f);
    wmma::fill_fragment(acc_ii, 0.f);

    #pragma unroll
    for (int k = 0; k < N_BS_ANTS; k += WK) {
        wmma::load_matrix_sync(a_frag, &smA_re[warp_id][0][k], N_BS_ANTS);
        wmma::load_matrix_sync(b_frag, &smB_re[warp_id][k][0], WN);
        wmma::mma_sync(acc_rr, a_frag, b_frag, acc_rr);

        wmma::load_matrix_sync(b_frag, &smB_im[warp_id][k][0], WN);
        wmma::mma_sync(acc_ri, a_frag, b_frag, acc_ri);

        wmma::load_matrix_sync(a_frag, &smA_im[warp_id][0][k], N_BS_ANTS);
        wmma::mma_sync(acc_ir, a_frag, b_frag, acc_ir);

        wmma::load_matrix_sync(b_frag, &smB_re[warp_id][k][0], WN);
        wmma::mma_sync(acc_ii, a_frag, b_frag, acc_ii);
    }

    __shared__ float tmp_rr[SC_PER_BLK][WM][WN], tmp_ri[SC_PER_BLK][WM][WN];
    __shared__ float tmp_ir[SC_PER_BLK][WM][WN], tmp_ii[SC_PER_BLK][WM][WN];
    wmma::store_matrix_sync(&tmp_rr[warp_id][0][0], acc_rr, WN, wmma::mem_row_major);
    wmma::store_matrix_sync(&tmp_ri[warp_id][0][0], acc_ri, WN, wmma::mem_row_major);
    wmma::store_matrix_sync(&tmp_ir[warp_id][0][0], acc_ir, WN, wmma::mem_row_major);
    wmma::store_matrix_sync(&tmp_ii[warp_id][0][0], acc_ii, WN, wmma::mem_row_major);
    __syncwarp();

    for (int idx = lane; idx < WM * WN; idx += 32) {
        int layer = idx / WN;
        int sym   = idx % WN;
        if (layer >= N_LAYERS || sym >= nSym) continue;
        float re = tmp_rr[warp_id][layer][sym] - tmp_ir[warp_id][layer][sym];
        float im = tmp_ri[warp_id][layer][sym] + tmp_ii[warp_id][layer][sym];
        Y_eq[sc * N_LAYERS * nSym + layer * nSym + sym] = {re, im};
    }
}

// ── Correctness check ─────────────────────────────────────────────────────────
static void check(const cf32* v1, const cf32* v2, int N,
                  double* cos_out, double* mae_out) {
    double dot=0, na=0, nb=0, mae=0;
    for (int i=0;i<N;i++) {
        double ar=v1[i].re, ai=v1[i].im;
        double br=v2[i].re, bi=v2[i].im;
        dot += ar*br + ai*bi;
        na  += ar*ar + ai*ai;
        nb  += br*br + bi*bi;
        double d = sqrt((ar-br)*(ar-br)+(ai-bi)*(ai-bi));
        if (d > mae) mae = d;
    }
    *cos_out = dot / (sqrt(na)*sqrt(nb) + 1e-30);
    *mae_out = mae;
}

// ── Timing ────────────────────────────────────────────────────────────────────
static double median_ms(float* t, int n) {
    for (int i=1;i<n;i++){float v=t[i];int j=i-1;
        while(j>=0&&t[j]>v){t[j+1]=t[j];j--;}t[j+1]=v;}
    return t[n/2];
}

int main(void) {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("==========================================================\n");
    printf(" Aerial coef_apply: Scalar FP32 vs WMMA FP16 Tensor Core\n");
    printf("==========================================================\n");
    printf(" GPU: %s (%d SMs, SM %d.%d)\n", p.name,
           p.multiProcessorCount, p.major, p.minor);
    printf(" Config: %d PRB × %d SC × %d layers × %d syms × %d ants\n\n",
           N_PRB, N_SC_PRB, N_LAYERS, N_DATA_SYMS, N_BS_ANTS);

    int nSC = N_SC, nSym = N_DATA_SYMS;
    size_t sz_C   = (size_t)nSC * N_LAYERS  * N_BS_ANTS * sizeof(cf32);
    size_t sz_Yrx = (size_t)nSC * nSym      * N_BS_ANTS * sizeof(cf32);
    size_t sz_Yeq = (size_t)nSC * N_LAYERS  * nSym      * sizeof(cf32);

    // Host init
    cf32* h_C   = (cf32*)malloc(sz_C);
    cf32* h_Yrx = (cf32*)malloc(sz_Yrx);
    cf32* h_v1  = (cf32*)malloc(sz_Yeq);
    cf32* h_v2  = (cf32*)malloc(sz_Yeq);
    srand(42);
    for (size_t i=0;i<sz_C/sizeof(cf32);i++)
        h_C[i]={((float)rand()/RAND_MAX-.5f),((float)rand()/RAND_MAX-.5f)};
    for (size_t i=0;i<sz_Yrx/sizeof(cf32);i++)
        h_Yrx[i]={((float)rand()/RAND_MAX-.5f),((float)rand()/RAND_MAX-.5f)};

    cf32* h_v3  = (cf32*)malloc(sz_Yeq);
    cf32 *d_C, *d_Yrx, *d_v1, *d_v2, *d_v3;
    CUDA_CHECK(cudaMalloc(&d_C,   sz_C));
    CUDA_CHECK(cudaMalloc(&d_Yrx, sz_Yrx));
    CUDA_CHECK(cudaMalloc(&d_v1,  sz_Yeq));
    CUDA_CHECK(cudaMalloc(&d_v2,  sz_Yeq));
    CUDA_CHECK(cudaMalloc(&d_v3,  sz_Yeq));
    CUDA_CHECK(cudaMemcpy(d_C,   h_C,   sz_C,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Yrx, h_Yrx, sz_Yrx, cudaMemcpyHostToDevice));

    // ── Kernel configs ───────────────────────────────────────────────────────
    dim3 blk1(N_SC_PRB, N_DATA_SYMS);        // 12×11 = 132 threads
    dim3 grd1(N_PRB, N_LAYERS);              // 132×8 = 1056 blocks

    dim3 blk2(32);                            // 1 warp per block
    dim3 grd2(N_SC);                         // 1 block per subcarrier (1584)

    dim3 blk3(SC_PER_BLK * 32);              // 8 warps = 256 threads
    dim3 grd3((N_SC + SC_PER_BLK - 1) / SC_PER_BLK);  // 198 blocks

    // ── Correctness: run once each ───────────────────────────────────────────
    coef_apply_v1_scalar      <<<grd1,blk1>>>(d_C,d_Yrx,d_v1,nSC,nSym);
    coef_apply_v2_wmma        <<<grd2,blk2>>>(d_C,d_Yrx,d_v2,nSC,nSym);
    coef_apply_v3_wmma_multiw <<<grd3,blk3>>>(d_C,d_Yrx,d_v3,nSC,nSym);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_v1, d_v1, sz_Yeq, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v2, d_v2, sz_Yeq, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v3, d_v3, sz_Yeq, cudaMemcpyDeviceToHost));

    double cos_v, mae_v;
    check(h_v1, h_v2, (int)(sz_Yeq/sizeof(cf32)), &cos_v, &mae_v);
    printf("Correctness (V2 WMMA vs V1 scalar FP32):\n");
    printf("  Cosine similarity : %.8f  (threshold >0.999)\n", cos_v);
    printf("  Max abs error     : %.6f\n", mae_v);
    check(h_v1, h_v3, (int)(sz_Yeq/sizeof(cf32)), &cos_v, &mae_v);
    printf("Correctness (V3 multi-warp vs V1):\n");
    printf("  Cosine similarity : %.8f\n", cos_v);
    printf("  Max abs error     : %.6f\n\n", mae_v);

    // ── Warmup ───────────────────────────────────────────────────────────────
    for (int i=0;i<WARMUP;i++) {
        coef_apply_v1_scalar      <<<grd1,blk1>>>(d_C,d_Yrx,d_v1,nSC,nSym);
        coef_apply_v2_wmma        <<<grd2,blk2>>>(d_C,d_Yrx,d_v2,nSC,nSym);
        coef_apply_v3_wmma_multiw <<<grd3,blk3>>>(d_C,d_Yrx,d_v3,nSC,nSym);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Benchmark alternating ─────────────────────────────────────────────────
    float* t1=(float*)malloc(ITERS*sizeof(float));
    float* t2=(float*)malloc(ITERS*sizeof(float));
    float* t3=(float*)malloc(ITERS*sizeof(float));
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    float ms;

    for (int i=0;i<ITERS;i++) {
        cudaEventRecord(s);
        coef_apply_v1_scalar<<<grd1,blk1>>>(d_C,d_Yrx,d_v1,nSC,nSym);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms,s,e); t1[i]=ms;

        cudaEventRecord(s);
        coef_apply_v2_wmma<<<grd2,blk2>>>(d_C,d_Yrx,d_v2,nSC,nSym);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms,s,e); t2[i]=ms;

        cudaEventRecord(s);
        coef_apply_v3_wmma_multiw<<<grd3,blk3>>>(d_C,d_Yrx,d_v3,nSC,nSym);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms,s,e); t3[i]=ms;
    }

    double m1 = median_ms(t1,ITERS);
    double m2 = median_ms(t2,ITERS);
    double m3 = median_ms(t3,ITERS);

    double flops = (double)nSC * nSym * N_LAYERS * 8.0 * N_BS_ANTS;
    double bytes = (double)(sz_C + sz_Yrx + sz_Yeq);

    printf("Performance (median over %d alternating iterations):\n", ITERS);
    printf("  V1 scalar FP32   [Aerial baseline]  : %7.3f ms  %6.1f GFLOP/s\n",
           m1, flops/m1/1e6);
    printf("  V2 WMMA 1w/SC    [TC, 1584 blocks]  : %7.3f ms  %6.1f GFLOP/s  speedup %.2fx\n",
           m2, flops/m2/1e6, m1/m2);
    printf("  V3 WMMA %dw/blk   [TC, %3d blocks]   : %7.3f ms  %6.1f GFLOP/s  speedup %.2fx\n",
           SC_PER_BLK, (N_SC+SC_PER_BLK-1)/SC_PER_BLK,
           m3, flops/m3/1e6, m1/m3);
    printf("\n  A100 FP16 TC peak 312 TFLOP/s\n");
    printf("  V2 TC utilization: %.2f%%  |  V3 TC utilization: %.2f%%\n",
           flops/m2/1e6/312000.0*100.0, flops/m3/1e6/312000.0*100.0);

    free(h_C); free(h_Yrx); free(h_v1); free(h_v2); free(h_v3);
    free(t1); free(t2); free(t3);
    cudaFree(d_C); cudaFree(d_Yrx); cudaFree(d_v1); cudaFree(d_v2); cudaFree(d_v3);
    return 0;
}
