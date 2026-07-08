/*
 * srs_wmma_bench.cu — Standalone benchmark: srsFilterMultiply
 *   V1: scalar __hcmadd loop  (active Aerial code, lines 1275-1315)
 *   V2: WMMA FP16 w/ padding  (generalizes the commented-out block, fixes
 *                              nSrsScBlock=12 and handles non-aligned nRxAntSrs)
 *
 * Operation:  H_est[ant, sc_out, port] = Σ_k conj(W[sc_out,k]) * FOCC[port,k] * SRS[ant,k]
 *   → batched complex GEMM: A[nRxAntSrs × K] @ B[K × N]
 *   where K = nSrsScBlock (12 or 24), N = nSrsScBlock * nAntPorts
 *
 * Compile:
 *   nvcc -O3 -arch=sm_80 -o srs_wmma_bench srs_wmma_bench.cu
 */

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

namespace cg = cooperative_groups;
using namespace nvcuda;

// ── config ──────────────────────────────────────────────────────────────────
#define N_RX_ANT    64     // massive MIMO: 64 receive antennas
#define N_ANT_PORTS  4     // SRS ports (typical: 1,2,4)
#define FOCC_LEN     8     // FOCC table length (comb-2)
#define WARMUP      50
#define ITERS      500
// Number of SRS computation blocks per slot (typ. 10–50 in 100MHz TDD)
#define N_COMP_BLOCKS 66

#define CUDA_CHECK(e) do { cudaError_t _=(e); \
  if(_!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_));exit(1);} } while(0)

// complex FP16 helpers
struct ch2 { __half re, im; };

__device__ __forceinline__ ch2 complex_conjmul(ch2 a, ch2 b) {
    // conj(a) * b = (a.re - j*a.im)(b.re + j*b.im)
    return { __hsub(__hmul(a.re,b.re), __hmul(a.im,b.im)),
             __hadd(__hmul(a.re,b.im), __hmul(a.im,b.re)) };
}
__device__ __forceinline__ ch2 hcmadd(ch2 prod, ch2 sig, ch2 acc) {
    // acc += prod * sig
    return { __hfma(prod.re, sig.re, __hfma(__hneg(prod.im), sig.im, acc.re)),
             __hfma(prod.re, sig.im, __hfma(prod.im, sig.re, acc.im)) };
}

// ══════════════════════════════════════════════════════════════════════════════
// V1 BASELINE — scalar loop, exact Aerial pattern (no delay offset for clarity)
// ══════════════════════════════════════════════════════════════════════════════
template<int K, int N_PORTS, int FOCC_LENGTH>
__global__ void srs_filter_v1_scalar(
    ch2* __restrict__ H_est,        // [nBlocks, nAnt, K, N_PORTS]
    const ch2* __restrict__ SRS,    // [nBlocks, nAnt, K]
    const ch2* __restrict__ W,      // [K, K]  (filter matrix)
    const ch2* __restrict__ FOCC,   // [N_PORTS, FOCC_LENGTH]
    int nAnt, int nBlocks)
{
    int bid  = blockIdx.z;               // computation block
    int tid  = threadIdx.x;
    int bsz  = blockDim.x;

    const ch2* srs_blk = SRS  + bid * nAnt * K;
    ch2*       out_blk = H_est + bid * nAnt * K * N_PORTS;

    int total = nAnt * K * N_PORTS;
    for (int i = tid; i < total; i += bsz) {
        int portIdx = i % N_PORTS;
        int scIdx   = (i / N_PORTS) % K;
        int antIdx  = i / (N_PORTS * K);

        const ch2* srs = srs_blk + antIdx * K;
        const ch2* w   = W + scIdx * K;
        const ch2* focc_row = FOCC + portIdx * FOCC_LENGTH;

        ch2 est = {__float2half(0.f), __float2half(0.f)};
        for (int k = 0; k < K; k++) {
            ch2 focc_k  = focc_row[k % FOCC_LENGTH];
            ch2 wf      = complex_conjmul(w[k], focc_k);
            est         = hcmadd(wf, srs[k], est);
        }
        out_blk[i] = est;
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// V2 WMMA — generalizes the commented-out Aerial approach with zero-padding
//
// Strategy (same Re/Im decomposition as NVIDIA's original):
//   B_smem layout: for each (scIdx, portIdx) output pair, 2 rows in B:
//     row 2p:   Re(conj(W)*FOCC) — gives Re(H_est) after matmul
//     row 2p+1: Im(conj(W)*FOCC) — gives Im(H_est)
//   A: SRS[ant, k] as row-major FP16 complex (= 2*K real elements per row)
//   Padding: K_pad = ceil(K/8)*8*2 half-elements per row, zeros beyond K*2
// ══════════════════════════════════════════════════════════════════════════════
template<int K, int N_PORTS, int FOCC_LENGTH>
__global__ void __launch_bounds__(256) srs_filter_v2_wmma(
    ch2* __restrict__ H_est,
    const ch2* __restrict__ SRS,
    const ch2* __restrict__ W,
    const ch2* __restrict__ FOCC,
    int nAnt, int nBlocks)
{
    // WMMA tile: m=16, n=16, k=16 FP16
    constexpr int M = 16, N = 16, MMA_K = 16;
    // Pad K*2 halfs up to next multiple of 16
    constexpr int K2      = K * 2;                          // real halfs per row
    constexpr int K2_PAD  = ((K2 + MMA_K - 1) / MMA_K) * MMA_K;  // padded
    constexpr int MMA_STEPS = K2_PAD / MMA_K;              // MMA iterations

    // N_B_COLS_REAL = 2 * N_PORTS (real + imag per port)
    constexpr int NBC = 2 * K * N_PORTS;   // B columns: Re+Im per (sc_out, port)
    // Pad NBC to multiple of 16
    constexpr int NBC_PAD = ((NBC + N - 1) / N) * N;
    constexpr int N_B_FRAGS = NBC_PAD / N;   // # of B column-tile fragments

    // smem: A[M × K2_PAD] + B[K2_PAD × NBC_PAD], both FP16
    __shared__ __half smA[M][K2_PAD];   // 16 × K2_PAD half
    __shared__ __half smB[K2_PAD][NBC_PAD]; // K2_PAD × NBC_PAD half
    __shared__ __half smC[8][M * N];            // per-warp WMMA store target

    int warp_id  = threadIdx.x / 32;
    int lane     = threadIdx.x % 32;
    int n_warps  = blockDim.x / 32;
    int tid      = threadIdx.x;
    int bsz      = blockDim.x;
    int bid      = blockIdx.z;

    const ch2* srs_blk = SRS   + bid * nAnt * K;
    ch2*       out_blk = H_est + bid * nAnt * K * N_PORTS;

    // Number of 16-row tiles of antennas
    int n_ant_tiles = (nAnt + M - 1) / M;

    // ── Build B in smem ONCE (constant across all antenna tiles) ──
    for (int i = tid; i < K2_PAD * NBC_PAD; i += bsz)
        smB[i / NBC_PAD][i % NBC_PAD] = __float2half(0.f);
    for (int i = tid; i < K * N_PORTS; i += bsz) {
        int scIdx   = i / N_PORTS;
        int portIdx = i % N_PORTS;
        const ch2* w_row    = W + scIdx * K;
        const ch2* focc_row = FOCC + portIdx * FOCC_LENGTH;
        for (int k = 0; k < K; k++) {
            ch2 wf = complex_conjmul(w_row[k], focc_row[k % FOCC_LENGTH]);
            int out_col_re = 2 * (scIdx * N_PORTS + portIdx);
            smB[k * 2    ][out_col_re]     = wf.re;
            smB[k * 2    ][out_col_re + 1] = wf.im;
            smB[k * 2 + 1][out_col_re]     = __hneg(wf.im);
            smB[k * 2 + 1][out_col_re + 1] = wf.re;
        }
    }
    __syncthreads();

    for (int at = 0; at < n_ant_tiles; at++) {
        int ant_base = at * M;

        // ── Build A in smem: rows=antennas [16], cols=2*K halfs (interleaved Re,Im) ──
        // Zero-initialize for padding
        for (int i = tid; i < M * K2_PAD; i += bsz)
            smA[i / K2_PAD][i % K2_PAD] = __float2half(0.f);

        __syncthreads();

        // Fill A: SRS[ant_base..ant_base+M-1, 0..K-1] → smA[0..M-1, 0..2K-1]
        for (int i = tid; i < M * K; i += bsz) {
            int row = i / K, col = i % K;
            int ant = ant_base + row;
            if (ant < nAnt) {
                ch2 v = srs_blk[ant * K + col];
                smA[row][col * 2]     = v.re;
                smA[row][col * 2 + 1] = v.im;
            }
        }

        __syncthreads();

        // ── WMMA: each warp computes one [M×N] output tile ──
        // Total output tiles: n_B_frags = NBC/N (column tiles)
        // Assign warps round-robin over column tiles
        for (int bf = warp_id; bf < N_B_FRAGS; bf += n_warps) {
            wmma::fragment<wmma::matrix_a, M, N, MMA_K, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, M, N, MMA_K, __half, wmma::row_major> b_frag;
            wmma::fragment<wmma::accumulator, M, N, MMA_K, __half> c_frag;

            wmma::fill_fragment(c_frag, __float2half(0.f));

            for (int s = 0; s < MMA_STEPS; s++) {
                // A tile: rows=antenna tile, cols=MMA_K starting at s*MMA_K
                wmma::load_matrix_sync(a_frag,
                    &smA[0][s * MMA_K], K2_PAD);
                // B tile: rows=MMA_K starting at s*MMA_K, cols=N starting at bf*N
                wmma::load_matrix_sync(b_frag,
                    &smB[s * MMA_K][bf * N], NBC_PAD);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            // Store: C[M × N] → H_est[ant_tile × (K*N_PORTS)]
            // C layout: rows=antennas, cols=output (scIdx, portIdx) Re/Im interleaved
            // Each pair of columns (2p, 2p+1) = (Re, Im) of H_est[ant, sc, port]
            wmma::store_matrix_sync(smC[warp_id], c_frag, N, wmma::mem_row_major);

            // Scatter tmp back to H_est[ant, sc_out, port]
            int col_base = bf * N;  // which NBC columns this fragment covers
            for (int i = lane; i < M * N; i += 32) {
                int row = i / N;       // antenna index within tile
                int col = i % N;       // column in B output space
                int abs_col = col_base + col;
                if (abs_col >= NBC) continue;
                int portIdx = (abs_col / 2) % N_PORTS;
                int scIdx   = (abs_col / 2) / N_PORTS;
                int is_imag = abs_col & 1;

                int ant = ant_base + row;
                if (ant >= nAnt || scIdx >= K) continue;

                int out_idx = ant * K * N_PORTS + scIdx * N_PORTS + portIdx;
                if (is_imag)
                    out_blk[out_idx].im = smC[warp_id][i];
                else
                    out_blk[out_idx].re = smC[warp_id][i];
            }
        }
        __syncthreads();
    }
}

// ── timing helper ────────────────────────────────────────────────────────────
double median_ms(float* times, int n) {
    // simple insertion sort on small array
    for (int i = 1; i < n; i++) {
        float k = times[i]; int j = i-1;
        while (j >= 0 && times[j] > k) { times[j+1]=times[j]; j--; }
        times[j+1] = k;
    }
    return times[n/2];
}

// ── correctness ──────────────────────────────────────────────────────────────
double cosine_sim_h(const ch2* a, const ch2* b, int n) {
    double dot=0, na=0, nb=0;
    for (int i=0;i<n;i++) {
        float ar=__half2float(a[i].re), ai=__half2float(a[i].im);
        float br=__half2float(b[i].re), bi=__half2float(b[i].im);
        dot += ar*br + ai*bi;
        na  += ar*ar + ai*ai;
        nb  += br*br + bi*bi;
    }
    return dot / (sqrt(na)*sqrt(nb) + 1e-12);
}

// ── run one config ────────────────────────────────────────────────────────────
template<int K, int N_PORTS, int FOCC_LENGTH>
void run_config(const char* name) {
    int nAnt    = N_RX_ANT;
    int nBlocks = N_COMP_BLOCKS;

    size_t sz_srs  = (size_t)nBlocks * nAnt * K     * sizeof(ch2);
    size_t sz_W    = (size_t)K * K                   * sizeof(ch2);
    size_t sz_focc = (size_t)N_PORTS * FOCC_LENGTH   * sizeof(ch2);
    size_t sz_out  = (size_t)nBlocks * nAnt * K * N_PORTS * sizeof(ch2);

    ch2 *h_srs=(ch2*)malloc(sz_srs), *h_W=(ch2*)malloc(sz_W),
        *h_focc=(ch2*)malloc(sz_focc);
    ch2 *h_v1=(ch2*)malloc(sz_out), *h_v2=(ch2*)malloc(sz_out);

    // Random FP16 inputs (unit magnitude)
    srand(42);
    auto rh = []{ return __float2half(((float)rand()/RAND_MAX - 0.5f) * 1.4f); };
    for (size_t i=0; i<sz_srs/sizeof(ch2); i++)  h_srs[i]  = {rh(),rh()};
    for (size_t i=0; i<sz_W/sizeof(ch2);   i++)  h_W[i]    = {rh(),rh()};
    for (size_t i=0; i<sz_focc/sizeof(ch2);i++)  h_focc[i] = {rh(),rh()};

    ch2 *d_srs, *d_W, *d_focc, *d_v1, *d_v2;
    CUDA_CHECK(cudaMalloc(&d_srs,  sz_srs));
    CUDA_CHECK(cudaMalloc(&d_W,    sz_W));
    CUDA_CHECK(cudaMalloc(&d_focc, sz_focc));
    CUDA_CHECK(cudaMalloc(&d_v1,   sz_out));
    CUDA_CHECK(cudaMalloc(&d_v2,   sz_out));
    CUDA_CHECK(cudaMemcpy(d_srs,  h_srs,  sz_srs,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W,    h_W,    sz_W,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_focc, h_focc, sz_focc, cudaMemcpyHostToDevice));

    dim3 blk1(256), grd1(1, 1, nBlocks);
    dim3 blk2(256), grd2(1, 1, nBlocks);

    // Warmup
    for (int i=0;i<WARMUP;i++) {
        srs_filter_v1_scalar<K,N_PORTS,FOCC_LENGTH><<<grd1,blk1>>>(
            d_v1, d_srs, d_W, d_focc, nAnt, nBlocks);
        srs_filter_v2_wmma<K,N_PORTS,FOCC_LENGTH><<<grd2,blk2>>>(
            d_v2, d_srs, d_W, d_focc, nAnt, nBlocks);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark — alternating to remove run-order bias
    float t1s[ITERS], t2s[ITERS];
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    for (int i=0;i<ITERS;i++) {
        float ms;
        cudaEventRecord(s);
        srs_filter_v1_scalar<K,N_PORTS,FOCC_LENGTH><<<grd1,blk1>>>(
            d_v1, d_srs, d_W, d_focc, nAnt, nBlocks);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms,s,e); t1s[i]=ms;

        cudaEventRecord(s);
        srs_filter_v2_wmma<K,N_PORTS,FOCC_LENGTH><<<grd2,blk2>>>(
            d_v2, d_srs, d_W, d_focc, nAnt, nBlocks);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms,s,e); t2s[i]=ms;
    }

    double m1 = median_ms(t1s, ITERS);
    double m2 = median_ms(t2s, ITERS);

    // Copy back for correctness
    CUDA_CHECK(cudaMemcpy(h_v1, d_v1, sz_out, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v2, d_v2, sz_out, cudaMemcpyDeviceToHost));

    // Ops: nBlocks * nAnt * K * N_PORTS * K * 8 (complex MAC = 8 real ops)
    double ops = (double)nBlocks * nAnt * K * N_PORTS * K * 8.0;
    double cos = cosine_sim_h(h_v1, h_v2, sz_out/sizeof(ch2));

    printf("%-30s  V1: %6.3f ms  %6.1f GFLOP/s  |  V2: %6.3f ms  %6.1f GFLOP/s"
           "  speedup: %.2fx  cosine: %.8f\n",
           name, m1, ops/m1/1e6, m2, ops/m2/1e6, m1/m2, cos);

    free(h_srs); free(h_W); free(h_focc); free(h_v1); free(h_v2);
    cudaFree(d_srs); cudaFree(d_W); cudaFree(d_focc); cudaFree(d_v1); cudaFree(d_v2);
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("==========================================================\n");
    printf(" SRS srsFilterMultiply: Scalar FP16 vs WMMA FP16 TC\n");
    printf("==========================================================\n");
    printf(" GPU: %s (SM %d.%d)  nAnt=%d  nBlocks=%d  iters=%d\n\n",
           p.name, p.major, p.minor, N_RX_ANT, N_COMP_BLOCKS, ITERS);

    // comb-2: K=24, 4 ports, FOCC_LEN=8
    run_config<24, 4, 8>("comb-2 K=24 nPorts=4");
    // comb-2: K=24, 2 ports
    run_config<24, 2, 8>("comb-2 K=24 nPorts=2");
    // comb-4: K=12, 4 ports, FOCC_LEN=12
    run_config<12, 4, 12>("comb-4 K=12 nPorts=4");
    // comb-4: K=12, 2 ports
    run_config<12, 2, 12>("comb-4 K=12 nPorts=2");

    return 0;
}
