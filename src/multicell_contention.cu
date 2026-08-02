/*
 * multicell_contention.cu — Experiment 1: Multi-cell memory contention
 *
 * Launches N concurrent instances of the V2 WMMA coefficient application
 * kernel on N separate CUDA streams, each with independent input buffers.
 * Measures per-cell latency as N increases from 1 to MAX_CELLS.
 *
 * Each "cell" = one independent equalization kernel (same config as coef_rigor).
 *
 * Compile:
 *   nvcc -O3 -arch=sm_80 -std=c++17 multicell_contention.cu -o multicell_contention
 *
 * Run:
 *   ./multicell_contention
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

using namespace nvcuda;

#define CK(e) do{cudaError_t _e=(e);if(_e){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);}}while(0)

// ── Config (same as coef_rigor.cu) ────────────────────────────────────────────
#define WM 16
#define WN 16
#define WK 16
#define N_LAYERS 8
#define N_SC_PRB 12
#define N_ANT 64
#define N_SYM 11
#define N_PRB 132
#define N_SC (N_PRB * N_SC_PRB)  // 1584
#define WARMUP 50
#define ITERS 200
#define MAX_CELLS 8

struct cf32 { float re, im; };

__device__ __forceinline__ cf32 cma(cf32 a, cf32 b, cf32 acc) {
    return {acc.re + a.re*b.re - a.im*b.im,
            acc.im + a.re*b.im + a.im*b.re};
}

// ── V1: scalar baseline (same as coef_rigor.cu) ───────────────────────────────
__global__ void v1_scalar(const cf32* C, const cf32* Y, cf32* O, int nSC) {
    int prb = blockIdx.x, layer = blockIdx.y, sc_l = threadIdx.x, sym = threadIdx.y;
    int sc = prb * N_SC_PRB + sc_l;
    if (sc >= nSC) return;
    __shared__ cf32 sC[N_ANT][N_SC_PRB];
    if (sym == 0) { for (int a = 0; a < N_ANT; a++) sC[a][sc_l] = C[sc*N_LAYERS*N_ANT + layer*N_ANT + a]; }
    __syncthreads();
    cf32 acc = {0,0};
    cf32 Cn = sC[0][sc_l];
    cf32 Yn = Y[sc*N_SYM*N_ANT + sym*N_ANT + 0];
    #pragma unroll 4
    for (int a = 0; a+1 < N_ANT; a++) {
        acc = cma(Cn, Yn, acc);
        Cn = sC[a+1][sc_l];
        Yn = Y[sc*N_SYM*N_ANT + sym*N_ANT + a+1];
    }
    acc = cma(Cn, Yn, acc);
    O[sc*N_LAYERS*N_SYM + layer*N_SYM + sym] = acc;
}

// ── V2: WMMA FP16 Tensor Core (same as coef_rigor.cu) ─────────────────────────
__global__ void v2_wmma(const cf32* C, const cf32* Y, cf32* O, int nSC) {
    constexpr int AP = ((N_ANT + WK - 1) / WK) * WK;
    int sc = blockIdx.x, lane = threadIdx.x;
    if (sc >= nSC) return;
    __shared__ __half Ar[WM][AP], Ai[WM][AP], Br[AP][WN], Bi[AP][WN];
    for (int i = lane; i < WM*AP; i += 32) {
        int r = i/AP, c = i%AP;
        if (r < N_LAYERS && c < N_ANT) {
            cf32 v = C[sc*N_LAYERS*N_ANT + r*N_ANT + c];
            Ar[r][c] = __float2half(v.re); Ai[r][c] = __float2half(v.im);
        } else { Ar[r][c] = 0; Ai[r][c] = 0; }
    }
    for (int i = lane; i < AP*WN; i += 32) {
        int a = i/WN, s = i%WN;
        if (a < N_ANT && s < N_SYM) {
            cf32 v = Y[sc*N_SYM*N_ANT + s*N_ANT + a];
            Br[a][s] = __float2half(v.re); Bi[a][s] = __float2half(v.im);
        } else { Br[a][s] = 0; Bi[a][s] = 0; }
    }
    __syncwarp();
    wmma::fragment<wmma::matrix_a,WM,WN,WK,__half,wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,WM,WN,WK,__half,wmma::row_major> bf;
    wmma::fragment<wmma::accumulator,WM,WN,WK,float> rr,ri,ir,ii;
    wmma::fill_fragment(rr,0); wmma::fill_fragment(ri,0);
    wmma::fill_fragment(ir,0); wmma::fill_fragment(ii,0);
    #pragma unroll
    for (int k = 0; k < AP; k += WK) {
        wmma::load_matrix_sync(af, &Ar[0][k], AP);
        wmma::load_matrix_sync(bf, &Br[k][0], WN); wmma::mma_sync(rr,af,bf,rr);
        wmma::load_matrix_sync(bf, &Bi[k][0], WN); wmma::mma_sync(ri,af,bf,ri);
        wmma::load_matrix_sync(af, &Ai[0][k], AP); wmma::mma_sync(ir,af,bf,ir);
        wmma::load_matrix_sync(bf, &Br[k][0], WN); wmma::mma_sync(ii,af,bf,ii);
    }
    __shared__ float t[4][WM][WN];
    wmma::store_matrix_sync(&t[0][0][0], rr, WN, wmma::mem_row_major);
    wmma::store_matrix_sync(&t[1][0][0], ri, WN, wmma::mem_row_major);
    wmma::store_matrix_sync(&t[2][0][0], ir, WN, wmma::mem_row_major);
    wmma::store_matrix_sync(&t[3][0][0], ii, WN, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < WM*WN; i += 32) {
        int l = i/WN, s = i%WN;
        if (l >= N_LAYERS || s >= N_SYM) continue;
        O[sc*N_LAYERS*N_SYM + l*N_SYM + s] = {t[0][l][s]-t[2][l][s], t[1][l][s]+t[3][l][s]};
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
double pct(std::vector<float>& v, double p) {
    std::sort(v.begin(), v.end());
    size_t i = std::min(v.size()-1, (size_t)std::ceil(p/100.0*v.size())-1);
    return v[i];
}

// Per-cell buffer struct
struct CellBuffers {
    cf32 *dC, *dY, *dO;
    size_t szC, szY, szO;
};

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int nSC = N_SC;
    printf("=== Multi-Cell Memory Contention Experiment ===\n");
    printf("GPU: %s | %d SMs | %.0f GB/s HBM\n\n", prop.name,
           prop.multiProcessorCount,
           prop.memoryBusWidth*(prop.memoryClockRate/1e6)*2/8.0);
    printf("Config per cell: %d ant, %d RB, %d layers, %d sym, nSC=%d\n",
           N_ANT, N_PRB, N_LAYERS, N_SYM, nSC);
    printf("Iters: %d (after %d warmup)\n\n", ITERS, WARMUP);

    size_t szC = (size_t)nSC * N_LAYERS * N_ANT * sizeof(cf32);
    size_t szY = (size_t)nSC * N_SYM * N_ANT * sizeof(cf32);
    size_t szO = (size_t)nSC * N_LAYERS * N_SYM * sizeof(cf32);

    // Allocate MAX_CELLS independent buffer sets
    CellBuffers cells[MAX_CELLS];
    for (int c = 0; c < MAX_CELLS; c++) {
        cells[c].szC = szC; cells[c].szY = szY; cells[c].szO = szO;
        CK(cudaMalloc(&cells[c].dC, szC));
        CK(cudaMalloc(&cells[c].dY, szY));
        CK(cudaMalloc(&cells[c].dO, szO));
        // Fill with random data (different seed per cell)
        cf32 *hC = (cf32*)malloc(szC);
        cf32 *hY = (cf32*)malloc(szY);
        srand(42 + c*1000);
        auto rf = []() { return ((float)rand()/RAND_MAX - 0.5f) * 1.4f; };
        for (size_t i = 0; i < szC/sizeof(cf32); i++) hC[i] = {rf(), rf()};
        for (size_t i = 0; i < szY/sizeof(cf32); i++) hY[i] = {rf(), rf()};
        CK(cudaMemcpy(cells[c].dC, hC, szC, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(cells[c].dY, hY, szY, cudaMemcpyHostToDevice));
        free(hC); free(hY);
    }

    // Create MAX_CELLS streams
    cudaStream_t streams[MAX_CELLS];
    for (int c = 0; c < MAX_CELLS; c++)
        CK(cudaStreamCreate(&streams[c]));

    // Kernel configs
    dim3 blk_v1(N_SC_PRB, N_SYM), grd_v1(N_PRB, N_LAYERS);
    dim3 blk_v2(32), grd_v2(nSC);

    // CUDA events for per-cell timing
    cudaEvent_t start[MAX_CELLS], stop[MAX_CELLS];
    for (int c = 0; c < MAX_CELLS; c++) {
        cudaEventCreate(&start[c]);
        cudaEventCreate(&stop[c]);
    }

    // Global start/stop for total batch time
    cudaEvent_t gstart, gstop;
    cudaEventCreate(&gstart);
    cudaEventCreate(&gstop);

    printf("%-6s  %-12s  %-12s  %-12s  %-12s  %-12s  %-12s\n",
           "Ncells", "V2_p5(ms)", "V2_p50(ms)", "V2_p95(ms)", "V2_mean(ms)",
           "speedup_v1", "total(ms)");
    printf("-------------------------------------------------------------------------------------\n");

    // For each cell count, measure both V1 and V2
    for (int ncells = 1; ncells <= MAX_CELLS; ncells++) {

        // ── Warmup ─────────────────────────────────────────────────────────────
        for (int it = 0; it < WARMUP; it++) {
            for (int c = 0; c < ncells; c++) {
                v2_wmma<<<grd_v2, blk_v2, 0, streams[c]>>>(
                    cells[c].dC, cells[c].dY, cells[c].dO, nSC);
            }
            CK(cudaDeviceSynchronize());
        }

        // ── Measure V2 (concurrent) ────────────────────────────────────────────
        std::vector<float> per_cell_latencies;
        std::vector<float> total_latencies;

        for (int it = 0; it < ITERS; it++) {
            // Record per-cell start/stop on each stream
            for (int c = 0; c < ncells; c++)
                cudaEventRecord(start[c], streams[c]);

            for (int c = 0; c < ncells; c++) {
                v2_wmma<<<grd_v2, blk_v2, 0, streams[c]>>>(
                    cells[c].dC, cells[c].dY, cells[c].dO, nSC);
            }

            for (int c = 0; c < ncells; c++)
                cudaEventRecord(stop[c], streams[c]);

            // Global timing
            cudaEventRecord(gstart);
            for (int c = 0; c < ncells; c++) {
                v2_wmma<<<grd_v2, blk_v2, 0, streams[c]>>>(
                    cells[c].dC, cells[c].dY, cells[c].dO, nSC);
            }
            cudaEventRecord(gstop);
            CK(cudaDeviceSynchronize());

            // Collect per-cell latencies
            float ms;
            for (int c = 0; c < ncells; c++) {
                cudaEventElapsedTime(&ms, start[c], stop[c]);
                per_cell_latencies.push_back(ms);
            }
            cudaEventElapsedTime(&ms, gstart, gstop);
            total_latencies.push_back(ms);
        }

        // ── Measure V1 (concurrent) for speedup reference ──────────────────────
        std::vector<float> v1_per_cell;
        for (int it = 0; it < WARMUP; it++) {
            for (int c = 0; c < ncells; c++) {
                v1_scalar<<<grd_v1, blk_v1, 0, streams[c]>>>(
                    cells[c].dC, cells[c].dY, cells[c].dO, nSC);
            }
            CK(cudaDeviceSynchronize());
        }

        for (int it = 0; it < ITERS; it++) {
            for (int c = 0; c < ncells; c++)
                cudaEventRecord(start[c], streams[c]);
            for (int c = 0; c < ncells; c++) {
                v1_scalar<<<grd_v1, blk_v1, 0, streams[c]>>>(
                    cells[c].dC, cells[c].dY, cells[c].dO, nSC);
            }
            for (int c = 0; c < ncells; c++)
                cudaEventRecord(stop[c], streams[c]);
            CK(cudaDeviceSynchronize());
            float ms;
            for (int c = 0; c < ncells; c++) {
                cudaEventElapsedTime(&ms, start[c], stop[c]);
                v1_per_cell.push_back(ms);
            }
        }

        // ── Stats ──────────────────────────────────────────────────────────────
        double v2_p5 = pct(per_cell_latencies, 5);
        double v2_p50 = pct(per_cell_latencies, 50);
        double v2_p95 = pct(per_cell_latencies, 95);
        double v2_mean = 0;
        for (float v : per_cell_latencies) v2_mean += v;
        v2_mean /= per_cell_latencies.size();

        double v1_p50 = pct(v1_per_cell, 50);
        double speedup = v1_p50 / v2_p50;

        double total_p50 = pct(total_latencies, 50);

        printf("%-6d  %-12.4f  %-12.4f  %-12.4f  %-12.4f  %-12.2f  %-12.4f\n",
               ncells, v2_p5, v2_p50, v2_p95, v2_mean, speedup, total_p50);
    }

    printf("\n");

    // ── Bandwidth analysis ─────────────────────────────────────────────────────
    double flops = 8.0 * N_SC * N_LAYERS * N_SYM * N_ANT;
    double bytes = szC + szY + szO;
    printf("Per-cell: FLOPs=%.3e  bytes=%.3e  AI=%.2f FLOP/byte\n",
           flops, bytes, flops/bytes);
    printf("Per-cell working set: %.1f MB (C=%.1f Y=%.1f O=%.1f)\n",
           (szC+szY+szO)/1e6, szC/1e6, szY/1e6, szO/1e6);
    printf("A100 HBM: %.0f GB/s  L2: 40 MB\n\n",
           prop.memoryBusWidth*(prop.memoryClockRate/1e6)*2/8.0);

    printf("Projected bandwidth saturation:\n");
    double hbm_peak = prop.memoryBusWidth*(prop.memoryClockRate/1e6)*2/8.0;
    double single_cell_bw = 627.0;  // measured from coef_rigor
    for (int n = 1; n <= MAX_CELLS; n++) {
        printf("  %d cells: working set=%.1f MB, projected BW demand=%.0f GB/s (%.0f%% of HBM)\n",
               n, n * (szC+szY+szO)/1e6, n * single_cell_bw, n * single_cell_bw / hbm_peak * 100);
    }

    // Cleanup
    for (int c = 0; c < MAX_CELLS; c++) {
        cudaFree(cells[c].dC);
        cudaFree(cells[c].dY);
        cudaFree(cells[c].dO);
        cudaStreamDestroy(streams[c]);
        cudaEventDestroy(start[c]);
        cudaEventDestroy(stop[c]);
    }
    cudaEventDestroy(gstart);
    cudaEventDestroy(gstop);

    return 0;
}
