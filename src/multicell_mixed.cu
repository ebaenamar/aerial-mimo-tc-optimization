/*
 * multicell_mixed.cu — Experiment 2: Mixed-kernel co-scheduling
 *
 * Tests whether kernels with complementary bandwidth profiles can coexist
 * without contention. V1 scalar is L1-bound (93.6% L1, 4.4% DRAM),
 * V2 WMMA is HBM-bound (55.0% L1, 21.2% DRAM).
 *
 * Scenarios:
 *   A) N x V2 only (all HBM-bound) — baseline contention
 *   B) N x V1 only (all L1-bound) — L1 contention
 *   C) 1 V2 + (N-1) x V1 — mixed: 1 HBM-bound + rest L1-bound
 *   D) 1 V1 + (N-1) x V2 — mixed: 1 L1-bound + rest HBM-bound
 *
 * Compile:
 *   nvcc -O3 -arch=sm_80 -std=c++17 multicell_mixed.cu -o multicell_mixed
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

#define WM 16
#define WN 16
#define WK 16
#define N_LAYERS 8
#define N_SC_PRB 12
#define N_ANT 64
#define N_SYM 11
#define N_PRB 132
#define N_SC (N_PRB * N_SC_PRB)
#define WARMUP 50
#define ITERS 200
#define MAX_CELLS 8

struct cf32 { float re, im; };

__device__ __forceinline__ cf32 cma(cf32 a, cf32 b, cf32 acc) {
    return {acc.re + a.re*b.re - a.im*b.im,
            acc.im + a.re*b.im + a.im*b.re};
}

// V1: scalar FP32 — L1-bound (93.6% L1, 4.4% DRAM)
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

// V2: WMMA FP16 — HBM-bound (55.0% L1, 21.2% DRAM)
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

// Gram matrix kernel — L1-bound, small working set
__global__ void gram_kernel(const cf32* M, const cf32* H, cf32* G, int nSC) {
    int sc = blockIdx.x + blockIdx.y * N_SC_PRB;
    if (sc >= nSC) return;
    int row = threadIdx.x, col = threadIdx.y;
    cf32 acc = {0,0};
    const cf32* Mrow = M + sc * N_LAYERS * N_ANT + row * N_ANT;
    const cf32* Hcol = H + sc * N_ANT * N_LAYERS + col;
    #pragma unroll 4
    for (int k = 0; k < N_ANT; k++) {
        cf32 m = Mrow[k];
        cf32 h = Hcol[k * N_LAYERS];
        acc = cma(m, h, acc);
    }
    G[sc * N_LAYERS * N_LAYERS + row * N_LAYERS + col] = acc;
}

double pct(std::vector<float>& v, double p) {
    std::sort(v.begin(), v.end());
    size_t i = std::min(v.size()-1, (size_t)std::ceil(p/100.0*v.size())-1);
    return v[i];
}

struct CellBuf {
    cf32 *dC, *dY, *dO;
    // For gram: M, H, G
    cf32 *dM, *dH, *dG;
};

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int nSC = N_SC;

    printf("=== Experiment 2: Mixed-Kernel Co-scheduling ===\n");
    printf("GPU: %s | %d SMs | %.0f GB/s HBM\n\n", prop.name,
           prop.multiProcessorCount,
           prop.memoryBusWidth*(prop.memoryClockRate/1e6)*2/8.0);

    size_t szC = (size_t)nSC * N_LAYERS * N_ANT * sizeof(cf32);
    size_t szY = (size_t)nSC * N_SYM * N_ANT * sizeof(cf32);
    size_t szO = (size_t)nSC * N_LAYERS * N_SYM * sizeof(cf32);
    size_t szM = szC;  // same shape as C
    size_t szH = (size_t)nSC * N_ANT * N_LAYERS * sizeof(cf32);
    size_t szG = (size_t)nSC * N_LAYERS * N_LAYERS * sizeof(cf32);

    CellBuf cells[MAX_CELLS];
    for (int c = 0; c < MAX_CELLS; c++) {
        CK(cudaMalloc(&cells[c].dC, szC));
        CK(cudaMalloc(&cells[c].dY, szY));
        CK(cudaMalloc(&cells[c].dO, szO));
        CK(cudaMalloc(&cells[c].dM, szM));
        CK(cudaMalloc(&cells[c].dH, szH));
        CK(cudaMalloc(&cells[c].dG, szG));
        cf32 *hC = (cf32*)malloc(szC), *hY = (cf32*)malloc(szY);
        cf32 *hM = (cf32*)malloc(szM), *hH = (cf32*)malloc(szH);
        srand(42 + c*1000);
        auto rf = []() { return ((float)rand()/RAND_MAX - 0.5f) * 1.4f; };
        for (size_t i = 0; i < szC/sizeof(cf32); i++) hC[i] = {rf(), rf()};
        for (size_t i = 0; i < szY/sizeof(cf32); i++) hY[i] = {rf(), rf()};
        for (size_t i = 0; i < szM/sizeof(cf32); i++) hM[i] = {rf(), rf()};
        for (size_t i = 0; i < szH/sizeof(cf32); i++) hH[i] = {rf(), rf()};
        CK(cudaMemcpy(cells[c].dC, hC, szC, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(cells[c].dY, hY, szY, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(cells[c].dM, hM, szM, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(cells[c].dH, hH, szH, cudaMemcpyHostToDevice));
        free(hC); free(hY); free(hM); free(hH);
    }

    cudaStream_t streams[MAX_CELLS];
    cudaEvent_t start[MAX_CELLS], stop[MAX_CELLS];
    for (int c = 0; c < MAX_CELLS; c++) {
        cudaStreamCreate(&streams[c]);
        cudaEventCreate(&start[c]);
        cudaEventCreate(&stop[c]);
    }

    dim3 blk_v1(N_SC_PRB, N_SYM), grd_v1(N_PRB, N_LAYERS);
    dim3 blk_v2(32), grd_v2(nSC);
    dim3 blk_g(N_LAYERS, N_LAYERS), grd_g(N_SC_PRB, N_PRB);

    // Kernel type: 0=V2 WMMA (HBM-bound), 1=V1 scalar (L1-bound), 2=Gram (L1-bound, small)
    auto launch_kernel = [&](int cell_idx, int ktype) {
        auto& cb = cells[cell_idx];
        if (ktype == 0)
            v2_wmma<<<grd_v2, blk_v2, 0, streams[cell_idx]>>>(cb.dC, cb.dY, cb.dO, nSC);
        else if (ktype == 1)
            v1_scalar<<<grd_v1, blk_v1, 0, streams[cell_idx]>>>(cb.dC, cb.dY, cb.dO, nSC);
        else
            gram_kernel<<<grd_g, blk_g, 0, streams[cell_idx]>>>(cb.dM, cb.dH, cb.dG, nSC);
    };

    // Run a scenario: ktypes[0..ncells-1], return per-cell p50 latencies
    auto run_scenario = [&](int ncells, int* ktypes, const char* label) {
        // Warmup
        for (int it = 0; it < WARMUP; it++) {
            for (int c = 0; c < ncells; c++) launch_kernel(c, ktypes[c]);
            CK(cudaDeviceSynchronize());
        }
        // Measure
        std::vector<float> latencies[MAX_CELLS];
        for (int it = 0; it < ITERS; it++) {
            for (int c = 0; c < ncells; c++) cudaEventRecord(start[c], streams[c]);
            for (int c = 0; c < ncells; c++) launch_kernel(c, ktypes[c]);
            for (int c = 0; c < ncells; c++) cudaEventRecord(stop[c], streams[c]);
            CK(cudaDeviceSynchronize());
            float ms;
            for (int c = 0; c < ncells; c++) {
                cudaEventElapsedTime(&ms, start[c], stop[c]);
                latencies[c].push_back(ms);
            }
        }
        // Print
        printf("%-28s", label);
        for (int c = 0; c < ncells; c++) {
            const char* kname = ktypes[c]==0 ? "V2" : (ktypes[c]==1 ? "V1" : "Gr");
            printf("  %s[%d]:%.4f", kname, c, pct(latencies[c], 50));
        }
        printf("\n");
        // Return V2 cell p50 if present
        for (int c = 0; c < ncells; c++)
            if (ktypes[c] == 0) return pct(latencies[c], 50);
        return 0.0;
    };

    printf("Per-cell p50 latency (ms) under concurrent execution:\n");
    printf("Each row = scenario, each cell runs the indicated kernel type\n");
    printf("V2=WMMA(HBM-bound) V1=Scalar(L1-bound) Gr=Gram(L1-bound,small)\n\n");

    // Baselines: single kernel in isolation
    {
        int kt[1] = {0};
        double v2_iso = run_scenario(1, kt, "V2 isolated:");
        printf("  -> V2 isolated p50: %.4f ms (baseline)\n\n", v2_iso);
    }
    {
        int kt[1] = {1};
        double v1_iso = run_scenario(1, kt, "V1 isolated:");
        printf("  -> V1 isolated p50: %.4f ms (baseline)\n\n", v1_iso);
    }
    {
        int kt[1] = {2};
        double gr_iso = run_scenario(1, kt, "Gram isolated:");
        printf("  -> Gram isolated p50: %.4f ms (baseline)\n\n", gr_iso);
    }

    // Scenario A: N x V2 (all HBM-bound) — for N=2,4
    for (int n : {2, 4}) {
        int kt[MAX_CELLS] = {};
        for (int c = 0; c < n; c++) kt[c] = 0;
        char label[64]; snprintf(label, 64, "%dx V2 (all HBM):", n);
        run_scenario(n, kt, label);
    }
    printf("\n");

    // Scenario B: N x V1 (all L1-bound) — for N=2,4
    for (int n : {2, 4}) {
        int kt[MAX_CELLS] = {};
        for (int c = 0; c < n; c++) kt[c] = 1;
        char label[64]; snprintf(label, 64, "%dx V1 (all L1):", n);
        run_scenario(n, kt, label);
    }
    printf("\n");

    // Scenario C: 1 V2 + (N-1) x V1 — mixed
    for (int n : {2, 4}) {
        int kt[MAX_CELLS] = {};
        kt[0] = 0;  // V2
        for (int c = 1; c < n; c++) kt[c] = 1;  // V1
        char label[64]; snprintf(label, 64, "1xV2 + %dxV1 (mixed):", n-1);
        run_scenario(n, kt, label);
    }
    printf("\n");

    // Scenario D: 1 V2 + (N-1) x Gram — mixed with small L1-bound kernel
    for (int n : {2, 4}) {
        int kt[MAX_CELLS] = {};
        kt[0] = 0;  // V2
        for (int c = 1; c < n; c++) kt[c] = 2;  // Gram
        char label[64]; snprintf(label, 64, "1xV2 + %dxGram (mixed):", n-1);
        run_scenario(n, kt, label);
    }
    printf("\n");

    // Scenario E: 2 V2 + 2 V1 — 4 cells mixed
    {
        int kt[4] = {0, 0, 1, 1};
        run_scenario(4, kt, "2xV2 + 2xV1 (4 mixed):");
    }

    // Scenario F: 2 V2 + 2 Gram — 4 cells mixed with small
    {
        int kt[4] = {0, 0, 2, 2};
        run_scenario(4, kt, "2xV2 + 2xGram (4 mixed):");
    }

    printf("\n=== Summary: V2 p50 latency under different co-schedules ===\n");
    printf("Co-schedule               V2_p50(ms)  vs_iso(1.0x)\n");

    // Re-run key scenarios for summary
    {
        int kt[1] = {0};
        printf("%-26s  %.4f      1.00x\n", "V2 isolated", run_scenario(1, kt, ""));
    }
    {
        int kt[2] = {0, 0};
        printf("%-26s  %.4f      %.2fx\n", "2x V2", run_scenario(2, kt, ""), 0.0266/run_scenario(2, kt, ""));
    }
    {
        int kt[2] = {0, 1};
        printf("%-26s  %.4f      %.2fx\n", "1x V2 + 1x V1", run_scenario(2, kt, ""), 0.0266/run_scenario(2, kt, ""));
    }
    {
        int kt[2] = {0, 2};
        printf("%-26s  %.4f      %.2fx\n", "1x V2 + 1x Gram", run_scenario(2, kt, ""), 0.0266/run_scenario(2, kt, ""));
    }
    {
        int kt[4] = {0, 0, 0, 0};
        printf("%-26s  %.4f      %.2fx\n", "4x V2", run_scenario(4, kt, ""), 0.0266/run_scenario(4, kt, ""));
    }
    {
        int kt[4] = {0, 0, 1, 1};
        printf("%-26s  %.4f      %.2fx\n", "2x V2 + 2x V1", run_scenario(4, kt, ""), 0.0266/run_scenario(4, kt, ""));
    }
    {
        int kt[4] = {0, 0, 2, 2};
        printf("%-26s  %.4f      %.2fx\n", "2x V2 + 2x Gram", run_scenario(4, kt, ""), 0.0266/run_scenario(4, kt, ""));
    }

    // Cleanup
    for (int c = 0; c < MAX_CELLS; c++) {
        cudaFree(cells[c].dC); cudaFree(cells[c].dY); cudaFree(cells[c].dO);
        cudaFree(cells[c].dM); cudaFree(cells[c].dH); cudaFree(cells[c].dG);
        cudaStreamDestroy(streams[c]);
        cudaEventDestroy(start[c]); cudaEventDestroy(stop[c]);
    }
    return 0;
}
