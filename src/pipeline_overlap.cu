/*
 * pipeline_overlap.cu — Experiment 4: Pipeline-aware stage scheduling
 *
 * Tests the core hypothesis of pipeline-aware scheduling:
 *   When cell 1 is in stage X and cell 2 is in stage Y (X != Y),
 *   do they contend less than when both are in the same stage?
 *
 * Pipeline stages per cell:
 *   Stage 0: Gram matrix compute  (L1-bound, WS=0.5MB, ~0.035ms)
 *   Stage 1: Coefficient apply    (HBM-bound, WS=16.5MB, ~0.027ms)
 *   Stage 2: LLR soft demapper    (L1-bound, WS=1.1MB, ~0.030ms)
 *
 * Scenarios for 2 cells:
 *   - Same stage: both in Gram, both in coef, both in LLR
 *   - Cross stage: cell1 Gram + cell2 coef, cell1 coef + cell2 LLR, etc.
 *
 * Also tests 4-cell pipeline rotation:
 *   Cell1:Gram, Cell2:Coef, Cell3:LLR, Cell4:Coef (staggered pipeline)
 *
 * Compile:
 *   nvcc -O3 -arch=sm_80 -std=c++17 pipeline_overlap.cu -o pipeline_overlap
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

// ── Config ────────────────────────────────────────────────────────────────────
#define WM 16
#define WN 16
#define WK 16
#define N_LAYERS 8
#define N_SC_PRB 12
#define N_ANT 64
#define N_SYM 11
#define N_PRB 132
#define N_SC (N_PRB * N_SC_PRB)  // 1584
#define N_MOD_ORDER 8  // 256-QAM → 8 bits per symbol
#define WARMUP 50
#define ITERS 300
#define MAX_CELLS 8

struct cf32 { float re, im; };

__device__ __forceinline__ cf32 cma(cf32 a, cf32 b, cf32 acc) {
    return {acc.re + a.re*b.re - a.im*b.im,
            acc.im + a.re*b.im + a.im*b.re};
}

// ══════════════════════════════════════════════════════════════════════════════
// STAGE 0: Gram matrix compute — G[L×L] = M[L×A] @ H[A×L]
// L1-bound, small working set (~0.5 MB per cell)
// Grid: (N_SC_PRB, N_PRB)  Block: (N_LAYERS, N_LAYERS) = (8,8)
// ══════════════════════════════════════════════════════════════════════════════
__global__ void k_gram(
    const cf32* __restrict__ M,   // [nSC, N_LAYERS, N_ANT]
    const cf32* __restrict__ H,   // [nSC, N_ANT, N_LAYERS]
    cf32*       __restrict__ G,   // [nSC, N_LAYERS, N_LAYERS]
    int nSC)
{
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

// ══════════════════════════════════════════════════════════════════════════════
// STAGE 1: Coefficient apply (WMMA FP16) — Y_eq = C @ Y_rx
// HBM-bound, large working set (~16.5 MB per cell)
// Grid: (nSC,)  Block: (32,) — 1 warp per subcarrier
// ══════════════════════════════════════════════════════════════════════════════
__global__ void k_coef(
    const cf32* __restrict__ C,   // [nSC, N_LAYERS, N_ANT]
    const cf32* __restrict__ Y,   // [nSC, N_SYM, N_ANT]
    cf32*       __restrict__ O,   // [nSC, N_LAYERS, N_SYM]
    int nSC)
{
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

// ══════════════════════════════════════════════════════════════════════════════
// STAGE 2: LLR soft demapper — approximate log-likelihood ratios
// L1-bound, small working set (~1.1 MB per cell)
// Grid: (N_PRB, N_LAYERS)  Block: (N_SC_PRB, N_SYM)
// Each thread: 1 subcarrier × 1 symbol → N_MOD_ORDER LLR values
// ══════════════════════════════════════════════════════════════════════════════
__global__ void k_llr(
    const cf32* __restrict__ Y_eq,  // [nSC, N_LAYERS, N_SYM]
    float*      __restrict__ LLR,   // [nSC, N_LAYERS, N_SYM, N_MOD_ORDER]
    int nSC, int nSym)
{
    int prb = blockIdx.x, layer = blockIdx.y;
    int sc_l = threadIdx.x, sym = threadIdx.y;
    int sc = prb * N_SC_PRB + sc_l;
    if (sc >= nSC || sym >= nSym) return;

    // Load equalized symbol
    cf32 y = Y_eq[sc * N_LAYERS * nSym + layer * nSym + sym];

    // Simple max-log demapper for 256-QAM (8 bits)
    // Normalization factor (approximate channel gain)
    float norm = 1.0f / (sqrtf(y.re*y.re + y.im*y.im) + 1e-6f);
    float yr = y.re * norm;
    float yi = y.im * norm;

    // 8 LLR values (one per bit position in 256-QAM)
    // Simplified: use sign-based approximation
    float* out = LLR + sc * N_LAYERS * nSym * N_MOD_ORDER
                     + layer * nSym * N_MOD_ORDER
                     + sym * N_MOD_ORDER;
    // Bits 0-3 from real part, bits 4-7 from imag part
    float levels[4] = {0.3f, 0.7f, 1.1f, 1.5f};
    for (int b = 0; b < 4; b++) {
        out[b]     = (yr > levels[b]) ? levels[b] : -levels[b];
        out[b + 4] = (yi > levels[b]) ? levels[b] : -levels[b];
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
double pct(std::vector<float>& v, double p) {
    std::sort(v.begin(), v.end());
    size_t i = std::min(v.size()-1, (size_t)std::ceil(p/100.0*v.size())-1);
    return v[i];
}

// Cell buffers: all 3 stages
struct CellBuf {
    // Gram inputs/outputs
    cf32 *dM, *dH, *dG;
    // Coef inputs/outputs
    cf32 *dC, *dY, *dO;
    // LLR input (reuse dO) / output
    float *dLLR;
    // Sizes
    size_t szM, szH, szG, szC, szY, szO, szLLR;
};

CellBuf make_cell(int nPRB, int seed) {
    CellBuf cb;
    int nSC = nPRB * N_SC_PRB;
    cb.szM = (size_t)nSC * N_LAYERS * N_ANT * sizeof(cf32);
    cb.szH = (size_t)nSC * N_ANT * N_LAYERS * sizeof(cf32);
    cb.szG = (size_t)nSC * N_LAYERS * N_LAYERS * sizeof(cf32);
    cb.szC = cb.szM;
    cb.szY = (size_t)nSC * N_SYM * N_ANT * sizeof(cf32);
    cb.szO = (size_t)nSC * N_LAYERS * N_SYM * sizeof(cf32);
    cb.szLLR = (size_t)nSC * N_LAYERS * N_SYM * N_MOD_ORDER * sizeof(float);

    CK(cudaMalloc(&cb.dM, cb.szM));
    CK(cudaMalloc(&cb.dH, cb.szH));
    CK(cudaMalloc(&cb.dG, cb.szG));
    CK(cudaMalloc(&cb.dC, cb.szC));
    CK(cudaMalloc(&cb.dY, cb.szY));
    CK(cudaMalloc(&cb.dO, cb.szO));
    CK(cudaMalloc(&cb.dLLR, cb.szLLR));

    cf32 *hM = (cf32*)malloc(cb.szM), *hH = (cf32*)malloc(cb.szH);
    cf32 *hC = (cf32*)malloc(cb.szC), *hY = (cf32*)malloc(cb.szY);
    srand(42 + seed*1000);
    auto rf = []() { return ((float)rand()/RAND_MAX - 0.5f) * 1.4f; };
    for (size_t i = 0; i < cb.szM/sizeof(cf32); i++) hM[i] = {rf(), rf()};
    for (size_t i = 0; i < cb.szH/sizeof(cf32); i++) hH[i] = {rf(), rf()};
    for (size_t i = 0; i < cb.szC/sizeof(cf32); i++) hC[i] = {rf(), rf()};
    for (size_t i = 0; i < cb.szY/sizeof(cf32); i++) hY[i] = {rf(), rf()};
    CK(cudaMemcpy(cb.dM, hM, cb.szM, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cb.dH, hH, cb.szH, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cb.dC, hC, cb.szC, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cb.dY, hY, cb.szY, cudaMemcpyHostToDevice));
    free(hM); free(hH); free(hC); free(hY);
    return cb;
}

void free_cell(CellBuf& cb) {
    cudaFree(cb.dM); cudaFree(cb.dH); cudaFree(cb.dG);
    cudaFree(cb.dC); cudaFree(cb.dY); cudaFree(cb.dO);
    cudaFree(cb.dLLR);
}

// Stage names
const char* stage_name(int s) {
    if (s == 0) return "Gram";
    if (s == 1) return "Coef";
    if (s == 2) return "LLR";
    return "?";
}

// Working set per stage in MB
double stage_ws_mb(int s, int nPRB) {
    int nSC = nPRB * N_SC_PRB;
    if (s == 0) // Gram: M + H + G
        return ((size_t)nSC * (N_LAYERS*N_ANT + N_ANT*N_LAYERS + N_LAYERS*N_LAYERS) * sizeof(cf32)) / 1e6;
    if (s == 1) // Coef: C + Y + O
        return ((size_t)nSC * (N_LAYERS*N_ANT + N_SYM*N_ANT + N_LAYERS*N_SYM) * sizeof(cf32)) / 1e6;
    if (s == 2) // LLR: Y_eq (reuse dO) + LLR output
        return ((size_t)nSC * (N_LAYERS*N_SYM*sizeof(cf32) + N_LAYERS*N_SYM*N_MOD_ORDER*sizeof(float))) / 1e6;
    return 0;
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    double hbm_peak = prop.memoryBusWidth * (prop.memoryClockRate/1e6) * 2 / 8.0;
    int nSMs = prop.multiProcessorCount;
    int nSC = N_SC;

    printf("=== Experiment 4: Pipeline-Aware Stage Overlap ===\n");
    printf("GPU: %s | %d SMs | %.0f GB/s HBM | L2: %.0f MB\n\n",
           prop.name, nSMs, hbm_peak, prop.l2CacheSize/1e6);

    printf("Pipeline stages per cell (100MHz, 132RB):\n");
    printf("  Stage 0: Gram   — L1-bound,  WS=%.2f MB, ~0.035 ms\n", stage_ws_mb(0, 132));
    printf("  Stage 1: Coef   — HBM-bound, WS=%.2f MB, ~0.027 ms\n", stage_ws_mb(1, 132));
    printf("  Stage 2: LLR    — L1-bound,  WS=%.2f MB, ~0.030 ms\n", stage_ws_mb(2, 132));
    printf("\n");

    // Allocate cells
    CellBuf cells[MAX_CELLS];
    for (int c = 0; c < MAX_CELLS; c++) cells[c] = make_cell(132, c);

    cudaStream_t streams[MAX_CELLS];
    cudaEvent_t estart[MAX_CELLS], estop[MAX_CELLS];
    for (int c = 0; c < MAX_CELLS; c++) {
        cudaStreamCreate(&streams[c]);
        cudaEventCreate(&estart[c]); cudaEventCreate(&estop[c]);
    }

    // Kernel launch configs
    dim3 gram_grid(N_SC_PRB, N_PRB), gram_blk(N_LAYERS, N_LAYERS);
    dim3 coef_grid(nSC), coef_blk(32);
    dim3 llr_grid(N_PRB, N_LAYERS), llr_blk(N_SC_PRB, N_SYM);

    // Launch a specific stage for a cell on its stream
    auto launch_stage = [&](int cell, int stage) {
        auto& cb = cells[cell];
        if (stage == 0)
            k_gram<<<gram_grid, gram_blk, 0, streams[cell]>>>(cb.dM, cb.dH, cb.dG, nSC);
        else if (stage == 1)
            k_coef<<<coef_grid, coef_blk, 0, streams[cell]>>>(cb.dC, cb.dY, cb.dO, nSC);
        else
            k_llr<<<llr_grid, llr_blk, 0, streams[cell]>>>(cb.dO, cb.dLLR, nSC, N_SYM);
    };

    // Measure isolated stage latency
    printf("=== Isolated stage latencies (p50) ===\n");
    double iso_lat[3];
    for (int s = 0; s < 3; s++) {
        for (int it = 0; it < WARMUP; it++) {
            launch_stage(0, s);
            CK(cudaDeviceSynchronize());
        }
        std::vector<float> ts;
        for (int it = 0; it < ITERS; it++) {
            cudaEventRecord(estart[0], streams[0]);
            launch_stage(0, s);
            cudaEventRecord(estop[0], streams[0]);
            CK(cudaDeviceSynchronize());
            float ms; cudaEventElapsedTime(&ms, estart[0], estop[0]);
            ts.push_back(ms);
        }
        iso_lat[s] = pct(ts, 50);
        printf("  %-6s isolated: %.4f ms  WS=%.2f MB\n",
               stage_name(s), iso_lat[s], stage_ws_mb(s, 132));
    }
    printf("\n");

    // ── 2-cell overlap: same stage vs cross stage ──────────────────────────────
    printf("=== 2-Cell Overlap: Same Stage vs Cross Stage ===\n");
    printf("Measures per-cell p50 latency when two cells run concurrently.\n\n");

    printf("%-22s %-10s %-10s %-10s %-10s %-10s\n",
           "Combo", "Cell0(ms)", "Cell1(ms)", "Iso0(ms)", "Iso1(ms)", "Degrade0");
    printf("----------------------------------------------------------------\n");

    auto run_pair = [&](int s0, int s1) {
        for (int it = 0; it < WARMUP; it++) {
            launch_stage(0, s0); launch_stage(1, s1);
            CK(cudaDeviceSynchronize());
        }
        std::vector<float> t0, t1;
        for (int it = 0; it < ITERS; it++) {
            cudaEventRecord(estart[0], streams[0]);
            cudaEventRecord(estart[1], streams[1]);
            launch_stage(0, s0);
            launch_stage(1, s1);
            cudaEventRecord(estop[0], streams[0]);
            cudaEventRecord(estop[1], streams[1]);
            CK(cudaDeviceSynchronize());
            float ms0, ms1;
            cudaEventElapsedTime(&ms0, estart[0], estop[0]);
            cudaEventElapsedTime(&ms1, estart[1], estop[1]);
            t0.push_back(ms0); t1.push_back(ms1);
        }
        double p0 = pct(t0, 50), p1 = pct(t1, 50);
        char label[32]; snprintf(label, 32, "%s + %s", stage_name(s0), stage_name(s1));
        printf("%-22s %-10.4f %-10.4f %-10.4f %-10.4f %-10.2fx\n",
               label, p0, p1, iso_lat[s0], iso_lat[s1], p0/iso_lat[s0]);
        return p0;
    };

    // Same-stage baselines
    printf("--- Same stage (baseline contention) ---\n");
    double gram_gram = run_pair(0, 0);
    double coef_coef = run_pair(1, 1);
    double llr_llr   = run_pair(2, 2);
    printf("\n");

    // Cross-stage (pipeline-aware candidates)
    printf("--- Cross stage (pipeline-aware overlap) ---\n");
    double gram_coef = run_pair(0, 1);
    double gram_llr  = run_pair(0, 2);
    double coef_gram = run_pair(1, 0);
    double coef_llr  = run_pair(1, 2);
    double llr_gram  = run_pair(2, 0);
    double llr_coef  = run_pair(2, 1);
    printf("\n");

    // ── 4-cell pipeline rotation ───────────────────────────────────────────────
    printf("=== 4-Cell Pipeline Rotation ===\n");
    printf("All cells at same stage vs staggered (each cell at different stage).\n\n");

    auto run_4cell = [&](int stages[4], const char* label) {
        for (int it = 0; it < WARMUP; it++) {
            for (int c = 0; c < 4; c++) launch_stage(c, stages[c]);
            CK(cudaDeviceSynchronize());
        }
        std::vector<float> tc[4];
        for (int it = 0; it < ITERS; it++) {
            for (int c = 0; c < 4; c++) cudaEventRecord(estart[c], streams[c]);
            for (int c = 0; c < 4; c++) launch_stage(c, stages[c]);
            for (int c = 0; c < 4; c++) cudaEventRecord(estop[c], streams[c]);
            CK(cudaDeviceSynchronize());
            float ms;
            for (int c = 0; c < 4; c++) {
                cudaEventElapsedTime(&ms, estart[c], estop[c]);
                tc[c].push_back(ms);
            }
        }
        printf("%-30s", label);
        double max_p50 = 0;
        for (int c = 0; c < 4; c++) {
            double p50 = pct(tc[c], 50);
            printf("  %s[%d]:%.4f", stage_name(stages[c]), c, p50);
            if (p50 > max_p50) max_p50 = p50;
        }
        printf("  max:%.4f\n", max_p50);
        return max_p50;
    };

    // All same stage
    int all_coef[4] = {1,1,1,1};
    double same4 = run_4cell(all_coef, "4x Coef (all same):");

    // Staggered: each cell at different stage
    int staggered[4] = {0,1,2,1};  // Gram, Coef, LLR, Coef
    double stag4 = run_4cell(staggered, "Staggered (G,C,L,C):");

    // 2+2 split
    int split22[4] = {0,0,1,1};  // 2 Gram + 2 Coef
    double split4 = run_4cell(split22, "2 Gram + 2 Coef:");

    // 1+3 split
    int split13[4] = {0,1,1,1};  // 1 Gram + 3 Coef
    double split13_4 = run_4cell(split13, "1 Gram + 3 Coef:");

    int split31[4] = {1,1,1,2};  // 3 Coef + 1 LLR
    double split31_4 = run_4cell(split31, "3 Coef + 1 LLR:");

    printf("\n");

    // ── Full pipeline simulation: 2 cells with real pipeline dependencies ──────
    printf("=== Full Pipeline: 2 Cells with Dependencies ===\n");
    printf("Cell pipeline: Gram → Coef → LLR (sequential within cell)\n");
    printf("Compare: synchronized (both start same stage) vs staggered (offset by 1 stage)\n\n");

    // Synchronized: both cells run same stage at same time
    {
        for (int it = 0; it < WARMUP; it++) {
            // Stage 0: both Gram
            launch_stage(0, 0); launch_stage(1, 0);
            CK(cudaDeviceSynchronize());
            // Stage 1: both Coef
            launch_stage(0, 1); launch_stage(1, 1);
            CK(cudaDeviceSynchronize());
            // Stage 2: both LLR
            launch_stage(0, 2); launch_stage(1, 2);
            CK(cudaDeviceSynchronize());
        }
        std::vector<float> total_times;
        cudaEvent_t gst, gsp; cudaEventCreate(&gst); cudaEventCreate(&gsp);
        for (int it = 0; it < ITERS; it++) {
            cudaEventRecord(gst);
            launch_stage(0, 0); launch_stage(1, 0);
            cudaStreamSynchronize(streams[0]); cudaStreamSynchronize(streams[1]);
            launch_stage(0, 1); launch_stage(1, 1);
            cudaStreamSynchronize(streams[0]); cudaStreamSynchronize(streams[1]);
            launch_stage(0, 2); launch_stage(1, 2);
            cudaStreamSynchronize(streams[0]); cudaStreamSynchronize(streams[1]);
            cudaEventRecord(gsp);
            CK(cudaDeviceSynchronize());
            float ms; cudaEventElapsedTime(&ms, gst, gsp);
            total_times.push_back(ms);
        }
        double sync_p50 = pct(total_times, 50);
        printf("  Synchronized (both same stage):  %.4f ms total\n", sync_p50);
        cudaEventDestroy(gst); cudaEventDestroy(gsp);
    }

    // Staggered: cell 1 starts at stage 0, cell 2 starts at stage 1
    // This requires cell 2 to have pre-computed data from previous slot
    // We simulate by having cell 2 use its own independent buffers
    {
        for (int it = 0; it < WARMUP; it++) {
            // T=0: cell1 Gram + cell2 Coef (overlap)
            launch_stage(0, 0); launch_stage(1, 1);
            cudaStreamSynchronize(streams[0]); cudaStreamSynchronize(streams[1]);
            // T=1: cell1 Coef + cell2 LLR (overlap)
            launch_stage(0, 1); launch_stage(1, 2);
            cudaStreamSynchronize(streams[0]); cudaStreamSynchronize(streams[1]);
            // T=2: cell1 LLR (alone, cell2 done)
            launch_stage(0, 2);
            cudaStreamSynchronize(streams[0]);
        }
        std::vector<float> total_times;
        cudaEvent_t gst, gsp; cudaEventCreate(&gst); cudaEventCreate(&gsp);
        for (int it = 0; it < ITERS; it++) {
            cudaEventRecord(gst);
            launch_stage(0, 0); launch_stage(1, 1);
            cudaStreamSynchronize(streams[0]); cudaStreamSynchronize(streams[1]);
            launch_stage(0, 1); launch_stage(1, 2);
            cudaStreamSynchronize(streams[0]); cudaStreamSynchronize(streams[1]);
            launch_stage(0, 2);
            cudaStreamSynchronize(streams[0]);
            cudaEventRecord(gsp);
            CK(cudaDeviceSynchronize());
            float ms; cudaEventElapsedTime(&ms, gst, gsp);
            total_times.push_back(ms);
        }
        double stag_p50 = pct(total_times, 50);
        printf("  Staggered (offset by 1 stage):   %.4f ms total\n", stag_p50);
        cudaEventDestroy(gst); cudaEventDestroy(gsp);
    }

    // Fully pipelined: 3 cells, each at different stage
    {
        printf("\n  3-cell fully pipelined (each at different stage):\n");
        for (int it = 0; it < WARMUP; it++) {
            // T=0: c0 Gram + c1 Coef + c2 LLR
            launch_stage(0, 0); launch_stage(1, 1); launch_stage(2, 2);
            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
            cudaStreamSynchronize(streams[2]);
            // T=1: c0 Coef + c1 LLR + c2 Gram (next slot)
            launch_stage(0, 1); launch_stage(1, 2); launch_stage(2, 0);
            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
            cudaStreamSynchronize(streams[2]);
            // T=2: c0 LLR + c1 Gram + c2 Coef
            launch_stage(0, 2); launch_stage(1, 0); launch_stage(2, 1);
            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
            cudaStreamSynchronize(streams[2]);
        }
        std::vector<float> total_times;
        cudaEvent_t gst, gsp; cudaEventCreate(&gst); cudaEventCreate(&gsp);
        for (int it = 0; it < ITERS; it++) {
            cudaEventRecord(gst);
            launch_stage(0, 0); launch_stage(1, 1); launch_stage(2, 2);
            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
            cudaStreamSynchronize(streams[2]);
            launch_stage(0, 1); launch_stage(1, 2); launch_stage(2, 0);
            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
            cudaStreamSynchronize(streams[2]);
            launch_stage(0, 2); launch_stage(1, 0); launch_stage(2, 1);
            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
            cudaStreamSynchronize(streams[2]);
            cudaEventRecord(gsp);
            CK(cudaDeviceSynchronize());
            float ms; cudaEventElapsedTime(&ms, gst, gsp);
            total_times.push_back(ms);
        }
        double pipe3_p50 = pct(total_times, 50);
        printf("    3-cell pipelined: %.4f ms total (3 stages × 3 cells)\n", pipe3_p50);

        // Compare: 3 cells synchronized (all same stage at each step)
        std::vector<float> sync3_times;
        for (int it = 0; it < WARMUP; it++) {
            for (int s = 0; s < 3; s++) {
                launch_stage(0, s); launch_stage(1, s); launch_stage(2, s);
                CK(cudaDeviceSynchronize());
            }
        }
        for (int it = 0; it < ITERS; it++) {
            cudaEventRecord(gst);
            for (int s = 0; s < 3; s++) {
                launch_stage(0, s); launch_stage(1, s); launch_stage(2, s);
                CK(cudaDeviceSynchronize());
            }
            cudaEventRecord(gsp);
            CK(cudaDeviceSynchronize());
            float ms; cudaEventElapsedTime(&ms, gst, gsp);
            sync3_times.push_back(ms);
        }
        double sync3_p50 = pct(sync3_times, 50);
        printf("    3-cell synchronized: %.4f ms total\n", sync3_p50);
        printf("    Speedup of pipelined: %.2fx\n", sync3_p50 / pipe3_p50);
        cudaEventDestroy(gst); cudaEventDestroy(gsp);
    }

    // ── Summary ────────────────────────────────────────────────────────────────
    printf("\n=== Summary: Stage Overlap Contention ===\n");
    printf("Same-stage degradation (2 cells):\n");
    printf("  Gram+Gram:  %.2fx  (WS total=%.1f MB)\n", gram_gram/iso_lat[0], 2*stage_ws_mb(0,132));
    printf("  Coef+Coef:  %.2fx  (WS total=%.1f MB)\n", coef_coef/iso_lat[1], 2*stage_ws_mb(1,132));
    printf("  LLR+LLR:    %.2fx  (WS total=%.1f MB)\n", llr_llr/iso_lat[2], 2*stage_ws_mb(2,132));
    printf("Cross-stage degradation (2 cells):\n");
    printf("  Gram+Coef:  %.2fx  (WS total=%.1f MB)\n", gram_coef/iso_lat[0], stage_ws_mb(0,132)+stage_ws_mb(1,132));
    printf("  Coef+LLR:   %.2fx  (WS total=%.1f MB)\n", coef_llr/iso_lat[1], stage_ws_mb(1,132)+stage_ws_mb(2,132));
    printf("  Gram+LLR:   %.2fx  (WS total=%.1f MB)\n", gram_llr/iso_lat[0], stage_ws_mb(0,132)+stage_ws_mb(2,132));

    // Cleanup
    for (int c = 0; c < MAX_CELLS; c++) {
        free_cell(cells[c]);
        cudaStreamDestroy(streams[c]);
        cudaEventDestroy(estart[c]); cudaEventDestroy(estop[c]);
    }
    return 0;
}
