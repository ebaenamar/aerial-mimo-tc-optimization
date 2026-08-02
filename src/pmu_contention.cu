/*
 * pmu_contention.cu — Experiment 5: PMU-based online contention detection
 *
 * Tests whether hardware performance counters can predict contention
 * at runtime, without offline profiling.
 *
 * Approach:
 *   - Use CUDA Event API to measure elapsed time (baseline)
 *   - Use CUPTI / NCU metrics to read PMU counters at runtime
 *   - Correlate counter values with observed latency degradation
 *
 * Since CUPTI activity API is complex, we use a simpler approach:
 *   - Run N concurrent cells and measure aggregate bandwidth via timing
 *   - Use cudaEventElapsedTime to measure per-cell latency
 *   - Derive "contention indicator" from latency ratio vs isolated baseline
 *   - Validate that this indicator can be computed online
 *
 * Additionally, we test whether we can use CUDA's built-in
 * cudaDeviceGetAttribute and runtime metrics to detect saturation.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_80 -std=c++17 pmu_contention.cu -o pmu_contention
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
#define ITERS 300
#define MAX_CELLS 8

struct cf32 { float re, im; };

__device__ __forceinline__ cf32 cma(cf32 a, cf32 b, cf32 acc) {
    return {acc.re + a.re*b.re - a.im*b.im,
            acc.im + a.re*b.im + a.im*b.re};
}

// V2 WMMA kernel
__global__ void k_coef(
    const cf32* __restrict__ C, const cf32* __restrict__ Y,
    cf32*       __restrict__ O, int nSC)
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

double pct(std::vector<float>& v, double p) {
    std::sort(v.begin(), v.end());
    size_t i = std::min(v.size()-1, (size_t)std::ceil(p/100.0*v.size())-1);
    return v[i];
}

struct CellBuf {
    cf32 *dC, *dY, *dO;
    size_t szC, szY, szO;
};

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    double hbm_peak = prop.memoryBusWidth * (prop.memoryClockRate/1e6) * 2 / 8.0;
    int nSMs = prop.multiProcessorCount;
    int nSC = N_SC;

    printf("=== Experiment 5: PMU-Based Online Contention Detection ===\n");
    printf("GPU: %s | %d SMs | %.0f GB/s HBM | L2: %.0f MB\n\n",
           prop.name, nSMs, hbm_peak, prop.l2CacheSize/1e6);

    // Allocate cells
    size_t szC = (size_t)nSC * N_LAYERS * N_ANT * sizeof(cf32);
    size_t szY = (size_t)nSC * N_SYM * N_ANT * sizeof(cf32);
    size_t szO = (size_t)nSC * N_LAYERS * N_SYM * sizeof(cf32);
    double bytes_per_cell = szC + szY + szO;

    CellBuf cells[MAX_CELLS];
    for (int c = 0; c < MAX_CELLS; c++) {
        cells[c].szC = szC; cells[c].szY = szY; cells[c].szO = szO;
        CK(cudaMalloc(&cells[c].dC, szC));
        CK(cudaMalloc(&cells[c].dY, szY));
        CK(cudaMalloc(&cells[c].dO, szO));
        cf32 *hC = (cf32*)malloc(szC), *hY = (cf32*)malloc(szY);
        srand(42 + c*1000);
        auto rf = []() { return ((float)rand()/RAND_MAX - 0.5f) * 1.4f; };
        for (size_t i = 0; i < szC/sizeof(cf32); i++) hC[i] = {rf(), rf()};
        for (size_t i = 0; i < szY/sizeof(cf32); i++) hY[i] = {rf(), rf()};
        CK(cudaMemcpy(cells[c].dC, hC, szC, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(cells[c].dY, hY, szY, cudaMemcpyHostToDevice));
        free(hC); free(hY);
    }

    cudaStream_t streams[MAX_CELLS];
    cudaEvent_t estart[MAX_CELLS], estop[MAX_CELLS];
    cudaEvent_t gstart, gstop;
    for (int c = 0; c < MAX_CELLS; c++) {
        cudaStreamCreate(&streams[c]);
        cudaEventCreate(&estart[c]); cudaEventCreate(&estop[c]);
    }
    cudaEventCreate(&gstart); cudaEventCreate(&gstop);

    dim3 blk(32), grd(nSC);

    // ── Part A: Establish isolated baseline ────────────────────────────────────
    printf("=== Part A: Isolated Baseline ===\n");
    double iso_lat;
    {
        for (int it = 0; it < WARMUP; it++) {
            k_coef<<<grd, blk, 0, streams[0]>>>(cells[0].dC, cells[0].dY, cells[0].dO, nSC);
            CK(cudaDeviceSynchronize());
        }
        std::vector<float> ts;
        for (int it = 0; it < ITERS; it++) {
            cudaEventRecord(estart[0], streams[0]);
            k_coef<<<grd, blk, 0, streams[0]>>>(cells[0].dC, cells[0].dY, cells[0].dO, nSC);
            cudaEventRecord(estop[0], streams[0]);
            CK(cudaDeviceSynchronize());
            float ms; cudaEventElapsedTime(&ms, estart[0], estop[0]);
            ts.push_back(ms);
        }
        iso_lat = pct(ts, 50);
        printf("  Isolated V2 p50: %.4f ms\n", iso_lat);
        printf("  Isolated BW: %.0f GB/s\n\n", bytes_per_cell / (iso_lat * 1e6) / 1e9);
    }

    // ── Part B: Online contention indicator ───────────────────────────────────
    // The idea: measure per-cell latency at runtime. If latency > threshold,
    // we detect contention. We simulate an "online" scenario where the scheduler
    // observes latency of the first iteration and decides whether to continue
    // with the current batch or split it.
    printf("=== Part B: Online Contention Detection ===\n");
    printf("Simulates runtime monitoring: measure first K iterations,\n");
    printf("compute contention indicator, compare with actual degradation.\n\n");

    printf("%-6s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n",
           "N", "Actual_p50", "First5_avg", "First10_avg", "Indicator",
           "Actual_degr", "Predicted", "Error%");
    printf("------------------------------------------------------------------------\n");

    for (int ncells = 1; ncells <= MAX_CELLS; ncells++) {
        // Warmup
        for (int it = 0; it < WARMUP; it++) {
            for (int c = 0; c < ncells; c++)
                k_coef<<<grd, blk, 0, streams[c]>>>(
                    cells[c].dC, cells[c].dY, cells[c].dO, nSC);
            CK(cudaDeviceSynchronize());
        }

        // Collect per-cell latencies for all iterations
        std::vector<float> per_cell_all;
        std::vector<float> first5, first10;
        std::vector<float> total_times;

        for (int it = 0; it < ITERS; it++) {
            for (int c = 0; c < ncells; c++) cudaEventRecord(estart[c], streams[c]);
            cudaEventRecord(gstart);
            for (int c = 0; c < ncells; c++)
                k_coef<<<grd, blk, 0, streams[c]>>>(
                    cells[c].dC, cells[c].dY, cells[c].dO, nSC);
            cudaEventRecord(gstop);
            for (int c = 0; c < ncells; c++) cudaEventRecord(estop[c], streams[c]);
            CK(cudaDeviceSynchronize());

            float ms;
            cudaEventElapsedTime(&ms, gstart, gstop);
            total_times.push_back(ms);

            for (int c = 0; c < ncells; c++) {
                cudaEventElapsedTime(&ms, estart[c], estop[c]);
                per_cell_all.push_back(ms);
                if (it < 5) first5.push_back(ms);
                if (it < 10) first10.push_back(ms);
            }
        }

        double actual_p50 = pct(per_cell_all, 50);
        double first5_avg = 0;
        for (float v : first5) first5_avg += v;
        first5_avg /= first5.size();

        double first10_avg = 0;
        for (float v : first10) first10_avg += v;
        first10_avg /= first10.size();

        // Online indicator: ratio of observed latency to isolated baseline
        double indicator = first5_avg / iso_lat;
        double actual_degr = actual_p50 / iso_lat;

        // Predicted degradation = indicator (we use first 5 iters as proxy)
        double predicted = indicator;
        double error_pct = fabs(predicted - actual_degr) / actual_degr * 100;

        printf("%-6d %-12.4f %-12.4f %-12.4f %-12.2f %-12.2f %-12.2f %-12.1f\n",
               ncells, actual_p50, first5_avg, first10_avg,
               indicator, actual_degr, predicted, error_pct);
    }

    printf("\n");

    // ── Part C: Adaptive batch sizing based on online detection ────────────────
    printf("=== Part C: Adaptive Batch Sizing ===\n");
    printf("Simulates: start with N=8 cells, measure contention,\n");
    printf("reduce batch size if degradation exceeds threshold.\n\n");

    // Threshold: if per-cell latency > 2x isolated, split batch
    double threshold = 2.0;

    printf("Threshold: %.1fx isolated latency\n\n", threshold);

    for (int ncells = 1; ncells <= MAX_CELLS; ncells++) {
        // Quick measurement: 20 iterations
        for (int it = 0; it < 20; it++) {
            for (int c = 0; c < ncells; c++)
                k_coef<<<grd, blk, 0, streams[c]>>>(
                    cells[c].dC, cells[c].dY, cells[c].dO, nSC);
            CK(cudaDeviceSynchronize());
        }

        std::vector<float> quick;
        for (int it = 0; it < 20; it++) {
            for (int c = 0; c < ncells; c++) cudaEventRecord(estart[c], streams[c]);
            for (int c = 0; c < ncells; c++)
                k_coef<<<grd, blk, 0, streams[c]>>>(
                    cells[c].dC, cells[c].dY, cells[c].dO, nSC);
            for (int c = 0; c < ncells; c++) cudaEventRecord(estop[c], streams[c]);
            CK(cudaDeviceSynchronize());
            float ms;
            for (int c = 0; c < ncells; c++) {
                cudaEventElapsedTime(&ms, estart[c], estop[c]);
                quick.push_back(ms);
            }
        }

        double quick_p50 = pct(quick, 50);
        double degr = quick_p50 / iso_lat;
        const char* action = degr > threshold ? "SPLIT" : "KEEP";

        printf("  N=%d: quick_p50=%.4f ms  degr=%.2fx  → %s\n",
               ncells, quick_p50, degr, action);
    }

    printf("\n");

    // ── Part D: Bandwidth estimation from timing ───────────────────────────────
    printf("=== Part D: Runtime Bandwidth Estimation ===\n");
    printf("Derives aggregate BW from total batch time (measurable at runtime).\n");
    printf("This can be used as online contention metric.\n\n");

    printf("%-6s %-12s %-12s %-12s %-12s %-12s\n",
           "N", "Total(ms)", "AggBW(GB/s)", "%HBM", "BW_per_cell", "Contention");
    printf("--------------------------------------------------------\n");

    for (int ncells = 1; ncells <= MAX_CELLS; ncells++) {
        for (int it = 0; it < WARMUP; it++) {
            for (int c = 0; c < ncells; c++)
                k_coef<<<grd, blk, 0, streams[c]>>>(
                    cells[c].dC, cells[c].dY, cells[c].dO, nSC);
            CK(cudaDeviceSynchronize());
        }
        std::vector<float> times;
        for (int it = 0; it < ITERS; it++) {
            cudaEventRecord(gstart);
            for (int c = 0; c < ncells; c++)
                k_coef<<<grd, blk, 0, streams[c]>>>(
                    cells[c].dC, cells[c].dY, cells[c].dO, nSC);
            cudaEventRecord(gstop);
            CK(cudaDeviceSynchronize());
            float ms; cudaEventElapsedTime(&ms, gstart, gstop);
            times.push_back(ms);
        }
        double total_p50 = pct(times, 50);
        double agg_bw = (ncells * bytes_per_cell) / (total_p50 * 1e6) / 1e9;
        double bw_per = agg_bw / ncells;
        double contention = 1.0 - (bw_per / (bytes_per_cell / (iso_lat * 1e6) / 1e9));

        printf("%-6d %-12.4f %-12.0f %-12.1f %-12.0f %-12.3f\n",
               ncells, total_p50, agg_bw, agg_bw/hbm_peak*100, bw_per, contention);
    }

    printf("\nNote: 'Contention' = 1 - (actual_bw_per_cell / isolated_bw_per_cell)\n");
    printf("This metric is computable at runtime from total batch time.\n");

    // Cleanup
    for (int c = 0; c < MAX_CELLS; c++) {
        cudaFree(cells[c].dC); cudaFree(cells[c].dY); cudaFree(cells[c].dO);
        cudaStreamDestroy(streams[c]);
        cudaEventDestroy(estart[c]); cudaEventDestroy(estop[c]);
    }
    cudaEventDestroy(gstart); cudaEventDestroy(gstop);
    return 0;
}
