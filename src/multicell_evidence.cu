/*
 * multicell_evidence.cu — Experiment 3: Evidence for scheduler design
 *
 * Four sub-experiments:
 *   A) Temporal serialization vs concurrent: is it better to run cells
 *      back-to-back on one stream or concurrently on N streams?
 *   B) Heterogeneous cell sizes: 100MHz + 20MHz + 5MHz concurrent
 *   C) Effective bandwidth vs N: derive the contention function delta
 *   D) MPS-like SM partitioning: restrict kernels to subset of SMs
 *      via launch bounds / grid size manipulation
 *
 * Compile:
 *   nvcc -O3 -arch=sm_80 -std=c++17 multicell_evidence.cu -o multicell_evidence
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
#define WARMUP 50
#define ITERS 300
#define MAX_CELLS 8

struct cf32 { float re, im; };

__device__ __forceinline__ cf32 cma(cf32 a, cf32 b, cf32 acc) {
    return {acc.re + a.re*b.re - a.im*b.im,
            acc.im + a.re*b.im + a.im*b.re};
}

// V2 WMMA — parameterized by nSC for heterogeneous sizes
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

double pct(std::vector<float>& v, double p) {
    std::sort(v.begin(), v.end());
    size_t i = std::min(v.size()-1, (size_t)std::ceil(p/100.0*v.size())-1);
    return v[i];
}

// Cell config: different bandwidths = different nSC
struct CellConfig {
    int nPRB;
    int nSC;
    size_t szC, szY, szO;
    cf32 *dC, *dY, *dO;
    double bytes;  // total bytes moved (C+Y+O)
    double flops;  // total FLOPs
};

CellConfig make_cell(int nPRB, int seed) {
    CellConfig cc;
    cc.nPRB = nPRB;
    cc.nSC = nPRB * N_SC_PRB;
    cc.szC = (size_t)cc.nSC * N_LAYERS * N_ANT * sizeof(cf32);
    cc.szY = (size_t)cc.nSC * N_SYM * N_ANT * sizeof(cf32);
    cc.szO = (size_t)cc.nSC * N_LAYERS * N_SYM * sizeof(cf32);
    cc.bytes = (double)(cc.szC + cc.szY + cc.szO);
    cc.flops = 8.0 * cc.nSC * N_LAYERS * N_SYM * N_ANT;
    CK(cudaMalloc(&cc.dC, cc.szC));
    CK(cudaMalloc(&cc.dY, cc.szY));
    CK(cudaMalloc(&cc.dO, cc.szO));
    cf32 *hC = (cf32*)malloc(cc.szC);
    cf32 *hY = (cf32*)malloc(cc.szY);
    srand(42 + seed*1000);
    auto rf = []() { return ((float)rand()/RAND_MAX - 0.5f) * 1.4f; };
    for (size_t i = 0; i < cc.szC/sizeof(cf32); i++) hC[i] = {rf(), rf()};
    for (size_t i = 0; i < cc.szY/sizeof(cf32); i++) hY[i] = {rf(), rf()};
    CK(cudaMemcpy(cc.dC, hC, cc.szC, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cc.dY, hY, cc.szY, cudaMemcpyHostToDevice));
    free(hC); free(hY);
    return cc;
}

void free_cell(CellConfig& cc) {
    cudaFree(cc.dC); cudaFree(cc.dY); cudaFree(cc.dO);
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    double hbm_peak = prop.memoryBusWidth * (prop.memoryClockRate/1e6) * 2 / 8.0;
    int nSMs = prop.multiProcessorCount;

    printf("=== Experiment 3: Evidence for Scheduler Design ===\n");
    printf("GPU: %s | %d SMs | %.0f GB/s HBM | L2: %.0f MB\n\n",
           prop.name, nSMs, hbm_peak, prop.l2CacheSize/1e6);

    // ============================================================
    // Sub-experiment A: Temporal serialization vs concurrent
    // ============================================================
    printf("=== A) Temporal Serialization vs Concurrent ===\n");
    printf("Question: Is it faster to run N cells back-to-back (serial)\n");
    printf("or concurrently on N streams?\n\n");

    {
        int nPRB = 132;  // 100 MHz
        int nSC = nPRB * N_SC_PRB;
        CellConfig cells[MAX_CELLS];
        for (int c = 0; c < MAX_CELLS; c++)
            cells[c] = make_cell(nPRB, c);

        cudaStream_t streams[MAX_CELLS];
        for (int c = 0; c < MAX_CELLS; c++) cudaStreamCreate(&streams[c]);
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        printf("%-8s %-16s %-16s %-16s %-16s %-16s\n",
               "Ncells", "Concurrent(ms)", "Serial(ms)", "Conc_per(ms)",
               "Serial_per(ms)", "Winner");
        printf("--------------------------------------------------------------------------------\n");

        for (int ncells = 1; ncells <= MAX_CELLS; ncells++) {
            // Warmup
            for (int it = 0; it < WARMUP; it++) {
                for (int c = 0; c < ncells; c++)
                    v2_wmma<<<nSC, 32, 0, streams[c]>>>(
                        cells[c].dC, cells[c].dY, cells[c].dO, nSC);
                CK(cudaDeviceSynchronize());
            }

            // Concurrent: all on separate streams
            std::vector<float> conc_times;
            for (int it = 0; it < ITERS; it++) {
                cudaEventRecord(start);
                for (int c = 0; c < ncells; c++)
                    v2_wmma<<<nSC, 32, 0, streams[c]>>>(
                        cells[c].dC, cells[c].dY, cells[c].dO, nSC);
                cudaEventRecord(stop);
                CK(cudaDeviceSynchronize());
                float ms; cudaEventElapsedTime(&ms, start, stop);
                conc_times.push_back(ms);
            }
            double conc_p50 = pct(conc_times, 50);

            // Serial: all on stream 0, back-to-back
            std::vector<float> ser_times;
            for (int it = 0; it < WARMUP; it++) {
                for (int c = 0; c < ncells; c++)
                    v2_wmma<<<nSC, 32, 0, streams[0]>>>(
                        cells[c].dC, cells[c].dY, cells[c].dO, nSC);
                CK(cudaDeviceSynchronize());
            }
            for (int it = 0; it < ITERS; it++) {
                cudaEventRecord(start);
                for (int c = 0; c < ncells; c++)
                    v2_wmma<<<nSC, 32, 0, streams[0]>>>(
                        cells[c].dC, cells[c].dY, cells[c].dO, nSC);
                cudaEventRecord(stop);
                CK(cudaDeviceSynchronize());
                float ms; cudaEventElapsedTime(&ms, start, stop);
                ser_times.push_back(ms);
            }
            double ser_p50 = pct(ser_times, 50);

            printf("%-8d %-16.4f %-16.4f %-16.4f %-16.4f %-16s\n",
                   ncells, conc_p50, ser_p50,
                   conc_p50/ncells, ser_p50/ncells,
                   conc_p50 < ser_p50 ? "CONCURRENT" : "SERIAL");
        }

        printf("\n");
        for (int c = 0; c < MAX_CELLS; c++) { free_cell(cells[c]); cudaStreamDestroy(streams[c]); }
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    // ============================================================
    // Sub-experiment B: Heterogeneous cell sizes
    // ============================================================
    printf("=== B) Heterogeneous Cell Sizes ===\n");
    printf("Question: Do small cells (5MHz) contend less with large cells\n");
    printf("(100MHz) than two large cells with each other?\n\n");

    {
        // Cell sizes: 5MHz(11RB), 20MHz(52RB), 50MHz(106RB), 100MHz(132RB), 200MHz(264RB)
        struct { int prb; const char* name; } sizes[] = {
            {11, "5MHz"}, {52, "20MHz"}, {106, "50MHz"}, {132, "100MHz"}, {264, "200MHz"}
        };
        int nsizes = 5;

        // First: isolated baselines
        printf("Isolated baselines:\n");
        double iso_lat[5];
        for (int i = 0; i < nsizes; i++) {
            CellConfig cc = make_cell(sizes[i].prb, i);
            cudaStream_t s; cudaStreamCreate(&s);
            cudaEvent_t st, sp; cudaEventCreate(&st); cudaEventCreate(&sp);
            for (int it = 0; it < WARMUP; it++) {
                v2_wmma<<<cc.nSC, 32, 0, s>>>(cc.dC, cc.dY, cc.dO, cc.nSC);
                CK(cudaDeviceSynchronize());
            }
            std::vector<float> ts;
            for (int it = 0; it < ITERS; it++) {
                cudaEventRecord(st, s);
                v2_wmma<<<cc.nSC, 32, 0, s>>>(cc.dC, cc.dY, cc.dO, cc.nSC);
                cudaEventRecord(sp, s);
                CK(cudaDeviceSynchronize());
                float ms; cudaEventElapsedTime(&ms, st, sp);
                ts.push_back(ms);
            }
            iso_lat[i] = pct(ts, 50);
            printf("  %-8s (%3dRB, %4dSC): %.4f ms  WS=%.2fMB  BW=%.0f GB/s\n",
                   sizes[i].name, sizes[i].prb, cc.nSC, iso_lat[i],
                   cc.bytes/1e6, cc.bytes/(iso_lat[i]*1e6)/1e9);
            free_cell(cc); cudaStreamDestroy(s); cudaEventDestroy(st); cudaEventDestroy(sp);
        }
        printf("\n");

        // Pairs: large+large vs large+small
        printf("Concurrent pairs (2 cells on 2 streams):\n");
        printf("%-20s %-12s %-12s %-12s %-12s %-12s\n",
               "Pair", "Cell0(ms)", "Cell1(ms)", "Iso0(ms)", "Iso1(ms)", "Degrad0");
        printf("------------------------------------------------------------------------\n");

        auto run_pair = [&](int idx0, int idx1) {
            CellConfig c0 = make_cell(sizes[idx0].prb, 0);
            CellConfig c1 = make_cell(sizes[idx1].prb, 1);
            cudaStream_t s0, s1; cudaStreamCreate(&s0); cudaStreamCreate(&s1);
            cudaEvent_t e0s, e0e, e1s, e1e;
            cudaEventCreate(&e0s); cudaEventCreate(&e0e);
            cudaEventCreate(&e1s); cudaEventCreate(&e1e);

            for (int it = 0; it < WARMUP; it++) {
                v2_wmma<<<c0.nSC, 32, 0, s0>>>(c0.dC, c0.dY, c0.dO, c0.nSC);
                v2_wmma<<<c1.nSC, 32, 0, s1>>>(c1.dC, c1.dY, c1.dO, c1.nSC);
                CK(cudaDeviceSynchronize());
            }
            std::vector<float> t0, t1;
            for (int it = 0; it < ITERS; it++) {
                cudaEventRecord(e0s, s0); cudaEventRecord(e1s, s1);
                v2_wmma<<<c0.nSC, 32, 0, s0>>>(c0.dC, c0.dY, c0.dO, c0.nSC);
                v2_wmma<<<c1.nSC, 32, 0, s1>>>(c1.dC, c1.dY, c1.dO, c1.nSC);
                cudaEventRecord(e0e, s0); cudaEventRecord(e1e, s1);
                CK(cudaDeviceSynchronize());
                float ms0, ms1;
                cudaEventElapsedTime(&ms0, e0s, e0e);
                cudaEventElapsedTime(&ms1, e1s, e1e);
                t0.push_back(ms0); t1.push_back(ms1);
            }
            double p0 = pct(t0, 50), p1 = pct(t1, 50);
            char label[32]; snprintf(label, 32, "%s + %s", sizes[idx0].name, sizes[idx1].name);
            printf("%-20s %-12.4f %-12.4f %-12.4f %-12.4f %-12.2fx\n",
                   label, p0, p1, iso_lat[idx0], iso_lat[idx1], p0/iso_lat[idx0]);

            free_cell(c0); free_cell(c1);
            cudaStreamDestroy(s0); cudaStreamDestroy(s1);
            cudaEventDestroy(e0s); cudaEventDestroy(e0e);
            cudaEventDestroy(e1s); cudaEventDestroy(e1e);
        };

        run_pair(4, 4);  // 200+200
        run_pair(3, 3);  // 100+100
        run_pair(3, 4);  // 100+200
        run_pair(3, 1);  // 100+20
        run_pair(3, 0);  // 100+5
        run_pair(1, 1);  // 20+20
        run_pair(1, 0);  // 20+5
        run_pair(0, 0);  // 5+5

        printf("\n");
    }

    // ============================================================
    // Sub-experiment C: Effective bandwidth vs N (contention function)
    // ============================================================
    printf("=== C) Effective Bandwidth vs N (Contention Function) ===\n");
    printf("Question: What is the actual aggregate HBM bandwidth achieved\n");
    printf("as N increases? This gives us the contention function delta.\n\n");

    {
        int nPRB = 132;
        int nSC = nPRB * N_SC_PRB;
        CellConfig cells[MAX_CELLS];
        for (int c = 0; c < MAX_CELLS; c++) cells[c] = make_cell(nPRB, c);
        cudaStream_t streams[MAX_CELLS];
        cudaEvent_t start, stop;
        for (int c = 0; c < MAX_CELLS; c++) cudaStreamCreate(&streams[c]);
        cudaEventCreate(&start); cudaEventCreate(&stop);

        double bytes_per_cell = cells[0].bytes;

        printf("%-6s %-12s %-12s %-12s %-12s %-12s %-12s\n",
               "N", "Total(ms)", "PerCell(ms)", "AggBW(GB/s)", "%HBM", "IdealBW", "Delta");
        printf("------------------------------------------------------------------------\n");

        for (int ncells = 1; ncells <= MAX_CELLS; ncells++) {
            for (int it = 0; it < WARMUP; it++) {
                for (int c = 0; c < ncells; c++)
                    v2_wmma<<<nSC, 32, 0, streams[c]>>>(
                        cells[c].dC, cells[c].dY, cells[c].dO, nSC);
                CK(cudaDeviceSynchronize());
            }
            std::vector<float> times;
            for (int it = 0; it < ITERS; it++) {
                cudaEventRecord(start);
                for (int c = 0; c < ncells; c++)
                    v2_wmma<<<nSC, 32, 0, streams[c]>>>(
                        cells[c].dC, cells[c].dY, cells[c].dO, nSC);
                cudaEventRecord(stop);
                CK(cudaDeviceSynchronize());
                float ms; cudaEventElapsedTime(&ms, start, stop);
                times.push_back(ms);
            }
            double total_p50 = pct(times, 50);
            double per_cell = total_p50 / ncells;  // approximate
            double agg_bw = (ncells * bytes_per_cell) / (total_p50 * 1e6) / 1e9;  // GB/s
            double ideal_bw = ncells * (bytes_per_cell / (0.0266 * 1e6)) / 1e9;  // if no contention
            double delta = agg_bw / ideal_bw;  // <1 means contention

            printf("%-6d %-12.4f %-12.4f %-12.0f %-12.1f %-12.0f %-12.3f\n",
                   ncells, total_p50, per_cell, agg_bw,
                   agg_bw/hbm_peak*100, ideal_bw, delta);
        }

        printf("\n");
        for (int c = 0; c < MAX_CELLS; c++) { free_cell(cells[c]); cudaStreamDestroy(streams[c]); }
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    // ============================================================
    // Sub-experiment D: SM partitioning via grid size restriction
    // ============================================================
    printf("=== D) SM Partitioning via Grid Size Restriction ===\n");
    printf("Question: If we restrict each cell to a subset of SMs (by limiting\n");
    printf("grid blocks), does it reduce contention?\n\n");

    {
        int nPRB = 132;
        int nSC = nPRB * N_SC_PRB;
        // nSC=1584 blocks. With 108 SMs, each SM gets ~14.6 blocks.
        // For N cells, give each cell nSC/N blocks worth of SMs.
        // But we can't actually restrict SMs without MPS.
        // Instead, we simulate by reducing the grid size: each cell
        // processes nSC/N subcarriers, and we measure per-cell latency.

        CellConfig cells[MAX_CELLS];
        for (int c = 0; c < MAX_CELLS; c++) cells[c] = make_cell(nPRB, c);
        cudaStream_t streams[MAX_CELLS];
        cudaEvent_t estart[MAX_CELLS], estop[MAX_CELLS];
        for (int c = 0; c < MAX_CELLS; c++) {
            cudaStreamCreate(&streams[c]);
            cudaEventCreate(&estart[c]); cudaEventCreate(&estop[c]);
        }

        printf("Approach: Each cell launches nSC blocks (full grid).\n");
        printf("With MPS, each cell would be restricted to nSMs/N SMs.\n");
        printf("We compare: full-grid concurrent vs full-grid serial.\n");
        printf("If serial per-cell < concurrent per-cell, SM partitioning\n");
        printf("(which effectively serializes on SMs) would help.\n\n");

        printf("%-6s %-14s %-14s %-14s %-14s\n",
               "N", "Conc_per(ms)", "Serial_per(ms)", "Conc_total(ms)", "Serial_total(ms)");
        printf("--------------------------------------------------------\n");

        for (int ncells = 1; ncells <= MAX_CELLS; ncells++) {
            // Concurrent with per-cell events
            for (int it = 0; it < WARMUP; it++) {
                for (int c = 0; c < ncells; c++)
                    v2_wmma<<<nSC, 32, 0, streams[c]>>>(
                        cells[c].dC, cells[c].dY, cells[c].dO, nSC);
                CK(cudaDeviceSynchronize());
            }
            std::vector<float> per_cell_times;
            std::vector<float> conc_total;
            for (int it = 0; it < ITERS; it++) {
                for (int c = 0; c < ncells; c++) cudaEventRecord(estart[c], streams[c]);
                cudaEvent_t gst, gsp; cudaEventCreate(&gst); cudaEventCreate(&gsp);
                cudaEventRecord(gst);
                for (int c = 0; c < ncells; c++)
                    v2_wmma<<<nSC, 32, 0, streams[c]>>>(
                        cells[c].dC, cells[c].dY, cells[c].dO, nSC);
                cudaEventRecord(gsp);
                for (int c = 0; c < ncells; c++) cudaEventRecord(estop[c], streams[c]);
                CK(cudaDeviceSynchronize());
                float ms;
                for (int c = 0; c < ncells; c++) {
                    cudaEventElapsedTime(&ms, estart[c], estop[c]);
                    per_cell_times.push_back(ms);
                }
                cudaEventElapsedTime(&ms, gst, gsp);
                conc_total.push_back(ms);
                cudaEventDestroy(gst); cudaEventDestroy(gsp);
            }
            double conc_per = pct(per_cell_times, 50);
            double conc_tot = pct(conc_total, 50);

            // Serial: back-to-back on stream 0
            std::vector<float> ser_per;
            std::vector<float> ser_total;
            for (int it = 0; it < WARMUP; it++) {
                for (int c = 0; c < ncells; c++)
                    v2_wmma<<<nSC, 32, 0, streams[0]>>>(
                        cells[c].dC, cells[c].dY, cells[c].dO, nSC);
                CK(cudaDeviceSynchronize());
            }
            for (int it = 0; it < ITERS; it++) {
                cudaEvent_t st, sp; cudaEventCreate(&st); cudaEventCreate(&sp);
                cudaEventRecord(st, streams[0]);
                for (int c = 0; c < ncells; c++) {
                    cudaEvent_t cst, csp;
                    cudaEventCreate(&cst); cudaEventCreate(&csp);
                    cudaEventRecord(cst, streams[0]);
                    v2_wmma<<<nSC, 32, 0, streams[0]>>>(
                        cells[c].dC, cells[c].dY, cells[c].dO, nSC);
                    cudaEventRecord(csp, streams[0]);
                    cudaEventSynchronize(csp);
                    float ms; cudaEventElapsedTime(&ms, cst, csp);
                    ser_per.push_back(ms);
                    cudaEventDestroy(cst); cudaEventDestroy(csp);
                }
                cudaEventRecord(sp, streams[0]);
                CK(cudaDeviceSynchronize());
                float ms; cudaEventElapsedTime(&ms, st, sp);
                ser_total.push_back(ms);
                cudaEventDestroy(st); cudaEventDestroy(sp);
            }
            double ser_per_p50 = pct(ser_per, 50);
            double ser_tot_p50 = pct(ser_total, 50);

            printf("%-6d %-14.4f %-14.4f %-14.4f %-14.4f\n",
                   ncells, conc_per, ser_per_p50, conc_tot, ser_tot_p50);
        }

        printf("\n");
        for (int c = 0; c < MAX_CELLS; c++) {
            free_cell(cells[c]); cudaStreamDestroy(streams[c]);
            cudaEventDestroy(estart[c]); cudaEventDestroy(estop[c]);
        }
    }

    // ============================================================
    // Sub-experiment E: Deadline analysis
    // ============================================================
    printf("=== E) Deadline Analysis ===\n");
    printf("5G NR uplink slot deadline at 120kHz SCS = 0.5 ms (half-slot = 0.25 ms)\n");
    printf("How many cells can we serve within deadline?\n\n");

    {
        int nPRB = 132;
        int nSC = nPRB * N_SC_PRB;
        CellConfig cells[MAX_CELLS];
        for (int c = 0; c < MAX_CELLS; c++) cells[c] = make_cell(nPRB, c);
        cudaStream_t streams[MAX_CELLS];
        for (int c = 0; c < MAX_CELLS; c++) cudaStreamCreate(&streams[c]);
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        printf("%-6s %-12s %-12s %-12s %-12s\n",
               "N", "Conc(ms)", "Serial(ms)", "MeetsDeadline", "MaxCells(deadline)");
        printf("--------------------------------------------------\n");

        for (int ncells = 1; ncells <= MAX_CELLS; ncells++) {
            // Concurrent
            for (int it = 0; it < WARMUP; it++) {
                for (int c = 0; c < ncells; c++)
                    v2_wmma<<<nSC, 32, 0, streams[c]>>>(
                        cells[c].dC, cells[c].dY, cells[c].dO, nSC);
                CK(cudaDeviceSynchronize());
            }
            std::vector<float> conc;
            for (int it = 0; it < ITERS; it++) {
                cudaEventRecord(start);
                for (int c = 0; c < ncells; c++)
                    v2_wmma<<<nSC, 32, 0, streams[c]>>>(
                        cells[c].dC, cells[c].dY, cells[c].dO, nSC);
                cudaEventRecord(stop);
                CK(cudaDeviceSynchronize());
                float ms; cudaEventElapsedTime(&ms, start, stop);
                conc.push_back(ms);
            }
            double conc_p50 = pct(conc, 50);

            // Serial
            std::vector<float> ser;
            for (int it = 0; it < WARMUP; it++) {
                for (int c = 0; c < ncells; c++)
                    v2_wmma<<<nSC, 32, 0, streams[0]>>>(
                        cells[c].dC, cells[c].dY, cells[c].dO, nSC);
                CK(cudaDeviceSynchronize());
            }
            for (int it = 0; it < ITERS; it++) {
                cudaEventRecord(start);
                for (int c = 0; c < ncells; c++)
                    v2_wmma<<<nSC, 32, 0, streams[0]>>>(
                        cells[c].dC, cells[c].dY, cells[c].dO, nSC);
                cudaEventRecord(stop);
                CK(cudaDeviceSynchronize());
                float ms; cudaEventElapsedTime(&ms, start, stop);
                ser.push_back(ms);
            }
            double ser_p50 = pct(ser, 50);

            bool conc_ok = conc_p50 < 0.5;
            bool ser_ok = ser_p50 < 0.5;
            const char* verdict = conc_ok ? "CONC" : (ser_ok ? "SERIAL" : "FAIL");

            printf("%-6d %-12.4f %-12.4f %-12s %-12s\n",
                   ncells, conc_p50, ser_p50, verdict,
                   conc_ok ? "yes" : (ser_ok ? "serial-only" : "no"));
        }

        printf("\nNote: This is only the coef_apply kernel. Full pipeline\n");
        printf("(Gram + coef + LLR) would be ~1.2x this latency.\n");

        for (int c = 0; c < MAX_CELLS; c++) { free_cell(cells[c]); cudaStreamDestroy(streams[c]); }
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    return 0;
}
