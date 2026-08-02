/*
 * adaptive_smpartition.cu — Experiment 6: Adaptive SM partitioning
 *
 * Tests whether dynamic SM partitioning improves multi-cell performance.
 * Since CUDA MPS requires daemon setup, we simulate SM partitioning by:
 *
 *   1) Grid size restriction: limit blocks per kernel to target specific SMs
 *   2) Stream priority: assign different priorities to different cells
 *   3) Compare: unrestricted concurrent vs restricted concurrent vs serial
 *
 * The key question: can we achieve serial-like per-cell latency
 * while maintaining concurrent throughput?
 *
 * Compile:
 *   nvcc -O3 -arch=sm_80 -std=c++17 adaptive_smpartition.cu -o adaptive_smpartition
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

// V2 WMMA — processes nSC subcarriers starting at offset sc_offset
__global__ void k_coef_offset(
    const cf32* __restrict__ C, const cf32* __restrict__ Y,
    cf32*       __restrict__ O, int nSC, int sc_offset)
{
    constexpr int AP = ((N_ANT + WK - 1) / WK) * WK;
    int sc = blockIdx.x + sc_offset, lane = threadIdx.x;
    if (sc >= sc_offset + nSC) return;
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

    printf("=== Experiment 6: Adaptive SM Partitioning ===\n");
    printf("GPU: %s | %d SMs | %.0f GB/s HBM | L2: %.0f MB\n\n",
           prop.name, nSMs, hbm_peak, prop.l2CacheSize/1e6);

    // Allocate cells — each cell gets full nSC buffers
    size_t szC = (size_t)nSC * N_LAYERS * N_ANT * sizeof(cf32);
    size_t szY = (size_t)nSC * N_SYM * N_ANT * sizeof(cf32);
    size_t szO = (size_t)nSC * N_LAYERS * N_SYM * sizeof(cf32);

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

    // Create streams with different priorities
    // Stream priority: lower value = higher priority
    cudaStream_t streams[MAX_CELLS];
    cudaStream_t prio_streams[MAX_CELLS];
    int least_priority, greatest_priority;
    cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);

    for (int c = 0; c < MAX_CELLS; c++) {
        cudaStreamCreate(&streams[c]);
        // Alternate priorities: even=high, odd=low
        int prio = (c % 2 == 0) ? greatest_priority : least_priority;
        cudaStreamCreateWithPriority(&prio_streams[c], cudaStreamDefault, prio);
    }

    cudaEvent_t estart[MAX_CELLS], estop[MAX_CELLS];
    cudaEvent_t gstart, gstop;
    for (int c = 0; c < MAX_CELLS; c++) {
        cudaEventCreate(&estart[c]); cudaEventCreate(&estop[c]);
    }
    cudaEventCreate(&gstart); cudaEventCreate(&gstop);

    dim3 blk(32);

    // ── Baseline: isolated latency ─────────────────────────────────────────────
    double iso_lat;
    {
        for (int it = 0; it < WARMUP; it++) {
            k_coef_offset<<<nSC, blk, 0, streams[0]>>>(
                cells[0].dC, cells[0].dY, cells[0].dO, nSC, 0);
            CK(cudaDeviceSynchronize());
        }
        std::vector<float> ts;
        for (int it = 0; it < ITERS; it++) {
            cudaEventRecord(estart[0], streams[0]);
            k_coef_offset<<<nSC, blk, 0, streams[0]>>>(
                cells[0].dC, cells[0].dY, cells[0].dO, nSC, 0);
            cudaEventRecord(estop[0], streams[0]);
            CK(cudaDeviceSynchronize());
            float ms; cudaEventElapsedTime(&ms, estart[0], estop[0]);
            ts.push_back(ms);
        }
        iso_lat = pct(ts, 50);
        printf("Isolated V2 p50: %.4f ms\n\n", iso_lat);
    }

    // ── Strategy A: Unrestricted concurrent (baseline) ─────────────────────────
    printf("=== Strategy A: Unrestricted Concurrent (baseline) ===\n");
    printf("All cells launch full grid on separate streams.\n\n");

    printf("%-6s %-12s %-12s %-12s %-12s\n",
           "N", "PerCell(ms)", "Total(ms)", "Degrade", "Throughput");
    printf("------------------------------------------------\n");

    for (int ncells = 1; ncells <= MAX_CELLS; ncells++) {
        for (int it = 0; it < WARMUP; it++) {
            for (int c = 0; c < ncells; c++)
                k_coef_offset<<<nSC, blk, 0, streams[c]>>>(
                    cells[c].dC, cells[c].dY, cells[c].dO, nSC, 0);
            CK(cudaDeviceSynchronize());
        }
        std::vector<float> per_cell, total;
        for (int it = 0; it < ITERS; it++) {
            for (int c = 0; c < ncells; c++) cudaEventRecord(estart[c], streams[c]);
            cudaEventRecord(gstart);
            for (int c = 0; c < ncells; c++)
                k_coef_offset<<<nSC, blk, 0, streams[c]>>>(
                    cells[c].dC, cells[c].dY, cells[c].dO, nSC, 0);
            cudaEventRecord(gstop);
            for (int c = 0; c < ncells; c++) cudaEventRecord(estop[c], streams[c]);
            CK(cudaDeviceSynchronize());
            float ms;
            for (int c = 0; c < ncells; c++) {
                cudaEventElapsedTime(&ms, estart[c], estop[c]);
                per_cell.push_back(ms);
            }
            cudaEventElapsedTime(&ms, gstart, gstop);
            total.push_back(ms);
        }
        double pc_p50 = pct(per_cell, 50);
        double tot_p50 = pct(total, 50);
        double degrade = pc_p50 / iso_lat;
        double throughput = ncells / (tot_p50 / 1000.0);  // cells/sec
        printf("%-6d %-12.4f %-12.4f %-12.2f %-12.0f\n",
               ncells, pc_p50, tot_p50, degrade, throughput);
    }
    printf("\n");

    // ── Strategy B: Serial (back-to-back on one stream) ────────────────────────
    printf("=== Strategy B: Serial (back-to-back) ===\n");
    printf("All cells on stream 0, sequential.\n\n");

    printf("%-6s %-12s %-12s %-12s %-12s\n",
           "N", "PerCell(ms)", "Total(ms)", "Degrade", "Throughput");
    printf("------------------------------------------------\n");

    for (int ncells = 1; ncells <= MAX_CELLS; ncells++) {
        for (int it = 0; it < WARMUP; it++) {
            for (int c = 0; c < ncells; c++)
                k_coef_offset<<<nSC, blk, 0, streams[0]>>>(
                    cells[c].dC, cells[c].dY, cells[c].dO, nSC, 0);
            CK(cudaDeviceSynchronize());
        }
        std::vector<float> per_cell, total;
        for (int it = 0; it < ITERS; it++) {
            cudaEventRecord(gstart);
            for (int c = 0; c < ncells; c++) {
                cudaEventRecord(estart[c], streams[0]);
                k_coef_offset<<<nSC, blk, 0, streams[0]>>>(
                    cells[c].dC, cells[c].dY, cells[c].dO, nSC, 0);
                cudaEventRecord(estop[c], streams[0]);
                cudaEventSynchronize(estop[c]);
                float ms; cudaEventElapsedTime(&ms, estart[c], estop[c]);
                per_cell.push_back(ms);
            }
            cudaEventRecord(gstop);
            CK(cudaDeviceSynchronize());
            float ms; cudaEventElapsedTime(&ms, gstart, gstop);
            total.push_back(ms);
        }
        double pc_p50 = pct(per_cell, 50);
        double tot_p50 = pct(total, 50);
        double degrade = pc_p50 / iso_lat;
        double throughput = ncells / (tot_p50 / 1000.0);
        printf("%-6d %-12.4f %-12.4f %-12.2f %-12.0f\n",
               ncells, pc_p50, tot_p50, degrade, throughput);
    }
    printf("\n");

    // ── Strategy C: Stream priority (high/low alternating) ─────────────────────
    printf("=== Strategy C: Stream Priority (alternating high/low) ===\n");
    printf("Even cells get high priority, odd cells get low priority.\n\n");

    printf("%-6s %-12s %-12s %-12s %-12s %-12s %-12s\n",
           "N", "HiCell(ms)", "LoCell(ms)", "Total(ms)", "HiDegrade", "LoDegrade", "Throughput");
    printf("------------------------------------------------------------------\n");

    for (int ncells = 1; ncells <= MAX_CELLS; ncells++) {
        for (int it = 0; it < WARMUP; it++) {
            for (int c = 0; c < ncells; c++)
                k_coef_offset<<<nSC, blk, 0, prio_streams[c]>>>(
                    cells[c].dC, cells[c].dY, cells[c].dO, nSC, 0);
            CK(cudaDeviceSynchronize());
        }
        std::vector<float> hi_cell, lo_cell, total;
        for (int it = 0; it < ITERS; it++) {
            for (int c = 0; c < ncells; c++) cudaEventRecord(estart[c], prio_streams[c]);
            cudaEventRecord(gstart);
            for (int c = 0; c < ncells; c++)
                k_coef_offset<<<nSC, blk, 0, prio_streams[c]>>>(
                    cells[c].dC, cells[c].dY, cells[c].dO, nSC, 0);
            cudaEventRecord(gstop);
            for (int c = 0; c < ncells; c++) cudaEventRecord(estop[c], prio_streams[c]);
            CK(cudaDeviceSynchronize());
            float ms;
            for (int c = 0; c < ncells; c++) {
                cudaEventElapsedTime(&ms, estart[c], estop[c]);
                if (c % 2 == 0) hi_cell.push_back(ms);
                else lo_cell.push_back(ms);
            }
            cudaEventElapsedTime(&ms, gstart, gstop);
            total.push_back(ms);
        }
        double hi_p50 = hi_cell.empty() ? 0 : pct(hi_cell, 50);
        double lo_p50 = lo_cell.empty() ? 0 : pct(lo_cell, 50);
        double tot_p50 = pct(total, 50);
        double hi_deg = hi_p50 / iso_lat;
        double lo_deg = lo_p50 / iso_lat;
        double throughput = ncells / (tot_p50 / 1000.0);
        printf("%-6d %-12.4f %-12.4f %-12.4f %-12.2f %-12.2f %-12.0f\n",
               ncells, hi_p50, lo_p50, tot_p50, hi_deg, lo_deg, throughput);
    }
    printf("\n");

    // ── Strategy D: Chunked concurrent (split SC range per cell) ───────────────
    // Instead of each cell processing all nSC subcarriers,
    // each cell processes a chunk. This simulates SM partitioning by
    // reducing grid size per cell.
    // NOTE: This changes the problem (each cell does less work).
    // We measure: if we split 1584 SC among N cells, total time.
    printf("=== Strategy D: Chunked (split SC range among cells) ===\n");
    printf("Each cell processes nSC/N subcarriers. Total work = 1 cell.\n");
    printf("This simulates data parallelism (not multi-cell, but shows\n");
    printf("the overhead of splitting work across streams).\n\n");

    printf("%-6s %-12s %-12s %-12s\n",
           "N", "PerChunk(ms)", "Total(ms)", "Speedup_vs_1cell");
    printf("--------------------------------------------\n");

    for (int ncells = 1; ncells <= MAX_CELLS; ncells++) {
        int sc_per_cell = (nSC + ncells - 1) / ncells;
        int actual_per = std::min(sc_per_cell, nSC - 0);  // first cell
        // Each cell processes [c*sc_per_cell, min((c+1)*sc_per_cell, nSC))
        for (int it = 0; it < WARMUP; it++) {
            for (int c = 0; c < ncells; c++) {
                int offset = c * sc_per_cell;
                int count = std::min(sc_per_cell, nSC - offset);
                if (count <= 0) continue;
                k_coef_offset<<<count, blk, 0, streams[c]>>>(
                    cells[0].dC, cells[0].dY, cells[0].dO, count, offset);
            }
            CK(cudaDeviceSynchronize());
        }
        std::vector<float> per_chunk, total;
        for (int it = 0; it < ITERS; it++) {
            for (int c = 0; c < ncells; c++) cudaEventRecord(estart[c], streams[c]);
            cudaEventRecord(gstart);
            for (int c = 0; c < ncells; c++) {
                int offset = c * sc_per_cell;
                int count = std::min(sc_per_cell, nSC - offset);
                if (count <= 0) continue;
                k_coef_offset<<<count, blk, 0, streams[c]>>>(
                    cells[0].dC, cells[0].dY, cells[0].dO, count, offset);
            }
            cudaEventRecord(gstop);
            for (int c = 0; c < ncells; c++) cudaEventRecord(estop[c], streams[c]);
            CK(cudaDeviceSynchronize());
            float ms;
            for (int c = 0; c < ncells; c++) {
                int offset = c * sc_per_cell;
                if (offset >= nSC) continue;
                cudaEventElapsedTime(&ms, estart[c], estop[c]);
                per_chunk.push_back(ms);
            }
            cudaEventElapsedTime(&ms, gstart, gstop);
            total.push_back(ms);
        }
        double pc_p50 = per_chunk.empty() ? 0 : pct(per_chunk, 50);
        double tot_p50 = pct(total, 50);
        // Compare to processing all nSC on one stream
        double speedup = iso_lat / tot_p50;
        printf("%-6d %-12.4f %-12.4f %-12.2f\n",
               ncells, pc_p50, tot_p50, speedup);
    }
    printf("\n");

    // ── Strategy E: Batched serial with overlap (best of both?) ────────────────
    // Run cells in small batches of 2, serially between batches
    printf("=== Strategy E: Batched Serial (batch=2, serial between batches) ===\n");
    printf("Run 2 cells concurrent, then next 2, etc.\n");
    printf("Reduces contention while maintaining some parallelism.\n\n");

    printf("%-6s %-12s %-12s %-12s %-12s\n",
           "N", "PerCell(ms)", "Total(ms)", "Degrade", "Throughput");
    printf("------------------------------------------------\n");

    for (int ncells = 2; ncells <= MAX_CELLS; ncells++) {
        int batch_size = 2;
        int num_batches = (ncells + batch_size - 1) / batch_size;

        for (int it = 0; it < WARMUP; it++) {
            for (int b = 0; b < num_batches; b++) {
                int start = b * batch_size;
                int end = std::min(start + batch_size, ncells);
                for (int c = start; c < end; c++)
                    k_coef_offset<<<nSC, blk, 0, streams[c-start]>>>(
                        cells[c].dC, cells[c].dY, cells[c].dO, nSC, 0);
                CK(cudaDeviceSynchronize());
            }
        }
        std::vector<float> per_cell, total;
        for (int it = 0; it < ITERS; it++) {
            cudaEventRecord(gstart);
            for (int b = 0; b < num_batches; b++) {
                int start = b * batch_size;
                int end = std::min(start + batch_size, ncells);
                for (int c = start; c < end; c++)
                    cudaEventRecord(estart[c-start], streams[c-start]);
                for (int c = start; c < end; c++)
                    k_coef_offset<<<nSC, blk, 0, streams[c-start]>>>(
                        cells[c].dC, cells[c].dY, cells[c].dO, nSC, 0);
                for (int c = start; c < end; c++)
                    cudaEventRecord(estop[c-start], streams[c-start]);
                CK(cudaDeviceSynchronize());
                float ms;
                for (int c = start; c < end; c++) {
                    cudaEventElapsedTime(&ms, estart[c-start], estop[c-start]);
                    per_cell.push_back(ms);
                }
            }
            cudaEventRecord(gstop);
            CK(cudaDeviceSynchronize());
            float ms; cudaEventElapsedTime(&ms, gstart, gstop);
            total.push_back(ms);
        }
        double pc_p50 = pct(per_cell, 50);
        double tot_p50 = pct(total, 50);
        double degrade = pc_p50 / iso_lat;
        double throughput = ncells / (tot_p50 / 1000.0);
        printf("%-6d %-12.4f %-12.4f %-12.2f %-12.0f\n",
               ncells, pc_p50, tot_p50, degrade, throughput);
    }
    printf("\n");

    // ── Summary comparison ─────────────────────────────────────────────────────
    printf("=== Summary: Throughput comparison at N=4 and N=8 ===\n");
    printf("Strategy         N=4_total(ms)  N=4_throughput  N=8_total(ms)  N=8_throughput\n");
    printf("-----------------------------------------------------------------------\n");
    printf("(See individual strategy outputs above for values)\n");

    // Cleanup
    for (int c = 0; c < MAX_CELLS; c++) {
        cudaFree(cells[c].dC); cudaFree(cells[c].dY); cudaFree(cells[c].dO);
        cudaStreamDestroy(streams[c]); cudaStreamDestroy(prio_streams[c]);
        cudaEventDestroy(estart[c]); cudaEventDestroy(estop[c]);
    }
    cudaEventDestroy(gstart); cudaEventDestroy(gstop);
    return 0;
}
