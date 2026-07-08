/*
 * Benchmark standalone: computePamLlr (extraído de Aerial channel_eq.cu)
 *
 * Compara:
 *   v1 (baseline): LUT en __constant__ memory, 1 thread activo por símbolo
 *   v2 (optimized): LUT pre-cargada en smem, 4 threads activos por símbolo (I/Q x 2 PAM halves)
 *
 * Inputs sintéticos: señales equalizadas FP16 complejas [N_SYMS x N_LAYERS]
 * Output: LLRs FP16 [N_SYMS x N_LAYERS x N_BITS]
 *
 * Compilar:
 *   nvcc -O2 -arch=sm_80 -o aerial_llr_bench aerial_llr_bench.cu -lcuda
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

// ── Config ─────────────────────────────────────────────────────────────────
#define N_LAYERS      8
#define N_PRB         132
#define N_SC_PER_PRB  12
#define N_DATA_SYMS   11        // OFDM data symbols per slot (TDD F08)
#define N_SYMS        (N_PRB * N_SC_PER_PRB * N_DATA_SYMS)  // 17424 subcarrier-symbols
#define QAM_ORDER     256       // 256QAM → nPamBits=4 → 8 LLRs per symbol
#define N_PAM_BITS    4
#define N_LLR_PER_SYM (N_PAM_BITS * 2)  // 8 (I + Q)
#define WARMUP        20
#define ITERS         100

// ── LUT (exacta de channel_eq.cu) ──────────────────────────────────────────
__constant__ float LUT_SYMB_DIST_KPAM[] = {
    0.707106781186548f,
    0.632455532033676f, 0.316227766016838f,
    0.617213399848368f, 0.308606699924184f, 0.154303349962092f,
    0.613571991077897f, 0.306785995538948f, 0.153392997769474f, 0.076696498884737f
};
__constant__ uint8_t LUT_PAM_OFFSET[] = {0, 1, 3, 6};  // offset por nPamBits-1

// ── Structs ─────────────────────────────────────────────────────────────────
struct half2c { __half re, im; };  // complejo FP16

// ══════════════════════════════════════════════════════════════════════════
// V1 BASELINE: como Aerial — 1 thread activo por símbolo, LUT en __constant__
// ══════════════════════════════════════════════════════════════════════════
__global__ void llr_baseline_kernel(
    const half2c* __restrict__ softEst,  // [N_SYMS * N_LAYERS]
    const float* __restrict__  noiseInv, // [N_SYMS * N_LAYERS]
    __half* __restrict__       llrOut,   // [N_SYMS * N_LAYERS * N_LLR_PER_SYM]
    int nSyms, int nLayers, int nPamBits)
{
    // 1 thread per (sym, layer) — mismo patrón que Aerial PER_LAYER_THRD_IDX==0
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nSyms * nLayers;
    if (idx >= total) return;

    float pamSymb_I = __half2float(softEst[idx].re);
    float pamSymb_Q = __half2float(softEst[idx].im);
    float noise     = noiseInv[idx];
    int   lutOff    = LUT_PAM_OFFSET[nPamBits - 1];
    float dist      = LUT_SYMB_DIST_KPAM[lutOff + nPamBits - 1];

    __half* pLlr = llrOut + idx * N_LLR_PER_SYM;

    // Process I and Q independently (serial, same thread)
    for (int iq = 0; iq < 2; iq++) {
        float pSoftBits[4], pMinDist2[4];
        pSoftBits[0] = (iq == 0) ? pamSymb_I : pamSymb_Q;
        uint8_t signBmsk = 0;

        for (int i = 0; i < nPamBits - 1; i++) {
            pSoftBits[i+1] = LUT_SYMB_DIST_KPAM[lutOff + i] - fabsf(pSoftBits[i]);
            pMinDist2[i]   = dist + fabsf(pSoftBits[i]);
            pMinDist2[i]  *= pMinDist2[i];
            if (pSoftBits[i] < 0.f) signBmsk |= (1 << i);
        }
        pMinDist2[nPamBits-1]  = dist + fabsf(pSoftBits[nPamBits-1]);
        pMinDist2[nPamBits-1] *= pMinDist2[nPamBits-1];
        if (pSoftBits[nPamBits-1] < 0.f) signBmsk |= (1 << (nPamBits-1));
        float minDist1 = fabsf(pSoftBits[nPamBits-1]) - dist;
        minDist1 *= minDist1;

        for (int i = 0; i < nPamBits; i++) {
            float llr = (pMinDist2[i] - minDist1) * noise;
            if (signBmsk & (1 << i)) llr = -llr;
            llr = fmaxf(-65504.f, fminf(65504.f, llr));
            pLlr[i * 2 + iq] = __float2half(llr);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
// V2 OPTIMIZED: LUT en smem + 8 threads por símbolo (1 por LLR output)
// Técnica del hackathon: preload LUT a smem, warp-level parallelism
// ══════════════════════════════════════════════════════════════════════════
#define THREADS_PER_SYM  8   // N_LLR_PER_SYM = 8 para 256QAM
#define BLOCK_SYMS       32  // símbolos por bloque → blockDim = 32*8 = 256 threads

__global__ void llr_optimized_kernel(
    const half2c* __restrict__ softEst,
    const float* __restrict__  noiseInv,
    __half* __restrict__       llrOut,
    int nSyms, int nLayers, int nPamBits)
{
    // Precargar LUT completa en smem — 10 floats = 40 bytes, trivial
    __shared__ float sLUT[10];
    if (threadIdx.x < 10) sLUT[threadIdx.x] = LUT_SYMB_DIST_KPAM[threadIdx.x];
    __syncthreads();

    // 8 threads por símbolo
    int symThread = threadIdx.x % THREADS_PER_SYM;  // 0-7: qué LLR calcula este thread
    int symLocal  = threadIdx.x / THREADS_PER_SYM;  // símbolo local en el bloque
    int symGlobal = blockIdx.x * BLOCK_SYMS * nLayers + symLocal;
    if (symGlobal >= nSyms * nLayers) return;

    int iq  = symThread % 2;        // 0=I, 1=Q
    int bit = symThread / 2;        // 0-3: qué bit PAM

    float pamSymb_I = __half2float(softEst[symGlobal].re);
    float pamSymb_Q = __half2float(softEst[symGlobal].im);
    float noise     = noiseInv[symGlobal];
    int   lutOff    = LUT_PAM_OFFSET[nPamBits - 1];
    float dist      = sLUT[lutOff + nPamBits - 1];  // desde smem, no __constant__

    float pamSymb = (iq == 0) ? pamSymb_I : pamSymb_Q;

    // Cada thread calcula su propio SoftBits[bit] iterativamente
    float sb = pamSymb;
    float md2 = 0.f;
    uint8_t signBmsk = 0;
    for (int i = 0; i < nPamBits; i++) {
        float next = sLUT[lutOff + i] - fabsf(sb);
        float md   = dist + fabsf(sb);
        if (i < bit) {
            md2 = (i == bit - 1) ? md * md : md2;
            if (sb < 0.f) signBmsk |= (1 << i);
            sb = next;
        }
    }
    // minDist1
    float sb_last = sb;
    for (int i = bit; i < nPamBits - 1; i++) sb_last = sLUT[lutOff + i] - fabsf(sb_last);
    float minDist1 = fabsf(sb_last) - dist;
    minDist1 *= minDist1;
    md2 = dist + fabsf(sb); md2 *= md2;
    if (sb < 0.f) signBmsk |= (1 << bit);

    float llr = (md2 - minDist1) * noise;
    if (signBmsk & (1 << bit)) llr = -llr;
    llr = fmaxf(-65504.f, fminf(65504.f, llr));

    if (bit < nPamBits && symGlobal < nSyms * nLayers)
        llrOut[symGlobal * N_LLR_PER_SYM + bit * 2 + iq] = __float2half(llr);
}

// ── helpers ──────────────────────────────────────────────────────────────
#define CUDA_CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)

double cuda_time_ms(cudaEvent_t s, cudaEvent_t e) {
    float ms; cudaEventElapsedTime(&ms, s, e); return ms;
}

int main() {
    int nSyms   = N_SYMS;
    int nLayers = N_LAYERS;
    int nPamBits= N_PAM_BITS;
    int total   = nSyms * nLayers;
    int llrTotal= total * N_LLR_PER_SYM;

    printf("=== Aerial LLR Kernel Benchmark (standalone) ===\n");
    printf("GPU: "); {
        cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
        printf("%s (SM %d.%d)\n", p.name, p.major, p.minor);
    }
    printf("Config: %d PRBs x %d SC x %d syms x %d layers = %d symbols\n",
           N_PRB, N_SC_PER_PRB, N_DATA_SYMS, N_LAYERS, total);
    printf("QAM256: %d PAM bits → %d LLRs/symbol\n\n", nPamBits, N_LLR_PER_SYM);

    // Allocate & init host data
    half2c* h_softEst  = (half2c*)malloc(total * sizeof(half2c));
    float*  h_noiseInv = (float*)malloc(total * sizeof(float));
    for (int i = 0; i < total; i++) {
        h_softEst[i].re  = __float2half(((float)rand()/RAND_MAX - 0.5f) * 2.f);
        h_softEst[i].im  = __float2half(((float)rand()/RAND_MAX - 0.5f) * 2.f);
        h_noiseInv[i]    = 2.0f + (float)rand()/RAND_MAX;
    }

    // GPU alloc
    half2c *d_softEst; float *d_noiseInv; __half *d_llr_v1, *d_llr_v2;
    CUDA_CHECK(cudaMalloc(&d_softEst,  total * sizeof(half2c)));
    CUDA_CHECK(cudaMalloc(&d_noiseInv, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_llr_v1,   llrTotal * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_llr_v2,   llrTotal * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_softEst,  h_softEst,  total*sizeof(half2c), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_noiseInv, h_noiseInv, total*sizeof(float),  cudaMemcpyHostToDevice));

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    // ── V1 Baseline ─────────────────────────────────────────────────────
    dim3 blk_v1(256);
    dim3 grd_v1((total + 255) / 256);
    for (int i = 0; i < WARMUP; i++)
        llr_baseline_kernel<<<grd_v1, blk_v1>>>(d_softEst, d_noiseInv, d_llr_v1, nSyms, nLayers, nPamBits);
    cudaDeviceSynchronize();

    double t_v1 = 0;
    for (int i = 0; i < ITERS; i++) {
        cudaEventRecord(s);
        llr_baseline_kernel<<<grd_v1, blk_v1>>>(d_softEst, d_noiseInv, d_llr_v1, nSyms, nLayers, nPamBits);
        cudaEventRecord(e); cudaEventSynchronize(e);
        t_v1 += cuda_time_ms(s, e);
    }
    t_v1 /= ITERS;

    // ── V2 Optimized ────────────────────────────────────────────────────
    // 256 threads/block = 32 símbolos × 8 threads/símbolo
    dim3 blk_v2(BLOCK_SYMS * THREADS_PER_SYM);
    dim3 grd_v2((total + BLOCK_SYMS * nLayers - 1) / (BLOCK_SYMS * nLayers));
    for (int i = 0; i < WARMUP; i++)
        llr_optimized_kernel<<<grd_v2, blk_v2>>>(d_softEst, d_noiseInv, d_llr_v2, nSyms, nLayers, nPamBits);
    cudaDeviceSynchronize();

    double t_v2 = 0;
    for (int i = 0; i < ITERS; i++) {
        cudaEventRecord(s);
        llr_optimized_kernel<<<grd_v2, blk_v2>>>(d_softEst, d_noiseInv, d_llr_v2, nSyms, nLayers, nPamBits);
        cudaEventRecord(e); cudaEventSynchronize(e);
        t_v2 += cuda_time_ms(s, e);
    }
    t_v2 /= ITERS;

    // ── Throughput ──────────────────────────────────────────────────────
    double llr_bytes  = (double)llrTotal * sizeof(__half);
    double inp_bytes  = (double)total * (sizeof(half2c) + sizeof(float));
    double total_bytes= llr_bytes + inp_bytes;

    printf("--- V1 Baseline (Aerial pattern: 1 thread/sym, LUT __constant__) ---\n");
    printf("  Latency : %.3f ms\n", t_v1);
    printf("  Throughput: %.2f GB/s  (%.2f M LLRs/s)\n",
           total_bytes / t_v1 / 1e6, (double)llrTotal / t_v1 / 1e3);

    printf("\n--- V2 Optimized (smem LUT + 8 threads/sym) ---\n");
    printf("  Latency : %.3f ms\n", t_v2);
    printf("  Throughput: %.2f GB/s  (%.2f M LLRs/s)\n",
           total_bytes / t_v2 / 1e6, (double)llrTotal / t_v2 / 1e3);

    printf("\n--- Speedup: %.2fx ---\n", t_v1 / t_v2);
    printf("    LLR output size: %.2f MB  |  Input: %.2f MB\n",
           llr_bytes/1e6, inp_bytes/1e6);

    // Cleanup
    cudaFree(d_softEst); cudaFree(d_noiseInv);
    cudaFree(d_llr_v1);  cudaFree(d_llr_v2);
    free(h_softEst); free(h_noiseInv);
    return 0;
}
