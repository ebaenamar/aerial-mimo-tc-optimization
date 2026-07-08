/*
 * coef_scaling.cu — Deployment analysis for coef_apply WMMA optimization
 *
 * Uses the EXACT same V1 (Aerial production pattern) and V2 (WMMA) kernels
 * from aerial_coef_wmma.cu, templated over nAnt and nSym.
 *
 * Sweeps:
 *   1. nAnt:  8, 16, 32, 64, 128  (at 100MHz / 8L)
 *   2. nPRB: 11, 25, 52, 106, 132, 264  (BW: 5→200MHz, 64ant / 8L)
 *
 * Compile:
 *   nvcc -O3 -arch=sm_80 -std=c++17 coef_scaling.cu -o coef_scaling
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
using namespace nvcuda;

#define CUDA_CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)

#define WM 16
#define WN 16
#define WK 16
#define N_LAYERS   8
#define N_SC_PRB  12
#define WARMUP    30
#define ITERS    200

struct cf32 { float re, im; };
__device__ __forceinline__ cf32 cma(cf32 a, cf32 b, cf32 acc) {
    return {acc.re + a.re*b.re - a.im*b.im, acc.im + a.re*b.im + a.im*b.re};
}

// ── V1: Aerial scalar pattern (FP32 with smem cache for C) ───────────────────
template<int N_ANT, int N_SYM>
__global__ void v1_scalar(
    const cf32* __restrict__ C,
    const cf32* __restrict__ Y_rx,
    cf32*       __restrict__ Y_eq,
    int nSC)
{
    int prb   = blockIdx.x;
    int layer = blockIdx.y;
    int sc_l  = threadIdx.x;
    int sym   = threadIdx.y;
    int sc    = prb * N_SC_PRB + sc_l;
    if (sc >= nSC) return;

    __shared__ cf32 sC[N_ANT][N_SC_PRB];
    if (sym == 0) {
        for (int a = 0; a < N_ANT; a++)
            sC[a][sc_l] = C[sc * N_LAYERS * N_ANT + layer * N_ANT + a];
    }
    __syncthreads();

    cf32 acc = {0.f, 0.f};
    cf32 Cn = sC[0][sc_l];
    cf32 Yn = Y_rx[sc * N_SYM * N_ANT + sym * N_ANT + 0];
    #pragma unroll 4
    for (int a = 0; a + 1 < N_ANT; a++) {
        acc = cma(Cn, Yn, acc);
        Cn = sC[a+1][sc_l];
        Yn = Y_rx[sc * N_SYM * N_ANT + sym * N_ANT + a + 1];
    }
    acc = cma(Cn, Yn, acc);
    Y_eq[sc * N_LAYERS * N_SYM + layer * N_SYM + sym] = acc;
}

// ── V2: WMMA 1-warp-per-SC ───────────────────────────────────────────────────
// Pads nAnt to multiple of WK=16, nSym to WN=16
template<int N_ANT, int N_SYM>
__global__ void v2_wmma(
    const cf32* __restrict__ C,
    const cf32* __restrict__ Y_rx,
    cf32*       __restrict__ Y_eq,
    int nSC)
{
    constexpr int ANT_PAD = ((N_ANT + WK - 1) / WK) * WK;  // next mult of 16

    int sc   = blockIdx.x;
    int lane = threadIdx.x;
    if (sc >= nSC) return;

    __shared__ __half smA_re[WM][ANT_PAD];
    __shared__ __half smA_im[WM][ANT_PAD];
    __shared__ __half smB_re[ANT_PAD][WN];
    __shared__ __half smB_im[ANT_PAD][WN];

    // Fill A: C[sc, layer, ant]
    for (int idx = lane; idx < WM * ANT_PAD; idx += 32) {
        int row = idx / ANT_PAD, col = idx % ANT_PAD;
        if (row < N_LAYERS && col < N_ANT) {
            cf32 c = C[sc * N_LAYERS * N_ANT + row * N_ANT + col];
            smA_re[row][col] = __float2half(c.re);
            smA_im[row][col] = __float2half(c.im);
        } else {
            smA_re[row][col] = __float2half(0.f);
            smA_im[row][col] = __float2half(0.f);
        }
    }

    // Fill B: Y_rx[sc, sym, ant]
    for (int idx = lane; idx < ANT_PAD * WN; idx += 32) {
        int ant = idx / WN, sym = idx % WN;
        if (ant < N_ANT && sym < N_SYM) {
            cf32 y = Y_rx[sc * N_SYM * N_ANT + sym * N_ANT + ant];
            smB_re[ant][sym] = __float2half(y.re);
            smB_im[ant][sym] = __float2half(y.im);
        } else {
            smB_re[ant][sym] = __float2half(0.f);
            smB_im[ant][sym] = __float2half(0.f);
        }
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> rr, ri, ir, ii;
    wmma::fill_fragment(rr, 0.f); wmma::fill_fragment(ri, 0.f);
    wmma::fill_fragment(ir, 0.f); wmma::fill_fragment(ii, 0.f);

    #pragma unroll
    for (int k = 0; k < ANT_PAD; k += WK) {
        wmma::load_matrix_sync(a_frag, &smA_re[0][k], ANT_PAD);
        wmma::load_matrix_sync(b_frag, &smB_re[k][0], WN);
        wmma::mma_sync(rr, a_frag, b_frag, rr);
        wmma::load_matrix_sync(b_frag, &smB_im[k][0], WN);
        wmma::mma_sync(ri, a_frag, b_frag, ri);
        wmma::load_matrix_sync(a_frag, &smA_im[0][k], ANT_PAD);
        wmma::mma_sync(ir, a_frag, b_frag, ir);
        wmma::load_matrix_sync(b_frag, &smB_re[k][0], WN);
        wmma::mma_sync(ii, a_frag, b_frag, ii);
    }

    __shared__ float tmp[4][WM][WN];
    wmma::store_matrix_sync(&tmp[0][0][0], rr, WN, wmma::mem_row_major);
    wmma::store_matrix_sync(&tmp[1][0][0], ri, WN, wmma::mem_row_major);
    wmma::store_matrix_sync(&tmp[2][0][0], ir, WN, wmma::mem_row_major);
    wmma::store_matrix_sync(&tmp[3][0][0], ii, WN, wmma::mem_row_major);
    __syncwarp();

    for (int idx = lane; idx < WM * WN; idx += 32) {
        int layer = idx / WN, sym = idx % WN;
        if (layer >= N_LAYERS || sym >= N_SYM) continue;
        float re = tmp[0][layer][sym] - tmp[2][layer][sym];
        float im = tmp[1][layer][sym] + tmp[3][layer][sym];
        Y_eq[sc * N_LAYERS * N_SYM + layer * N_SYM + sym] = {re, im};
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────
double median_ms(float* t, int n) {
    for(int i=1;i<n;i++){float k=t[i];int j=i-1;while(j>=0&&t[j]>k){t[j+1]=t[j];j--;}t[j+1]=k;}
    return t[n/2];
}
double cosine_sim(const cf32* a, const cf32* b, int n) {
    double dot=0,na=0,nb=0;
    for(int i=0;i<n;i++){dot+=a[i].re*b[i].re+a[i].im*b[i].im;
        na+=a[i].re*a[i].re+a[i].im*a[i].im;nb+=b[i].re*b[i].re+b[i].im*b[i].im;}
    return dot/(sqrt(na)*sqrt(nb)+1e-30);
}

// ── run one (nAnt, nPRB) configuration ───────────────────────────────────────
template<int N_ANT, int N_SYM>
void run(int nPRB, const char* tag) {
    int nSC    = nPRB * N_SC_PRB;
    double flop = 2.0*N_LAYERS*nSC*N_ANT*N_SYM*8;  // 8 real flop/complex FMA

    size_t szC   = (size_t)nSC * N_LAYERS * N_ANT * sizeof(cf32);
    size_t szYrx = (size_t)nSC * N_SYM    * N_ANT * sizeof(cf32);
    size_t szYeq = (size_t)nSC * N_LAYERS * N_SYM * sizeof(cf32);

    cf32 *hC=(cf32*)malloc(szC), *hY=(cf32*)malloc(szYrx),
         *ho1=(cf32*)malloc(szYeq), *ho2=(cf32*)malloc(szYeq);
    srand(42);
    auto rf=[]{ return ((float)rand()/RAND_MAX-0.5f)*1.4f; };
    for(size_t i=0;i<szC/sizeof(cf32);i++)   hC[i]={rf(),rf()};
    for(size_t i=0;i<szYrx/sizeof(cf32);i++) hY[i]={rf(),rf()};

    cf32 *dC,*dY,*d1,*d2;
    CUDA_CHECK(cudaMalloc(&dC,szC));  CUDA_CHECK(cudaMalloc(&dY,szYrx));
    CUDA_CHECK(cudaMalloc(&d1,szYeq));CUDA_CHECK(cudaMalloc(&d2,szYeq));
    CUDA_CHECK(cudaMemcpy(dC,hC,szC,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY,hY,szYrx,cudaMemcpyHostToDevice));

    dim3 blk1(N_SC_PRB, N_SYM), grd1(nPRB, N_LAYERS);
    dim3 blk2(32), grd2(nSC);

    for(int i=0;i<WARMUP;i++){
        v1_scalar<N_ANT,N_SYM><<<grd1,blk1>>>(dC,dY,d1,nSC);
        v2_wmma  <N_ANT,N_SYM><<<grd2,blk2>>>(dC,dY,d2,nSC);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float t1[ITERS], t2[ITERS];
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    for(int i=0;i<ITERS;i++){
        float ms;
        cudaEventRecord(s); v1_scalar<N_ANT,N_SYM><<<grd1,blk1>>>(dC,dY,d1,nSC);
        cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); t1[i]=ms;
        cudaEventRecord(s); v2_wmma  <N_ANT,N_SYM><<<grd2,blk2>>>(dC,dY,d2,nSC);
        cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); t2[i]=ms;
    }
    CUDA_CHECK(cudaMemcpy(ho1,d1,szYeq,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ho2,d2,szYeq,cudaMemcpyDeviceToHost));
    double m1=median_ms(t1,ITERS), m2=median_ms(t2,ITERS);
    double g1=flop/m1/1e12, g2=flop/m2/1e12;  // TFLOP/s
    printf("%-38s V1:%6.3fms %5.2fTF  V2:%6.3fms %5.2fTF  spd:%5.2fx  cos:%.5f\n",
           tag, m1,g1, m2,g2, m1/m2,
           cosine_sim(ho1,ho2,(int)(szYeq/sizeof(cf32))));
    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dC); cudaFree(dY); cudaFree(d1); cudaFree(d2);
    free(hC); free(hY); free(ho1); free(ho2);
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("GPU: %s (SM %d.%d)\n\n", p.name, p.major, p.minor);

    printf("=== 1. nAnt scaling  (100 MHz / 132 RB / 8L / 11 sym) ===\n");
    run< 8,11>(132, "nAnt= 8");
    run<16,11>(132, "nAnt=16");
    run<32,11>(132, "nAnt=32");
    run<64,11>(132, "nAnt=64");
    // nAnt=128: smem = 2*(16*128)+(128*16)+4*(16*16)*4 = 4096+2048+4096 = >48KB → skip
    // Larger nAnt needs a different tiling strategy (noted in paper)

    printf("\n=== 2. Bandwidth scaling  (64 ant / 8L / 11 sym) ===\n");
    //                                      RBs  BW label
    run<64,11>( 11, "BW= 5MHz  ( 11 RB)");
    run<64,11>( 25, "BW=10MHz  ( 25 RB)");
    run<64,11>( 52, "BW=20MHz  ( 52 RB)");
    run<64,11>(106, "BW=50MHz  (106 RB)");
    run<64,11>(132, "BW=100MHz (132 RB)");
    run<64,11>(264, "BW=200MHz (264 RB)");

    printf("\n=== 3. Symbol count  (64 ant / 132 RB / 8L) ===\n");
    run<64, 1>(132, "nSym= 1 (1-sym burst)");
    run<64, 4>(132, "nSym= 4");
    run<64, 7>(132, "nSym= 7 (half slot)");
    run<64,11>(132, "nSym=11 (full slot)");
    run<64,14>(132, "nSym=14 (special slot)");

    return 0;
}
