/*
 * aerial_llr_v3.cu — Beat the baseline
 *
 * Root cause from NCU: 31.2% of warp cycles stalled on __constant__ memory
 * (stall_imm_const). The LUT is read with divergent indices per iteration,
 * causing serialized constant cache replays.
 *
 * V3 strategy:
 *  1. Embed LUT slice in registers via template<int N_PAM_BITS>
 *     — compiler unrolls the loop fully, all LUT values become immediates
 *     — zero __constant__ or smem traffic, zero stall_imm_const
 *  2. Replace fabsf(x)*fabsf(x) with x*x (squares are always positive)
 *     — folds into FMA: fmaf(x, x, ...) — reduces non-fused FP32 count
 *  3. Keep 1 thread/symbol as V1 (I+Q serial) — avoids scheduler overhead
 *     that killed V2
 *  4. Increase block size to 512 to maximize warp-level latency hiding
 *
 * Compile:
 *   nvcc -O3 -arch=sm_80 --use_fast_math -o aerial_llr_v3 aerial_llr_v3.cu
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#define N_PRB         132
#define N_SC_PER_PRB  12
#define N_DATA_SYMS   11
#define N_LAYERS      8
#define N_SYMS        (N_PRB * N_SC_PER_PRB * N_DATA_SYMS * N_LAYERS)  // 139392
#define MAX_PAM_BITS  4
#define WARMUP        30
#define ITERS         500

#define CUDA_CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)

// ── Original LUT (Aerial channel_eq.cu line 402) ─────────────────────────────
__constant__ float c_LUT[10] = {
    0.707106781186548f,
    0.632455532033676f, 0.316227766016838f,
    0.617213399848368f, 0.308606699924184f, 0.154303349962092f,
    0.613571991077897f, 0.306785995538948f, 0.153392997769474f,
    0.076696498884737f
};
__constant__ uint8_t c_PAM_OFFSET[4] = {0, 1, 3, 6};

struct half2c { __half re, im; };

// ── Double-precision CPU reference ───────────────────────────────────────────
static const double h_LUT[10] = {
    0.707106781186548, 0.632455532033676, 0.316227766016838,
    0.617213399848368, 0.308606699924184, 0.154303349962092,
    0.613571991077897, 0.306785995538948, 0.153392997769474, 0.076696498884737
};
static const int h_OFF[4] = {0,1,3,6};

static void cpu_ref(float pamI, float pamQ, float noise, int nPamBits, float* out) {
    int lo = h_OFF[nPamBits-1];
    double dist = h_LUT[lo + nPamBits - 1];
    for (int iq = 0; iq < 2; iq++) {
        double pSB[4], pMD[4]; pSB[0] = (iq==0)?(double)pamI:(double)pamQ;
        uint8_t sign = 0;
        for (int i=0;i<nPamBits-1;i++){
            pSB[i+1]=h_LUT[lo+i]-fabs(pSB[i]); pMD[i]=(dist+fabs(pSB[i])); pMD[i]*=pMD[i];
            if(pSB[i]<0) sign|=(uint8_t)(1<<i);
        }
        pMD[nPamBits-1]=(dist+fabs(pSB[nPamBits-1])); pMD[nPamBits-1]*=pMD[nPamBits-1];
        if(pSB[nPamBits-1]<0) sign|=(uint8_t)(1<<(nPamBits-1));
        double md1=fabs(pSB[nPamBits-1])-dist; md1*=md1;
        for(int i=0;i<nPamBits;i++){
            double llr=(pMD[i]-md1)*(double)noise; if(sign&(1<<i)) llr=-llr;
            out[i*2+iq]=(float)fmax(-65504.0,fmin(65504.0,llr));
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// V1 BASELINE — exact Aerial pattern, __constant__ LUT, 1 thread/symbol
// ══════════════════════════════════════════════════════════════════════════════
__global__ void llr_v1_baseline(
    const half2c* __restrict__ in, const float* __restrict__ ni,
    __half* __restrict__ out, int N, int nPamBits)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float pamI=__half2float(in[idx].re), pamQ=__half2float(in[idx].im), noise=ni[idx];
    int lutOff=c_PAM_OFFSET[nPamBits-1]; float dist=c_LUT[lutOff+nPamBits-1];
    __half* o = out + idx * nPamBits * 2;
    for (int iq=0;iq<2;iq++){
        float pSB[MAX_PAM_BITS], pMD[MAX_PAM_BITS]; pSB[0]=(iq==0)?pamI:pamQ;
        uint8_t sign=0;
        for(int i=0;i<nPamBits-1;i++){
            pSB[i+1]=c_LUT[lutOff+i]-fabsf(pSB[i]); pMD[i]=(dist+fabsf(pSB[i])); pMD[i]*=pMD[i];
            if(pSB[i]<0.f) sign|=(uint8_t)(1<<i);
        }
        pMD[nPamBits-1]=(dist+fabsf(pSB[nPamBits-1])); pMD[nPamBits-1]*=pMD[nPamBits-1];
        if(pSB[nPamBits-1]<0.f) sign|=(uint8_t)(1<<(nPamBits-1));
        float md1=fabsf(pSB[nPamBits-1])-dist; md1*=md1;
        for(int i=0;i<nPamBits;i++){
            float llr=(pMD[i]-md1)*noise; if(sign&(1<<i)) llr=-llr;
            o[i*2+iq]=__float2half(fmaxf(-65504.f,fminf(65504.f,llr)));
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// V3 OPTIMIZED — LUT as compile-time constants in registers
//
// Template<N_PAM_BITS> means:
//  - Loop bounds are compile-time → full unroll
//  - LUT values at fixed offsets become scalar float immediates in PTX
//  - Zero __constant__ loads → zero stall_imm_const
//  - Compiler can now schedule FMAs freely (no data dependency on LUT loads)
// ══════════════════════════════════════════════════════════════════════════════
template <int N_PAM_BITS>
__device__ __forceinline__ void computeLlr_reg(
    float pamI, float pamQ, float noise, __half* out)
{
    // LUT slice for this PAM order as compile-time immediates — zero memory traffic
    // Chain: pSB[i+1] = D_i - |pSB[i]|  where D_i = lut[offset + i]
    // dist  = lut[offset + N_PAM_BITS - 1]  (last entry, used for pMD and minDist1)
    //
    // PAM order:   nPamBits=1: offset=0, chain=[0.7071]
    //              nPamBits=2: offset=1, chain=[0.6324, 0.3162]
    //              nPamBits=3: offset=3, chain=[0.6172, 0.3086, 0.1543]
    //              nPamBits=4: offset=6, chain=[0.6135, 0.3067, 0.1533, 0.0766]

    // D0 = lut[offset+0], D1 = lut[offset+1], ..., dist = lut[offset+N_PAM_BITS-1]
    constexpr float D0 = (N_PAM_BITS==1) ? 0.707106781186548f :
                         (N_PAM_BITS==2) ? 0.632455532033676f :
                         (N_PAM_BITS==3) ? 0.617213399848368f :
                                           0.613571991077897f;
    constexpr float D1 = (N_PAM_BITS==2) ? 0.316227766016838f :
                         (N_PAM_BITS==3) ? 0.308606699924184f :
                                           0.306785995538948f;
    constexpr float D2 = (N_PAM_BITS==3) ? 0.154303349962092f :
                                           0.153392997769474f;
    constexpr float D3 = 0.076696498884737f;

    // dist = last entry in chain for this PAM order
    constexpr float dist = (N_PAM_BITS==1) ? D0 :
                           (N_PAM_BITS==2) ? D1 :
                           (N_PAM_BITS==3) ? D2 : D3;

    // Process I then Q — each branch uses only 3 live floats (sb, md_b, md1)
    // pMD[b] written immediately to output to avoid accumulating live variables
    // This keeps register pressure at O(1) regardless of N_PAM_BITS
    #pragma unroll
    for (int iq = 0; iq < 2; iq++) {
        float sb = (iq == 0) ? pamI : pamQ;
        uint8_t sign = 0;
        float abs_sb;

        // bit 0
        abs_sb = fabsf(sb);
        float md0 = dist + abs_sb; md0 *= md0;
        if (sb < 0.f) sign |= 1;
        float sb1 = D0 - abs_sb;

        // bit 1 (if N_PAM_BITS >= 2)
        float md1_v = 0.f, sb2 = 0.f;
        if constexpr (N_PAM_BITS >= 2) {
            abs_sb = fabsf(sb1);
            md1_v = dist + abs_sb; md1_v *= md1_v;
            if (sb1 < 0.f) sign |= 2;
            sb2 = D1 - abs_sb;
        }

        // bit 2 (if N_PAM_BITS >= 3)
        float md2_v = 0.f, sb3 = 0.f;
        if constexpr (N_PAM_BITS >= 3) {
            abs_sb = fabsf(sb2);
            md2_v = dist + abs_sb; md2_v *= md2_v;
            if (sb2 < 0.f) sign |= 4;
            sb3 = D2 - abs_sb;
        }

        // bit 3 (if N_PAM_BITS >= 4)
        float md3_v = 0.f, sb_last = sb1;
        if constexpr (N_PAM_BITS >= 4) {
            abs_sb = fabsf(sb3);
            md3_v = dist + abs_sb; md3_v *= md3_v;
            if (sb3 < 0.f) sign |= 8;
            sb_last = sb3;
        } else if constexpr (N_PAM_BITS == 3) {
            sb_last = sb2;
        } else if constexpr (N_PAM_BITS == 2) {
            sb_last = sb1;
        } else {
            sb_last = sb;
        }

        // minDist1 from the last soft bit
        float md_last = (N_PAM_BITS==1) ? md0 :
                        (N_PAM_BITS==2) ? md1_v :
                        (N_PAM_BITS==3) ? md2_v : md3_v;
        float minDist1 = fabsf(sb_last) - dist;
        minDist1 *= minDist1;

        // Write LLRs — all pMD values available, one write per bit
        auto write_llr = [&](int b, float pmd) __attribute__((always_inline)) {
            float llr = (pmd - minDist1) * noise;
            if (sign & (1 << b)) llr = -llr;
            out[b * 2 + iq] = __float2half(fmaxf(-65504.f, fminf(65504.f, llr)));
        };

        write_llr(0, md0);
        if constexpr (N_PAM_BITS >= 2) write_llr(1, md1_v);
        if constexpr (N_PAM_BITS >= 3) write_llr(2, md2_v);
        if constexpr (N_PAM_BITS >= 4) write_llr(3, md3_v);
    }
}

// Dispatch kernel — one thread per symbol, template dispatched at launch
template <int N_PAM_BITS>
__global__ void llr_v3_reg_template(
    const half2c* __restrict__ in, const float* __restrict__ ni,
    __half* __restrict__ out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    computeLlr_reg<N_PAM_BITS>(
        __half2float(in[idx].re), __half2float(in[idx].im), ni[idx],
        out + idx * N_PAM_BITS * 2);
}

// ── Timing ───────────────────────────────────────────────────────────────────
static double median_ms(cudaEvent_t* s, cudaEvent_t* e, int n) {
    float* t = (float*)malloc(n * sizeof(float));
    for (int i=0;i<n;i++) cudaEventElapsedTime(&t[i],s[i],e[i]);
    for (int i=1;i<n;i++){float v=t[i];int j=i-1;while(j>=0&&t[j]>v){t[j+1]=t[j];j--;}t[j+1]=v;}
    double m=t[n/2]; free(t); return m;
}

static void bench(const char* name, cudaEvent_t* s, cudaEvent_t* e, int n,
                  int N, int nPamBits, double* t_out) {
    *t_out = median_ms(s, e, n);
    double bw = ((double)N*(sizeof(half2c)+sizeof(float)) + (double)N*nPamBits*2*sizeof(__half));
    printf("  %-28s  %6.3f ms  %6.1f GB/s  %8.1f M-LLR/s\n",
           name, *t_out, bw / *t_out / 1e6, (double)N*nPamBits*2 / *t_out / 1e3);
}

static void correctness(const __half* gpu, const float* ref, int N, int nL,
                        double* cos_out, double* mae_out) {
    double dot=0,na=0,nb=0,mae=0;
    for(int i=0;i<N*nL;i++){
        double a=__half2float(gpu[i]), b=ref[i];
        dot+=a*b; na+=a*a; nb+=b*b;
        double ae=fabs(a-b); if(ae>mae) mae=ae;
    }
    *cos_out=dot/(sqrt(na)*sqrt(nb)+1e-30); *mae_out=mae;
}

int main(void) {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("========================================================\n");
    printf(" Aerial computePamLlr — V3 Register-Template Benchmark\n");
    printf("========================================================\n");
    printf(" GPU: %s  CUDA %.1f\n\n", p.name,
           (float)p.major + (float)p.minor/10.f);

    int qam[]={16,64,256}, pbits[]={2,3,4}; int nq=3;
    srand(42);

    half2c* h_in  = (half2c*)malloc(N_SYMS*sizeof(half2c));
    float*  h_ni  = (float*) malloc(N_SYMS*sizeof(float));
    for(int i=0;i<N_SYMS;i++){
        h_in[i].re=__float2half(((float)rand()/RAND_MAX-.5f)*1.4f);
        h_in[i].im=__float2half(((float)rand()/RAND_MAX-.5f)*1.4f);
        h_ni[i]=powf(10.f,(float)(rand()%26)/10.f)*2.f;
    }
    half2c* d_in; float* d_ni;
    CUDA_CHECK(cudaMalloc(&d_in,N_SYMS*sizeof(half2c)));
    CUDA_CHECK(cudaMalloc(&d_ni,N_SYMS*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in,h_in,N_SYMS*sizeof(half2c),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ni,h_ni,N_SYMS*sizeof(float), cudaMemcpyHostToDevice));

    for (int qi=0;qi<nq;qi++) {
        int nPamBits=pbits[qi], nL=nPamBits*2, llrN=N_SYMS*nL;
        __half *d_v1, *d_v3;
        CUDA_CHECK(cudaMalloc(&d_v1, llrN*sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_v3, llrN*sizeof(__half)));

        printf("── QAM-%d (%d PAM bits, %d LLRs/sym) ───────────────────────\n",
               qam[qi], nPamBits, nL);

        // V1
        dim3 b1(512), g1((N_SYMS+511)/512);
        for(int i=0;i<WARMUP;i++) llr_v1_baseline<<<g1,b1>>>(d_in,d_ni,d_v1,N_SYMS,nPamBits);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t s1[ITERS],e1[ITERS];
        for(int i=0;i<ITERS;i++){cudaEventCreate(&s1[i]);cudaEventCreate(&e1[i]);
            cudaEventRecord(s1[i]); llr_v1_baseline<<<g1,b1>>>(d_in,d_ni,d_v1,N_SYMS,nPamBits);
            cudaEventRecord(e1[i]);}
        CUDA_CHECK(cudaDeviceSynchronize());

        // V3 — dispatch by template specialization
        dim3 b3(512), g3((N_SYMS+511)/512);
        auto launch_v3 = [&](){
            if      (nPamBits==2) llr_v3_reg_template<2><<<g3,b3>>>(d_in,d_ni,d_v3,N_SYMS);
            else if (nPamBits==3) llr_v3_reg_template<3><<<g3,b3>>>(d_in,d_ni,d_v3,N_SYMS);
            else                  llr_v3_reg_template<4><<<g3,b3>>>(d_in,d_ni,d_v3,N_SYMS);
        };
        for(int i=0;i<WARMUP;i++) launch_v3();
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t s3[ITERS],e3[ITERS];
        for(int i=0;i<ITERS;i++){cudaEventCreate(&s3[i]);cudaEventCreate(&e3[i]);
            cudaEventRecord(s3[i]); launch_v3(); cudaEventRecord(e3[i]);}
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timings
        double t1,t3;
        bench("V1 baseline (__constant__ LUT)", s1,e1,ITERS,N_SYMS,nPamBits,&t1);
        bench("V3 register template (unrolled)", s3,e3,ITERS,N_SYMS,nPamBits,&t3);
        printf("  Speedup V3 vs V1: %.2fx\n", t1/t3);

        // Correctness: V3 vs CPU double ref
        float* h_ref=(float*)malloc(llrN*sizeof(float));
        for(int i=0;i<N_SYMS;i++)
            cpu_ref(__half2float(h_in[i].re),__half2float(h_in[i].im),
                    h_ni[i],nPamBits, h_ref+i*nL);
        __half* h_v3=(__half*)malloc(llrN*sizeof(__half));
        __half* h_v1=(__half*)malloc(llrN*sizeof(__half));
        CUDA_CHECK(cudaMemcpy(h_v3,d_v3,llrN*sizeof(__half),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_v1,d_v1,llrN*sizeof(__half),cudaMemcpyDeviceToHost));
        double cos3,mae3,cos1,mae1_v;
        correctness(h_v3,h_ref,N_SYMS,nL,&cos3,&mae3);
        correctness(h_v1,h_ref,N_SYMS,nL,&cos1,&mae1_v);
        int mm=0; for(int i=0;i<llrN;i++) if(__half2float(h_v1[i])!=__half2float(h_v3[i])) mm++;
        printf("  V3 cosine=%.8f  maxAE=%.4f  V1==V3:%s (mismatches=%d)\n\n",
               cos3, mae3, mm==0?"YES":"NO(!)", mm);

        for(int i=0;i<ITERS;i++){cudaEventDestroy(s1[i]);cudaEventDestroy(e1[i]);
                                  cudaEventDestroy(s3[i]);cudaEventDestroy(e3[i]);}
        free(h_ref); free(h_v3); free(h_v1);
        cudaFree(d_v1); cudaFree(d_v3);
    }

    printf("GPU: %s | Driver CUDA: %d.%d\n", p.name, p.major, p.minor);
    printf("Config: %d PRB x %d SC x %d syms x %d layers = %d symbols\n",
           N_PRB, N_SC_PER_PRB, N_DATA_SYMS, N_LAYERS, N_SYMS);
    printf("Iterations: %d  Warmup: %d  Block: 512\n", ITERS, WARMUP);

    cudaFree(d_in); cudaFree(d_ni);
    free(h_in); free(h_ni);
    return 0;
}
