/*
 * coef_rigor.cu — rigor pass for the coef_apply headline result.
 * Adds: latency distribution (p5/p50/p95/mean/stddev over 500 runs),
 *       EVM and max relative error (not just cosine similarity),
 *       roofline inputs (bytes moved, FLOPs, arithmetic intensity).
 * Reference config: 64 ant, 132 RB (100 MHz), 8 layers, 11 symbols.
 *
 * nvcc -O3 -arch=sm_80 -std=c++17 coef_rigor.cu -o coef_rigor
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
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
#define WARMUP 50
#define ITERS 500

struct cf32{float re,im;};
__device__ __forceinline__ cf32 cma(cf32 a,cf32 b,cf32 acc){
    return {acc.re+a.re*b.re-a.im*b.im, acc.im+a.re*b.im+a.im*b.re};}

// Aerial production pattern: block per PRB, threads=(N_SC_PRB,N_SYM),
// C cached in shared memory (matches eqMmseSoftDemapKernel_v4). This is the
// fair scalar baseline the paper's 5.5x is measured against.
__global__ void v1(const cf32* C,const cf32* Y,cf32* O,int nSC){
    int prb=blockIdx.x, layer=blockIdx.y, sc_l=threadIdx.x, sym=threadIdx.y;
    int sc=prb*N_SC_PRB+sc_l;
    if(sc>=nSC) return;
    __shared__ cf32 sC[N_ANT][N_SC_PRB];
    if(sym==0){ for(int a=0;a<N_ANT;a++) sC[a][sc_l]=C[sc*N_LAYERS*N_ANT+layer*N_ANT+a]; }
    __syncthreads();
    cf32 acc={0,0};
    cf32 Cn=sC[0][sc_l];
    cf32 Yn=Y[sc*N_SYM*N_ANT+sym*N_ANT+0];
    #pragma unroll 4
    for(int a=0;a+1<N_ANT;a++){ acc=cma(Cn,Yn,acc);
        Cn=sC[a+1][sc_l]; Yn=Y[sc*N_SYM*N_ANT+sym*N_ANT+a+1]; }
    acc=cma(Cn,Yn,acc);
    O[sc*N_LAYERS*N_SYM+layer*N_SYM+sym]=acc;
}
__global__ void v2(const cf32* C,const cf32* Y,cf32* O,int nSC){
    constexpr int AP=((N_ANT+WK-1)/WK)*WK;
    int sc=blockIdx.x,lane=threadIdx.x; if(sc>=nSC) return;
    __shared__ __half Ar[WM][AP],Ai[WM][AP],Br[AP][WN],Bi[AP][WN];
    for(int i=lane;i<WM*AP;i+=32){int r=i/AP,c=i%AP;
        if(r<N_LAYERS&&c<N_ANT){cf32 v=C[sc*N_LAYERS*N_ANT+r*N_ANT+c];
            Ar[r][c]=__float2half(v.re);Ai[r][c]=__float2half(v.im);}
        else{Ar[r][c]=0;Ai[r][c]=0;}}
    for(int i=lane;i<AP*WN;i+=32){int a=i/WN,s=i%WN;
        if(a<N_ANT&&s<N_SYM){cf32 v=Y[sc*N_SYM*N_ANT+s*N_ANT+a];
            Br[a][s]=__float2half(v.re);Bi[a][s]=__float2half(v.im);}
        else{Br[a][s]=0;Bi[a][s]=0;}}
    __syncwarp();
    wmma::fragment<wmma::matrix_a,WM,WN,WK,__half,wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,WM,WN,WK,__half,wmma::row_major> bf;
    wmma::fragment<wmma::accumulator,WM,WN,WK,float> rr,ri,ir,ii;
    wmma::fill_fragment(rr,0);wmma::fill_fragment(ri,0);
    wmma::fill_fragment(ir,0);wmma::fill_fragment(ii,0);
    #pragma unroll
    for(int k=0;k<AP;k+=WK){
        wmma::load_matrix_sync(af,&Ar[0][k],AP);
        wmma::load_matrix_sync(bf,&Br[k][0],WN);wmma::mma_sync(rr,af,bf,rr);
        wmma::load_matrix_sync(bf,&Bi[k][0],WN);wmma::mma_sync(ri,af,bf,ri);
        wmma::load_matrix_sync(af,&Ai[0][k],AP);wmma::mma_sync(ir,af,bf,ir);
        wmma::load_matrix_sync(bf,&Br[k][0],WN);wmma::mma_sync(ii,af,bf,ii);}
    __shared__ float t[4][WM][WN];
    wmma::store_matrix_sync(&t[0][0][0],rr,WN,wmma::mem_row_major);
    wmma::store_matrix_sync(&t[1][0][0],ri,WN,wmma::mem_row_major);
    wmma::store_matrix_sync(&t[2][0][0],ir,WN,wmma::mem_row_major);
    wmma::store_matrix_sync(&t[3][0][0],ii,WN,wmma::mem_row_major);
    __syncwarp();
    for(int i=lane;i<WM*WN;i+=32){int l=i/WN,s=i%WN;
        if(l>=N_LAYERS||s>=N_SYM) continue;
        O[sc*N_LAYERS*N_SYM+l*N_SYM+s]={t[0][l][s]-t[2][l][s],t[1][l][s]+t[3][l][s]};}
}

double pct(std::vector<float>& v,double p){std::sort(v.begin(),v.end());
    size_t i=std::min(v.size()-1,(size_t)std::ceil(p/100*v.size())-1);return v[i];}

int main(){
    cudaDeviceProp p;cudaGetDeviceProperties(&p,0);
    int nSC=N_PRB*N_SC_PRB;
    printf("GPU %s  cfg: %dant %dRB %dL %dsym  nSC=%d\n",p.name,N_ANT,N_PRB,N_LAYERS,N_SYM,nSC);
    size_t szC=(size_t)nSC*N_LAYERS*N_ANT*sizeof(cf32);
    size_t szY=(size_t)nSC*N_SYM*N_ANT*sizeof(cf32);
    size_t szO=(size_t)nSC*N_LAYERS*N_SYM*sizeof(cf32);
    cf32 *hC=(cf32*)malloc(szC),*hY=(cf32*)malloc(szY),*h1=(cf32*)malloc(szO),*h2=(cf32*)malloc(szO);
    srand(7);
    auto rf=[]{return ((float)rand()/RAND_MAX-0.5f)*1.4f;};
    for(size_t i=0;i<szC/8;i++)hC[i]={rf(),rf()};
    for(size_t i=0;i<szY/8;i++)hY[i]={rf(),rf()};
    cf32 *dC,*dY,*d1,*d2;
    CK(cudaMalloc(&dC,szC));CK(cudaMalloc(&dY,szY));CK(cudaMalloc(&d1,szO));CK(cudaMalloc(&d2,szO));
    CK(cudaMemcpy(dC,hC,szC,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dY,hY,szY,cudaMemcpyHostToDevice));
    dim3 b1(N_SC_PRB,N_SYM),g1(N_PRB,N_LAYERS);
    dim3 b2(32),g2(nSC);
    for(int i=0;i<WARMUP;i++){v1<<<g1,b1>>>(dC,dY,d1,nSC);v2<<<g2,b2>>>(dC,dY,d2,nSC);}
    CK(cudaDeviceSynchronize());
    std::vector<float> t1,t2; cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);
    for(int i=0;i<ITERS;i++){float ms;
        cudaEventRecord(s);v1<<<g1,b1>>>(dC,dY,d1,nSC);cudaEventRecord(e);cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms,s,e);t1.push_back(ms);
        cudaEventRecord(s);v2<<<g2,b2>>>(dC,dY,d2,nSC);cudaEventRecord(e);cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms,s,e);t2.push_back(ms);}
    CK(cudaMemcpy(h1,d1,szO,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h2,d2,szO,cudaMemcpyDeviceToHost));
    // distribution
    auto stat=[&](std::vector<float> v,const char* n){
        double m=0;for(float x:v)m+=x;m/=v.size();double sd=0;for(float x:v)sd+=(x-m)*(x-m);sd=sqrt(sd/v.size());
        std::vector<float> w=v;
        printf("%s  mean=%.4f sd=%.4f  p5=%.4f p50=%.4f p95=%.4f ms\n",
            n,m,sd,pct(w,5),pct(w,50),pct(w,95));
        return m;};
    double m1=stat(t1,"V1 scalar"); double m2=stat(t2,"V2 WMMA  ");
    std::vector<float> r=t1,q=t2;
    printf("speedup(median) = %.3fx   speedup(mean) = %.3fx\n",pct(r,50)/pct(q,50),m1/m2);
    // EVM and error
    double num=0,den=0,maxre=0;long N=szO/sizeof(cf32);
    for(long i=0;i<N;i++){double er=h1[i].re-h2[i].re, ei=h1[i].im-h2[i].im;
        double e2=er*er+ei*ei, ref2=(double)h1[i].re*h1[i].re+(double)h1[i].im*h1[i].im;
        num+=e2; den+=ref2; double rel=ref2>1e-9?sqrt(e2/ref2):0; if(rel>maxre)maxre=rel;}
    double evm=sqrt(num/den);
    printf("EVM(V2 vs V1) = %.4f%%  (%.1f dB)   max_rel_err = %.4f%%\n",
        evm*100, 20*log10(evm), maxre*100);
    // roofline inputs (V2): bytes = read C,Y once + write O; FLOPs = 8*nSC*L*sym*ant
    double flops=8.0*nSC*N_LAYERS*N_SYM*N_ANT;
    double bytesC=szC, bytesY=szY, bytesO=szO; double bytes=bytesC+bytesY+bytesO;
    printf("roofline: FLOPs=%.3e  min_bytes=%.3e (C+Y+O)  AI=%.2f FLOP/byte\n",
        flops, bytes, flops/bytes);
    printf("achieved(V2): %.1f GFLOP/s  effBW(min-traffic)=%.1f GB/s\n",
        flops/(m2/1e3)/1e9, bytes/(m2/1e3)/1e9);
    return 0;
}
