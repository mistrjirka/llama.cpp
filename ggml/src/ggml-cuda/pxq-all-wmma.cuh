#pragma once
#include <mma.h>

template <class POL>
static __global__ __launch_bounds__(256,2) void pxqa_grouped_wmma(
        const uint8_t * w, const char * x, float * out,
        const int * map, const pxq4_tile * tiles, const int * ntiles,
        int K,int M,int T,int U,int AC,size_t wnb2,size_t xnb1,size_t xnb2,size_t dnb1,size_t dnb2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    if(blockIdx.y>=*ntiles)return;
    const pxq4_tile tile=tiles[blockIdx.y];
    const int tid=threadIdx.x, warp=tid>>5;
    constexpr int LD=40;
    __shared__ __align__(32) half sw[64*LD], sx[32*LD];
    __shared__ __align__(32) float so[64*32];
    const int row=tid/4,seg=tid%4;
    const int xt=tid/8,xk=(tid%8)*4;
    const int mapval=xt<tile.count?map[(size_t)tile.expert*T*U+tile.start+xt]:0;
    const char * xv=x+(size_t)(mapval/U)*xnb2+(size_t)((mapval%U)%AC)*xnb1;
    const int kslabs=K/32;
    const uint8_t * panel=w+(size_t)tile.expert*wnb2+(size_t)blockIdx.x*pxqa_panel_stride<POL>(kslabs);
    const float anchor=__half2float(((const half*)panel)[row]);
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc,0.f);
    const int wm=warp%4,wn=warp/4;
    __syncthreads();
    for(int kb=0;kb<kslabs;++kb) {
        const uint8_t * slab=panel+POL::HDR+(size_t)kb*POL::SLAB;
        const uint8_t * q=slab+POL::CODE_OFF+(size_t)row*POL::CODE_BYTES;
#pragma unroll
        for(int b=0;b<8;++b) {
            const int j=seg*8+b;
            sw[row*LD+j]=__float2half_rn(anchor*POL::subscale(slab,row,j)*POL::book(POL::code(q,j)));
        }
#pragma unroll
        for(int b=0;b<4;++b)sx[xt*LD+xk+b]=__float2half_rn(xt<tile.count?((const float*)xv)[kb*32+xk+b]:0.f);
        __syncthreads();
#pragma unroll
        for(int kf=0;kf<2;++kf) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> af;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> bf;
            wmma::load_matrix_sync(af,sw+wm*16*LD+kf*16,LD);
            wmma::load_matrix_sync(bf,sx+wn*16*LD+kf*16,LD);
            wmma::mma_sync(acc,af,bf,acc);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(so+wm*16+wn*16*64,acc,64,wmma::mem_col_major);
    __syncthreads();
    for(int i=tid;i<64*tile.count;i+=256) {
        const int rr=i%64,tt=i/64;
        const int flat=map[(size_t)tile.expert*T*U+tile.start+tt];
        float * dst=(float *)((char*)out+(size_t)(flat/U)*dnb2+(size_t)(flat%U)*dnb1);
        dst[blockIdx.x*64+rr]=so[i];
    }
#else
    GGML_UNUSED_VARS(w,x,out,map,tiles,ntiles,K,M,T,U,AC,wnb2,xnb1,xnb2,dnb1,dnb2);
    NO_DEVICE_CODE;
#endif
}

// Dense 2D Volta WMMA path. Same arithmetic as the grouped kernel, without expert routing.
template <class POL>
static __global__ __launch_bounds__(256,2) void pxqa_dense_wmma(
        const uint8_t * w, const char * x, float * out,
        int K, int M, int N, size_t xnb1, size_t dnb1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    const int tid = threadIdx.x, warp = tid >> 5;
    constexpr int LD = 40;
    __shared__ __align__(32) half  sw[64*LD], sx[32*LD];
    __shared__ __align__(32) float so[64*32];
    const int row = tid/4, seg = tid%4;
    const int xt = tid/8, xk = (tid%8)*4;
    const int token0 = (int)blockIdx.y*32;
    const int token = token0 + xt;
    const char * xv = x + (size_t)(token < N ? token : 0)*xnb1;
    const int kslabs = K/32;
    const uint8_t * panel = w + (size_t)blockIdx.x*pxqa_panel_stride<POL>(kslabs);
    const float anchor = __half2float(((const half *)panel)[row]);
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc,0.f);
    const int wm = warp%4, wn = warp/4;
    __syncthreads();
    for (int kb=0; kb<kslabs; ++kb) {
        const uint8_t * slab = panel + POL::HDR + (size_t)kb*POL::SLAB;
        const uint8_t * q = slab + POL::CODE_OFF + (size_t)row*POL::CODE_BYTES;
#pragma unroll
        for (int b=0; b<8; ++b) {
            const int j=seg*8+b;
            sw[row*LD+j] = __float2half_rn(anchor*POL::subscale(slab,row,j)*POL::book(POL::code(q,j)));
        }
#pragma unroll
        for (int b=0; b<4; ++b) {
            sx[xt*LD+xk+b] = __float2half_rn(token < N ? ((const float *)xv)[kb*32+xk+b] : 0.f);
        }
        __syncthreads();
#pragma unroll
        for (int kf=0; kf<2; ++kf) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> af;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> bf;
            wmma::load_matrix_sync(af,sw+wm*16*LD+kf*16,LD);
            wmma::load_matrix_sync(bf,sx+wn*16*LD+kf*16,LD);
            wmma::mma_sync(acc,af,bf,acc);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(so+wm*16+wn*16*64,acc,64,wmma::mem_col_major);
    __syncthreads();
    const int nvalid = min(32, N-token0);
    for (int i=tid; i<64*nvalid; i+=256) {
        const int rr=i%64, tt=i/64;
        float * dst = (float *)(out + 0); // keep pointer arithmetic explicit below
        ((float *)((char *)dst + (size_t)(token0+tt)*dnb1))[blockIdx.x*64+rr] = so[i];
    }
#else
    GGML_UNUSED_VARS(w,x,out,K,M,N,xnb1,dnb1);
    NO_DEVICE_CODE;
#endif
}
