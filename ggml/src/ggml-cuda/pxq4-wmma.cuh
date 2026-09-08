// Copyright (c) 2026 PXA Network (PXQ4 wire format and frozen decoder tables).
// SPDX-License-Identifier: MIT
// Grouped Volta WMMA with stable, device-side expert maps and direct scatter.
// No host expert loop or full FP16 expert-weight materialization is needed.
#pragma once
#include <mma.h>

struct pxq4_tile { int expert, start, count; };

// Stable compaction over all (token, slot) occurrences, including repeated IDs.
// Each expert owns T*U map entries; normal unique top-k routes use only T of them.
static __global__ void pxq4_build_map(const int32_t * ids, int ids_stride,
        int tokens, int used, int * map, int * counts) {
    const int e=blockIdx.x, tid=threadIdx.x, lane=tid&31, warp=tid>>5;
    __shared__ int wc[8], total;
    if(tid==0)total=0;
    __syncthreads();
    for(int base=0;base<tokens;base+=256) {
        const int token=base+tid;
        int matches=0;
        if(token<tokens)for(int u=0;u<used;++u)matches+=ids[token*ids_stride+u]==e;
        int prefix=matches;
#pragma unroll
        for(int d=1;d<32;d*=2) {
            const int previous=__shfl_up_sync(0xffffffff,prefix,d);
            if(lane>=d)prefix+=previous;
        }
        if(lane==31)wc[warp]=prefix;
        __syncthreads();
        int off=total+prefix-matches;
        for(int w=0;w<warp;++w)off+=wc[w];
        if(matches)for(int u=0;u<used;++u)if(ids[token*ids_stride+u]==e)
            map[(size_t)e*tokens*used+off++]=token*used+u;
        __syncthreads();
        if(tid==0)for(int w=0;w<8;++w)total+=wc[w];
        __syncthreads();
    }
    if(tid==0)counts[e]=total;
}

static __global__ void pxq4_build_tiles(const int * counts, int experts,
        pxq4_tile * tiles, int * ntiles) {
    __shared__ int scan[512];
    int tid=threadIdx.x;
    int n=tid<experts?(counts[tid]+31)/32:0;
    scan[tid]=n;
    __syncthreads();
    for(int d=1;d<blockDim.x;d*=2) {
        int x=tid>=d?scan[tid-d]:0;
        __syncthreads();scan[tid]+=x;__syncthreads();
    }
    if(tid<experts)for(int i=0;i<n;++i)tiles[scan[tid]-n+i]={tid,i*32,min(32,counts[tid]-i*32)};
    if(tid==experts-1)*ntiles=scan[tid];
}

static __global__ __launch_bounds__(256,2) void pxq4_grouped_wmma(
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
    __shared__ float book[16],sub[16];
    if(tid<16){book[tid]=pxq4_port_book[tid];sub[tid]=pxq4_port_sub[tid];}
    const int row=tid/4,seg=tid%4;
    const int xt=tid/8,xk=(tid%8)*4;
    const int mapval=xt<tile.count?map[(size_t)tile.expert*T*U+tile.start+xt]:0;
    const char * xv=x+(size_t)(mapval/U)*xnb2+(size_t)((mapval%U)%AC)*xnb1;
    const uint8_t * panel=w+(size_t)tile.expert*wnb2+(size_t)blockIdx.x*(128u+(size_t)(K/32)*1088u);
    const float anchor=__half2float(((const half*)panel)[row]);
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc,0.f);
    const int wm=warp%4,wn=warp/4;
    __syncthreads();
    for(int kb=0;kb<K/32;++kb) {
        const uint8_t * slab=panel+128+(size_t)kb*1088;
        const int scale=(slab[row]>>(4*(seg/2)))&15;
        const float eff=anchor*sub[scale];
        const uint32_t q=*(const uint32_t *)(slab+64+16*row+4*seg);
#pragma unroll
        for(int b=0;b<4;++b) {
            sw[row*LD+seg*8+2*b]=__float2half_rn(eff*book[(q>>(b*8))&15]);
            sw[row*LD+seg*8+2*b+1]=__float2half_rn(eff*book[(q>>(b*8+4))&15]);
        }
#pragma unroll
        for(int b=0;b<4;++b)sx[xt*LD+xk+b]=__float2half_rn(xt<tile.count?((const float*)xv)[kb*32+xk+b]:0.f);
        __syncthreads();
#pragma unroll
        for(int kf=0;kf<2;++kf) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> b;
            wmma::load_matrix_sync(a,sw+wm*16*LD+kf*16,LD);
            wmma::load_matrix_sync(b,sx+wn*16*LD+kf*16,LD);
            wmma::mma_sync(acc,a,b,acc);
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
