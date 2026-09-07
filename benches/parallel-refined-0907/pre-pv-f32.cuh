#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

// Experimental sm_70 GQA8 direct-Q8 attention for 1..16 packed queries.
// Uses FP32 split accumulators; target/draft KV remains Q8. Not enabled by default.
// Persistent KV remains ordinary llama.cpp q8_0. K/V are widened only while loading each
// 16-key tile into FP16 shared memory, then consumed by Volta mma.sync.m8n8k4.
// The design follows the NInfer-V100 small-T Volta topology, adapted to llama.cpp's q8_0
// block layout and FLASH_ATTN_EXT tensor/mask strides.
namespace ggml_q8refined {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
// --- Fragment addressing -------------------------------------------------------------

// Q / P / PV-output: tile<32,4,half2 or float,I_MAJOR>. l in [0,4) for half2, [0,8) unused here.
__device__ __forceinline__ int volta_qp_get_i() { return threadIdx.x & 31; }
__device__ __forceinline__ int volta_qp_get_j(int l) { return l; }

// K: tile<8,4,half2,I_MAJOR_MIRRORED>.
__device__ __forceinline__ int volta_k_get_i() {
    const int lane = threadIdx.x & 31;
    return ((lane / 16) * 4) + (lane % 4);
}
__device__ __forceinline__ int volta_k_get_j(int l) { return l; }

// D (QK^T output float accumulator): tile<32,8,float,I_MAJOR>. l in [0,8).
__device__ __forceinline__ int volta_d_get_i(int l) {
    const int lane = threadIdx.x & 31;
    return (l & 2) + (lane & ~2);
}
__device__ __forceinline__ int volta_d_get_j(int l) {
    const int lane = threadIdx.x & 31;
    return (lane & 2) + (l & (4 + 1));
}

// V: tile<8,4,half2,J_MAJOR_MIRRORED>. l in [0,4).
__device__ __forceinline__ int volta_v_get_i(int l) {
    const int lane = threadIdx.x & 31;
    return ((l / 2) * 4) + (lane % 4);
}
__device__ __forceinline__ int volta_v_get_j(int l) {
    const int lane = threadIdx.x & 31;
    return ((lane / 16) * 2) + (l % 2);
}

// --- Loads (plain addressed loads -- Volta has no ldmatrix, sm_75+ only) -------------------

// Loads a full 4-half2 row for Q or P-shaped tiles (get_j(l)=l for all l, so one row covers
// all 4 registers). `stride` is in half2 units.
__device__ __forceinline__ void volta_load_qp(half2 (&dst)[4], const half2* __restrict__ base,
                                              int stride) {
    const int row = volta_qp_get_i();
#pragma unroll
    for (int l = 0; l < 4; ++l) { dst[l] = base[row * stride + l]; }
}

__device__ __forceinline__ void volta_load_k(half2 (&dst)[4], const half2* __restrict__ base,
                                             int stride) {
    const int row = volta_k_get_i();
#pragma unroll
    for (int l = 0; l < 4; ++l) { dst[l] = base[row * stride + l]; }
}

__device__ __forceinline__ void volta_load_v(half2 (&dst)[4], const half2* __restrict__ base,
                                             int stride) {
#pragma unroll
    for (int l = 0; l < 4; ++l) { dst[l] = base[volta_v_get_i(l) * stride + volta_v_get_j(l)]; }
}

// --- mma.sync.m8n8k4 wrappers ---------------------------------------------------------------

// QK^T: D[32x8 float] += Q[32x8 half, as A] @ K[8x8 half, as B]^T. K real k-dim per call = 8.
__device__ __forceinline__ void volta_mma_qk(float (&d)[8], const half2 (&q)[4],
                                             const half2 (&k)[4]) {
    const int* Axi = reinterpret_cast<const int*>(q);
    const int* Bxi = reinterpret_cast<const int*>(k);
    int* Dxi        = reinterpret_cast<int*>(d);
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3]), "+r"(Dxi[4]),
                   "+r"(Dxi[5]), "+r"(Dxi[6]), "+r"(Dxi[7])
                 : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]), "r"(Bxi[1]));
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3]), "+r"(Dxi[4]),
                   "+r"(Dxi[5]), "+r"(Dxi[6]), "+r"(Dxi[7])
                 : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[2]), "r"(Bxi[3]));
}

// Raw-operand form of the same instruction, for the quadpair-split-N mapping (see
// q4_volta_qpn_gemm.cuh). volta_mma_qk above passes one A and one B fragment that every quadpair
// shares; the QPN mapping gives each quadpair its own B and its own accumulator half, so its
// operands are per-quadpair registers rather than warp-wide fragments and cannot go through the
// fragment-array signature. Identical instruction, one slice (4 real k) per call.
__device__ __forceinline__ void volta_mma_qp_n(float (&d)[8], unsigned a0, unsigned a1,
                                               unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
                   "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// PV: PV[32x8 half, as output] += P[32x8 half, as A] @ V[8x8 half, as B]. Real k per call = 8.
__device__ __forceinline__ void volta_mma_pv(half2 (&pv)[4], const half2 (&p)[4],
                                             const half2 (&v)[4]) {
    const int* Pxi = reinterpret_cast<const int*>(p);
    const int* Vxi = reinterpret_cast<const int*>(v);
    int* PVxi        = reinterpret_cast<int*>(pv);
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 "
                 "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3};"
                 : "+r"(PVxi[0]), "+r"(PVxi[1]), "+r"(PVxi[2]), "+r"(PVxi[3])
                 : "r"(Pxi[0]), "r"(Pxi[1]), "r"(Vxi[0]), "r"(Vxi[1]));
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 "
                 "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3};"
                 : "+r"(PVxi[0]), "+r"(PVxi[1]), "+r"(PVxi[2]), "+r"(PVxi[3])
                 : "r"(Pxi[2]), "r"(Pxi[3]), "r"(Vxi[2]), "r"(Vxi[3]));
}

// Convert the QK^T float D-tile (post-softmax, i.e. P) directly into the half2 layout the PV
// mma's "A" operand needs -- register-only, one warp shuffle, no shared memory (Volta-
// specific; Turing+ needs an actual transpose here, Volta doesn't -- see docs/v100.md).
// Preserve the original lane permutation without any lane-dependent array
// indexing. The original p[swap_idx] forces a thread-local array on nvcc.
__device__ __forceinline__ void volta_softmax_to_half2(half2 (&p)[4], const float (&d)[8]) {
    const bool hi = (threadIdx.x & 2) != 0;
#pragma unroll
    for (int base = 0; base < 4; base += 2) {
        const half2 a = __floats2half2_rn(d[2*base], d[2*base+1]);
        const half2 b = __floats2half2_rn(d[2*base+2], d[2*base+3]);
        const half2 crossed = __shfl_xor_sync(0xffffffffu, hi ? a : b, 2, 32);
        p[base] = hi ? crossed : a;
        p[base+1] = hi ? b : crossed;
    }
}


#endif

constexpr int D = 256;
constexpr int QH = 16;
constexpr int KVH = 2;
constexpr int G = 8;
constexpr int T = 16; // maximum packed queries; kernel instances use 4, 8 or 16
constexpr int BC = 16;
constexpr int D_SLICE = 64;
constexpr int SMEM_STRIDE = 264;
constexpr float LOG2E = 1.4426950408889634074f;
constexpr int LONG_SPLITS = 160;

static inline int split_count(int n_kv) {
    const int tiles = (n_kv + BC - 1) / BC;
    const char * split_env = std::getenv("GGML_CUDA_VOLTA_Q8_MULTI_SPLITS");
    const int limit = split_env ? std::clamp(std::atoi(split_env), 1, 512) : LONG_SPLITS;
    const int requested = std::min(limit, tiles);
    const int tiles_per_split = (tiles + requested - 1) / requested;
    // Return only non-empty partitions. For example, 101120 keys = 6320 Bc16 tiles; a
    // nominal 160-way split uses 40 tiles/partition and therefore has 158 real partitions.
    return (tiles + tiles_per_split - 1) / tiles_per_split;
}

struct extra_data { uintptr_t pacc=0, pm=0, pl=0, end=0; int splits=0; };

static inline extra_data get_extra(const ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    extra_data e{};
    e.splits = split_count((int) K->ne[1]);
    e.end = (uintptr_t) dst->data + ggml_nbytes(dst);
    e.end = GGML_PAD(e.end, 128); e.pacc = e.end;
    e.end += (size_t)e.splits*T*QH*D*sizeof(float);
    e.end = GGML_PAD(e.end, 128); e.pm = e.end;
    e.end += (size_t)e.splits*T*QH*sizeof(float);
    e.end = GGML_PAD(e.end, 128); e.pl = e.end;
    e.end += (size_t)e.splits*T*QH*sizeof(float);
    return e;
}

static inline size_t get_alloc_size(const ggml_tensor * dst) {
    return get_extra(dst).end - (uintptr_t) dst->data;
}

__device__ __forceinline__ int row_qh(int row, int kvh) { return kvh*G + row % G; }
__device__ __forceinline__ int row_tok(int row) { return row/G; }
__device__ __forceinline__ int64_t pacc_idx(int qh,int d,int tok,int split) {
    return d + (int64_t)D*(qh + (int64_t)QH*(tok + (int64_t)T*split));
}
__device__ __forceinline__ int64_t pstat_idx(int qh,int tok,int split) {
    return qh + (int64_t)QH*(tok + (int64_t)T*split);
}

__device__ __forceinline__ int4 deq8(const block_q8_0 & b, int off) {
    const float s = __half2float(b.d);
    half2 h[4];
#pragma unroll
    for (int i=0;i<4;i++) h[i]=__floats2half2_rn((float)b.qs[off+2*i]*s,(float)b.qs[off+2*i+1]*s);
    return *reinterpret_cast<int4 *>(h);
}

template<int NT, bool COOPERATIVE_QK = false>
__launch_bounds__(NT*32,1)
static __global__ void partial_kernel(
        const float * q, const block_q8_0 * K, const block_q8_0 * V, const half * mask,
        int n_kv, int splits, int n_query, float scale,
        int64_t q_s1, int64_t q_s2, int64_t k_s1, int64_t k_s2, int64_t v_s1, int64_t v_s2, int64_t mask_s1,
        float * pacc, float * pm, float * pl) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    extern __shared__ __align__(16) half workspace[];
    half * q_s = workspace;
    half * k_s = q_s + NT*G*SMEM_STRIDE;
    half * v_s = k_s + BC*SMEM_STRIDE;
    half * m_s = v_s + BC*SMEM_STRIDE;
    // Four warps share QK reduction work instead of recomputing it independently.
    float * qk_partial = reinterpret_cast<float *>(m_s + NT*BC);

    const int kvh=blockIdx.x, split=blockIdx.y, tid=threadIdx.x, warp=tid>>5, lane=tid&31;
    const int dim_warp=warp%4, row_offset=(warp/4)*32;
    const int row_count=n_query*G;
    const int logical_tiles=(n_kv+BC-1)/BC;
    const int tiles_per_split=(logical_tiles+splits-1)/splits;
    const int first_tile=split*tiles_per_split*BC;
    const int split_end=min(n_kv, first_tile+tiles_per_split*BC);
    if (first_tile>=split_end) return;
    const int key_blocks=(split_end-first_tile+BC-1)/BC;

    for (int row=warp; row<NT*G; row+=NT) {
        float vals[8]={};
        if (row<row_count) {
            const int qh=row_qh(row,kvh), tok=row_tok(row);
#pragma unroll
            for (int p=0;p<8;p++) vals[p]=q[lane+32*p + q_s1*tok + q_s2*qh];
        }
#pragma unroll
        for (int p=0;p<8;p++) q_s[row*SMEM_STRIDE+lane+32*p]=__float2half(vals[p]);
    }
    __syncthreads();

    float acc[D_SLICE/8][8];
#pragma unroll
    for(int c=0;c<D_SLICE/8;c++) for(int i=0;i<8;i++) acc[c][i]=0.0f;
    float mlo=-CUDART_INF_F,mhi=-CUDART_INF_F,llo=0.0f,lhi=0.0f;

    for (int kb=0; kb<key_blocks; kb++) {
        const int k0=first_tile+kb*BC;
        for (int chunk=tid; chunk<BC*(D/8); chunk+=NT*32) {
            const int key_l=chunk/(D/8), d=(chunk-key_l*(D/8))*8, key=k0+key_l;
            half * kd=&k_s[key_l*SMEM_STRIDE+d], *vd=&v_s[key_l*SMEM_STRIDE+d];
            if (key<split_end) {
                const int bo=d/32, off=d&31;
                *reinterpret_cast<int4*>(kd)=deq8(K[key*k_s1 + kvh*k_s2 + bo],off);
                *reinterpret_cast<int4*>(vd)=deq8(V[key*v_s1 + kvh*v_s2 + bo],off);
            } else { const int4 z=make_int4(0,0,0,0); *reinterpret_cast<int4*>(kd)=z; *reinterpret_cast<int4*>(vd)=z; }
        }
        for (int i=tid; i<NT*BC; i+=NT*32) {
            const int tok=i/BC, col=i%BC, key=k0+col;
            m_s[i]=(key<n_kv && tok<n_query) ? mask[key + mask_s1*tok] : __float2half(-CUDART_INF_F);
        }
        __syncthreads();

#pragma unroll
        for (int sub=0;sub<2;sub++) {
            const int sub_k0=k0+sub*8;
            float ds[8]={};
#pragma unroll 4
            for (int c0=0;c0<(COOPERATIVE_QK ? D/32 : D/8);c0++) {
                const int c = c0 + (COOPERATIVE_QK ? dim_warp*(D/32) : 0);
                half2 qf[4],kf[4];
                volta_load_qp(qf,reinterpret_cast<const half2*>(&q_s[row_offset*SMEM_STRIDE+c*8]),SMEM_STRIDE/2);
                volta_load_k(kf,reinterpret_cast<const half2*>(&k_s[sub*8*SMEM_STRIDE+c*8]),SMEM_STRIDE/2);
                volta_mma_qk(ds,qf,kf);
            }
            if constexpr (COOPERATIVE_QK) {
                // Pairwise ordered reduction uses only half the naive scratch:
                // even warps publish, odd partners add, then everyone sums the pair.
                // At NT=16 this stays below Volta's per-block shared-memory limit.
                const int group = warp/4;
                const int pair_index = group*512 + (dim_warp/2)*256 + lane*8;
                if ((dim_warp & 1) == 0) {
#pragma unroll
                    for (int l=0;l<8;++l) qk_partial[pair_index+l] = ds[l];
                }
                __syncthreads();
                if ((dim_warp & 1) != 0) {
#pragma unroll
                    for (int l=0;l<8;++l) qk_partial[pair_index+l] += ds[l];
                }
                __syncthreads();
#pragma unroll
                for (int l=0;l<8;++l) {
                    ds[l] = qk_partial[group*512+lane*8+l] + qk_partial[group*512+256+lane*8+l];
                }
                __syncthreads();
            }
#pragma unroll
            for(int l=0;l<8;l++) {
                const int row=row_offset+volta_d_get_i(l), col=volta_d_get_j(l), key=sub_k0+col;
                if (row<row_count && key<split_end) {
                    const int tok=row_tok(row); ds[l]=ds[l]*scale+__half2float(m_s[tok*BC+sub*8+col]);
                } else ds[l]=-CUDART_INF_F;
            }
            float bmlo=fmaxf(fmaxf(ds[0],ds[1]),fmaxf(ds[4],ds[5]));
            float bmhi=fmaxf(fmaxf(ds[2],ds[3]),fmaxf(ds[6],ds[7]));
            bmlo=fmaxf(bmlo,__shfl_xor_sync(0xffffffffu,bmlo,2)); bmhi=fmaxf(bmhi,__shfl_xor_sync(0xffffffffu,bmhi,2));
            const float nmlo=fmaxf(mlo,bmlo), nmhi=fmaxf(mhi,bmhi);
            const float alo=mlo==-CUDART_INF_F?0.0f:exp2f((mlo-nmlo)*LOG2E);
            const float ahi=mhi==-CUDART_INF_F?0.0f:exp2f((mhi-nmhi)*LOG2E);
#pragma unroll
            for(int l=0;l<8;l++){const float nm=(l&2)?nmhi:nmlo;ds[l]=(nm>-CUDART_INF_F&&ds[l]>-CUDART_INF_F)?exp2f((ds[l]-nm)*LOG2E):0.0f;}
            float bllo=ds[0]+ds[1]+ds[4]+ds[5],blhi=ds[2]+ds[3]+ds[6]+ds[7];
            bllo+=__shfl_xor_sync(0xffffffffu,bllo,2);blhi+=__shfl_xor_sync(0xffffffffu,blhi,2);
            llo=llo*alo+bllo;lhi=lhi*ahi+blhi;mlo=nmlo;mhi=nmhi;
            const float alpha=(lane&2)?ahi:alo; half2 pp[4]; volta_softmax_to_half2(pp,ds);
#pragma unroll
            for(int c=0;c<D_SLICE/8;c++){half2 vf[4];volta_load_v(vf,reinterpret_cast<const half2*>(&v_s[sub*8*SMEM_STRIDE+dim_warp*D_SLICE+c*8]),SMEM_STRIDE/2);half2 pv[4]={{0,0},{0,0},{0,0},{0,0}};volta_mma_pv(pv,pp,vf);
#pragma unroll
                for(int n=0;n<4;n++){const float2 f=__half22float2(pv[n]);acc[c][2*n]=acc[c][2*n]*alpha+f.x;acc[c][2*n+1]=acc[c][2*n+1]*alpha+f.y;}
            }
        }
        __syncthreads();
    }
    const int row=row_offset+lane; const float ownm=(lane&2)?mhi:mlo, ownl=(lane&2)?lhi:llo;
    if(dim_warp==0&&row<row_count){const int qh=row_qh(row,kvh),tok=row_tok(row);pm[pstat_idx(qh,tok,split)]=ownm;pl[pstat_idx(qh,tok,split)]=ownl;}
    if(row<row_count){const int qh=row_qh(row,kvh),tok=row_tok(row);
#pragma unroll
        for(int c=0;c<D_SLICE/8;c++){const int d=dim_warp*D_SLICE+c*8;float o[8];for(int i=0;i<8;i++)o[i]=acc[c][i];for (int i=0;i<8;i++) pacc[pacc_idx(qh,d+i,tok,split)]=o[i];}
    }
#endif
}

static __global__ void reduce_kernel(const float *pa,const float *pm,const float *pl,int splits,float *out,int64_t o_s1,int64_t o_s2){
    const int qh=blockIdx.x,tok=blockIdx.y,d=threadIdx.x;__shared__ float red[256];
    float lm=-CUDART_INF_F;for(int s=d;s<splits;s+=256)lm=fmaxf(lm,pm[pstat_idx(qh,tok,s)]);red[d]=lm;__syncthreads();
    for(int st=128;st;st>>=1){if(d<st)red[d]=fmaxf(red[d],red[d+st]);__syncthreads();}const float hm=red[0];
    if(hm==-CUDART_INF_F){out[d+o_s1*qh+o_s2*tok]=0.0f;return;}
    float ll=0;for(int s=d;s<splits;s+=256){const float l=pl[pstat_idx(qh,tok,s)];if(l>0)ll+=l*expf(pm[pstat_idx(qh,tok,s)]-hm);}red[d]=ll;__syncthreads();
    for(int st=128;st;st>>=1){if(d<st)red[d]+=red[d+st];__syncthreads();}const float hl=red[0];
    float num=0;for(int s=0;s<splits;s++){const float l=pl[pstat_idx(qh,tok,s)];if(l>0)num+=pa[pacc_idx(qh,d,tok,s)]*expf(pm[pstat_idx(qh,tok,s)]-hm);}
    out[d+o_s1*qh+o_s2*tok]=hl>0?num/hl:0.0f;
}

template<int NT, bool COOPERATIVE_QK = false>
static inline void launch_partial(
        ggml_backend_cuda_context & ctx, const ggml_tensor * Q, const ggml_tensor * K,
        const ggml_tensor * V, const ggml_tensor * M, const extra_data & e, int first, int count, float scale) {
    constexpr size_t shared_bytes = ((NT*G + 2*BC)*SMEM_STRIDE + NT*BC)*sizeof(half) + (COOPERATIVE_QK ? (NT/4)*512*sizeof(float) : 0);
    static bool configured[GGML_CUDA_MAX_DEVICES] = {};
    if (!configured[ctx.device]) {
        CUDA_CHECK(cudaFuncSetAttribute(partial_kernel<NT, COOPERATIVE_QK>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
        configured[ctx.device] = true;
    }
    partial_kernel<NT, COOPERATIVE_QK><<<dim3(KVH,e.splits),NT*32,shared_bytes,ctx.stream()>>>(
        (const float*)Q->data+first*(Q->nb[1]/sizeof(float)),
        (const block_q8_0*)K->data,(const block_q8_0*)V->data,
        (const half*)M->data+first*(M->nb[1]/sizeof(half)),
        (int)K->ne[1],e.splits,count,scale,
        Q->nb[1]/sizeof(float),Q->nb[2]/sizeof(float),
        K->nb[1]/sizeof(block_q8_0),K->nb[2]/sizeof(block_q8_0),
        V->nb[1]/sizeof(block_q8_0),V->nb[2]/sizeof(block_q8_0),M->nb[1]/sizeof(half),
        (float*)e.pacc,(float*)e.pm,(float*)e.pl);
    CUDA_CHECK(cudaGetLastError());
}

static inline void launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor *Q=dst->src[0],*K=dst->src[1],*V=dst->src[2],*M=dst->src[3];
    const extra_data e=get_extra(dst);
    float scale=1.0f;memcpy(&scale,(const float*)dst->op_params,sizeof(float));
    const int64_t o_s1=dst->nb[1]/sizeof(float),o_s2=dst->nb[2]/sizeof(float);
    cudaStream_t stream=ctx.stream();
    const char * pack_env = std::getenv("GGML_CUDA_VOLTA_Q8_MULTI_PACK");
    const int requested = pack_env ? std::atoi(pack_env) : 4;
    const int pack = requested <= 4 ? 4 : requested <= 8 ? 8 : 16;
    for (int first=0; first<Q->ne[1]; first+=pack) {
        const int count=std::min(pack, (int)Q->ne[1]-first);
        if (count <= 4) {
            if (std::getenv("GGML_CUDA_VOLTA_Q8_COOP") != nullptr)
                launch_partial<4,true>(ctx,Q,K,V,M,e,first,count,scale);
            else launch_partial<4,false>(ctx,Q,K,V,M,e,first,count,scale);
        }
        else if (count <= 8) {
            if (std::getenv("GGML_CUDA_VOLTA_Q8_COOP") != nullptr)
                launch_partial<8,true>(ctx,Q,K,V,M,e,first,count,scale);
            else launch_partial<8,false>(ctx,Q,K,V,M,e,first,count,scale);
        } else {
            if (std::getenv("GGML_CUDA_VOLTA_Q8_COOP") != nullptr)
                launch_partial<16,true>(ctx,Q,K,V,M,e,first,count,scale);
            else launch_partial<16,false>(ctx,Q,K,V,M,e,first,count,scale);
        }
        reduce_kernel<<<dim3(QH,count),256,0,stream>>>(
            (const float*)e.pacc,(const float*)e.pm,(const float*)e.pl,e.splits,
            (float*)dst->data+first*o_s2,o_s1,o_s2);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ggml_q8refined
