// CUDA kernels for the SenseNova-Loop mHC router (GGML_OP_MHC_ROUTE / GGML_OP_MHC_MERGE).
// Mirrors ggml-cpu/ops.cpp ggml_compute_forward_mhc_{route,merge} exactly (same math, same
// 20 sinkhorn passes, same eps). Without these, 192 CPU<->GPU hops per token on CUDA builds.
#include "mhc.cuh"

#define MHC_ROUTE_BLOCK 256

// one block per (token, stream). threads stride over d; 9 reductions (sum-sq + 8 logits).
static __global__ void mhc_route_f32(
        const char * __restrict__ a, const char * __restrict__ b,
        const float * __restrict__ w, const float * __restrict__ bias, float * __restrict__ dst,
        const int64_t d, const int64_t n_tok,
        const size_t nb_a1, const size_t nb_a2, const size_t nb_b1, const size_t nb_b2) {
    const int64_t t = blockIdx.x;
    const int64_t s = blockIdx.y;
    const int tid = threadIdx.x;

    const float * va = (const float *) (a + t*nb_a1 + s*nb_a2);
    const float * vb = (const float *) (b + t*nb_b1 + s*nb_b2);
    float       * po = dst + t*(d + 6) + s*n_tok*(d + 6);

    float acc[9];
#pragma unroll
    for (int k = 0; k < 9; ++k) acc[k] = 0.0f;

    for (int64_t i = tid; i < d; i += blockDim.x) {
        const float xa = va[i];
        const float xb = vb[i];
        acc[0] += xa*xa + xb*xb;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            acc[k + 1] += w[k*2*d + i]*xa + w[k*2*d + d + i]*xb;
        }
    }

    __shared__ float red[9][MHC_ROUTE_BLOCK / WARP_SIZE];
    __shared__ float gates[4];
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
#pragma unroll
    for (int k = 0; k < 9; ++k) {
        float v = warp_reduce_sum(acc[k]);
        if (lane == 0) red[k][warp] = v;
    }
    __syncthreads();

    if (tid == 0) {
        float tot[9];
        const int nw = blockDim.x / WARP_SIZE;
        for (int k = 0; k < 9; ++k) {
            float v = 0.0f;
            for (int j = 0; j < nw; ++j) v += red[k][j];
            tot[k] = v;
        }
        const float rms = 1.0f/sqrtf(tot[0]/(2.0f*d) + 1e-6f);
        float lg[8];
        for (int k = 0; k < 8; ++k) lg[k] = tot[k + 1]*rms + bias[k];

        gates[0] = 1.0f/(1.0f + expf(-lg[0]));
        gates[1] = 1.0f/(1.0f + expf(-lg[1]));
        po[d + 0] = 2.0f/(1.0f + expf(-lg[2]));
        po[d + 1] = 2.0f/(1.0f + expf(-lg[3]));

        float m[2][2] = { { lg[4], lg[5] }, { lg[6], lg[7] } };
        for (int it = 0; it < 20; ++it) {
            for (int r = 0; r < 2; ++r) {
                const float mx  = fmaxf(m[r][0], m[r][1]);
                const float lse = mx + logf(expf(m[r][0] - mx) + expf(m[r][1] - mx));
                m[r][0] -= lse; m[r][1] -= lse;
            }
            for (int c = 0; c < 2; ++c) {
                const float mx  = fmaxf(m[0][c], m[1][c]);
                const float lse = mx + logf(expf(m[0][c] - mx) + expf(m[1][c] - mx));
                m[0][c] -= lse; m[1][c] -= lse;
            }
        }
        po[d + 2] = expf(m[0][0]);
        po[d + 3] = expf(m[0][1]);
        po[d + 4] = expf(m[1][0]);
        po[d + 5] = expf(m[1][1]);
    }
    __syncthreads();

    const float pre0 = gates[0];
    const float pre1 = gates[1];
    for (int64_t i = tid; i < d; i += blockDim.x) {
        po[i] = pre0*va[i] + pre1*vb[i];
    }
}

// grid (ceil(d/256), n_tok, n_str)
static __global__ void mhc_merge_f32(
        const char * __restrict__ a, const char * __restrict__ b, const char * __restrict__ x,
        const char * __restrict__ packed, float * __restrict__ dst,
        const int64_t d, const int64_t n_tok,
        const size_t nb_a1, const size_t nb_a2, const size_t nb_b1, const size_t nb_b2,
        const size_t nb_x1, const size_t nb_x2, const size_t nb_p1, const size_t nb_p2) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t t = blockIdx.y;
    const int64_t s = blockIdx.z;
    if (i >= d) return;

    const float * va = (const float *) (a + t*nb_a1 + s*nb_a2);
    const float * vb = (const float *) (b + t*nb_b1 + s*nb_b2);
    const float * vx = (const float *) (x + t*nb_x1 + s*nb_x2);
    const float * vp = (const float *) (packed + t*nb_p1 + s*nb_p2);
    float       * vo = dst + t*2*d + s*n_tok*2*d;

    const float vav = va[i], vbv = vb[i], vxv = vx[i];
    vo[i]     = vp[d + 2]*vav + vp[d + 3]*vbv + vp[d + 0]*vxv;
    vo[d + i] = vp[d + 4]*vav + vp[d + 5]*vbv + vp[d + 1]*vxv;
}

void ggml_cuda_op_mhc_route(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a    = dst->src[0];
    const ggml_tensor * b    = dst->src[1];
    const ggml_tensor * w    = dst->src[2];
    const ggml_tensor * bias = dst->src[3];
    GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(w) && ggml_is_contiguous(bias) && ggml_is_contiguous(dst));
    GGML_ASSERT(a->nb[0] == sizeof(float) && b->nb[0] == sizeof(float));
    const int64_t d = a->ne[0], n_tok = a->ne[1], n_str = a->ne[2];
    GGML_ASSERT(w->ne[0] == 2*d && w->ne[1] == 8);
    if (n_tok == 0 || n_str == 0) return;
    const dim3 grid(n_tok, n_str, 1);
    mhc_route_f32<<<grid, MHC_ROUTE_BLOCK, 0, ctx.stream()>>>(
        (const char *) a->data, (const char *) b->data, (const float *) w->data, (const float *) bias->data,
        (float *) dst->data, d, n_tok, a->nb[1], a->nb[2], b->nb[1], b->nb[2]);
}

void ggml_cuda_op_mhc_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a      = dst->src[0];
    const ggml_tensor * b      = dst->src[1];
    const ggml_tensor * x      = dst->src[2];
    const ggml_tensor * packed = dst->src[3];
    GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 && packed->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(a->nb[0] == sizeof(float) && b->nb[0] == sizeof(float) && x->nb[0] == sizeof(float) && packed->nb[0] == sizeof(float));
    const int64_t d = a->ne[0], n_tok = a->ne[1], n_str = a->ne[2];
    if (n_tok == 0 || n_str == 0) return;
    const dim3 grid((d + 255)/256, n_tok, n_str);
    mhc_merge_f32<<<grid, 256, 0, ctx.stream()>>>(
        (const char *) a->data, (const char *) b->data, (const char *) x->data, (const char *) packed->data,
        (float *) dst->data, d, n_tok,
        a->nb[1], a->nb[2], b->nb[1], b->nb[2], x->nb[1], x->nb[2], packed->nb[1], packed->nb[2]);
}
