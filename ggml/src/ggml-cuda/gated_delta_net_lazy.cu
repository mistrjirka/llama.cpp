// CUDA implementation of GGML_OP_GATED_DELTA_NET_LAZY.
//
// The state is kept as a base plus the rank-1 updates of the last K tokens, so a speculative
// batch can be cut after any of its tokens without writing the whole state per token.
//
// One warp handles one row of the state. The rows are independent: the delta of a row depends
// only on that row and on k, and the update is rank-1 over the same row. The only cross-lane
// work is the two dot products per token (row . k and row . q).
//
// Reference: ggml-cpu/ops.cpp ggml_compute_forward_gated_delta_net_lazy_one_chunk and
// ggml-metal/kernels/gated_delta_net.metal kernel_gated_delta_net_lazy_impl.
#include "gated_delta_net.cuh"

template <int NSG> // elements of a row held by one lane: S_v / WARP_SIZE
static __global__ void gated_delta_net_lazy_f32(
        const float   * __restrict__ q,
        const float   * __restrict__ k,
        const float   * __restrict__ v,
        const float   * __restrict__ g,
        const float   * __restrict__ beta,
        const float   * __restrict__ s,
        const float   * __restrict__ f,
        const int32_t * __restrict__ n_apply,
        float         * __restrict__ dst,
        const int64_t S_v, const int64_t H, const int64_t n_tokens, const int64_t n_seqs,
        const int64_t sq1, const int64_t sq2, const int64_t sq3,
        const int64_t sk1, const int64_t sk2, const int64_t sk3,
        const int64_t sv1, const int64_t sv2, const int64_t sv3,
        const int64_t sb1, const int64_t sb2, const int64_t sb3,
        const int64_t sg1, const int64_t sg2, const int64_t sg3,
        const int64_t ne_q1, const int64_t ne_k1, const int64_t K) {

    const int64_t row = (int64_t) blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    const int64_t i1  = blockIdx.y; // head
    const int64_t i3  = blockIdx.z; // sequence
    if (row >= S_v) {
        return;
    }
    const int64_t lane = threadIdx.x % WARP_SIZE;

    const int64_t iq1 = i1 % ne_q1;
    const int64_t ik1 = i1 % ne_k1;

    const float scale = 1.0f/sqrtf((float) S_v);

    // the factors of one token: d [S_v, H], k [S_v, H], exp(g) [H]
    const int64_t TB = H*(2*S_v + 1);

    // the state is stored transposed: M[row][i] = S[i][row], so a row is contiguous
    const float * s_ptr = s + (i3*H + i1)*S_v*S_v + row*S_v;

    float ls[NSG];
#pragma unroll
    for (int j = 0; j < NSG; ++j) {
        ls[j] = s_ptr[lane + j*WARP_SIZE];
    }

    // the stored updates that belong to the base: the same two steps as below, with d as stored
    const int64_t n_rep = MIN((int64_t) n_apply[i3], K);
    {
        const float * f_ptr = f + i3*K*TB;
        for (int64_t t = 0; t < n_rep; ++t) {
            const float   g_f = f_ptr[2*H*S_v + i1];
            const float   d_f = f_ptr[i1*S_v + row];
            const float * k_f = f_ptr + H*S_v + i1*S_v;
#pragma unroll
            for (int j = 0; j < NSG; ++j) {
                const int64_t i = lane + j*WARP_SIZE;
                ls[j] = fmaf(k_f[i], d_f, ls[j]*g_f);
            }
            f_ptr += TB;
        }
    }

    const int64_t attn_size = S_v*H*n_tokens*n_seqs;
    const int64_t base_size = S_v*S_v*H*n_seqs;

    // the factors are kept while the tokens are no more than K: then the new base is the state
    // the tokens start from, otherwise it is the state after all of them
    const bool keep = n_tokens <= K;

    float * base_out = dst + attn_size + (i3*H + i1)*S_v*S_v + row*S_v;
    if (keep) {
#pragma unroll
        for (int j = 0; j < NSG; ++j) {
            base_out[lane + j*WARP_SIZE] = ls[j];
        }
    }

    float * attn_out = dst + (i3*n_tokens*H + i1)*S_v + row;
    float * fac_out  = dst + attn_size + base_size + i3*n_tokens*TB; // n_keep == n_tokens when kept

    const float * q_ptr = q    + i3*sq3 + iq1*sq1;
    const float * k_ptr = k    + i3*sk3 + ik1*sk1;
    const float * v_ptr = v    + i3*sv3 + i1*sv1;
    const float * b_ptr = beta + i3*sb3 + i1*sb1;
    const float * g_ptr = g    + i3*sg3 + i1*sg1;

    for (int64_t t = 0; t < n_tokens; ++t) {
        const float g_exp = expf(g_ptr[0]);

        const float * qd = q_ptr + t*sq2;
        const float * kd = k_ptr + t*sk2;
        const float * vd = v_ptr + t*sv2;

        float s_k = 0.0f;
#pragma unroll
        for (int j = 0; j < NSG; ++j) {
            const int64_t i = lane + j*WARP_SIZE;
            ls[j] *= g_exp;
            s_k += ls[j]*kd[i];
        }
        s_k = warp_reduce_sum(s_k);

        const float delta = (vd[row] - s_k)*b_ptr[0];

        float y = 0.0f;
#pragma unroll
        for (int j = 0; j < NSG; ++j) {
            const int64_t i = lane + j*WARP_SIZE;
            ls[j] = fmaf(kd[i], delta, ls[j]);
            y += ls[j]*qd[i];
        }
        y = warp_reduce_sum(y);

        if (lane == 0) {
            attn_out[t*H*S_v] = y*scale;
        }

        if (keep) {
            if (lane == 0) {
                fac_out[i1*S_v + row] = delta;
            }
            // the k of the token and exp(g), written once per head
            if (row == 0) {
                float * fk = fac_out + H*S_v + i1*S_v;
#pragma unroll
                for (int j = 0; j < NSG; ++j) {
                    fk[lane + j*WARP_SIZE] = kd[lane + j*WARP_SIZE];
                }
                if (lane == 0) {
                    fac_out[2*H*S_v + i1] = g_exp;
                }
            }
            fac_out += TB;
        }

        b_ptr += sb2;
        g_ptr += sg2;
    }

    if (!keep) {
#pragma unroll
        for (int j = 0; j < NSG; ++j) {
            base_out[lane + j*WARP_SIZE] = ls[j];
        }
    }
}

void ggml_cuda_op_gated_delta_net_lazy(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src_q       = dst->src[0];
    const ggml_tensor * src_k       = dst->src[1];
    const ggml_tensor * src_v       = dst->src[2];
    const ggml_tensor * src_g       = dst->src[3];
    const ggml_tensor * src_beta    = dst->src[4];
    const ggml_tensor * src_state   = dst->src[5];
    const ggml_tensor * src_factors = dst->src[6];
    const ggml_tensor * src_n_apply = dst->src[7];

    GGML_ASSERT(src_q->type == GGML_TYPE_F32 && src_k->type == GGML_TYPE_F32 && src_v->type == GGML_TYPE_F32);
    GGML_ASSERT(src_g->type == GGML_TYPE_F32 && src_beta->type == GGML_TYPE_F32 && src_state->type == GGML_TYPE_F32);
    GGML_ASSERT(src_factors->type == GGML_TYPE_F32 && src_n_apply->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src_state) && ggml_is_contiguous(src_factors));
    GGML_ASSERT(src_g->ne[0] == 1 && src_beta->ne[0] == 1);

    const int64_t S_v      = src_v->ne[0];
    const int64_t H        = src_v->ne[1];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t n_seqs   = src_v->ne[3];
    const int64_t K        = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(S_v % WARP_SIZE == 0 && S_v <= 128 && S_v != 96);
    GGML_ASSERT(src_state->ne[0] == S_v && src_state->ne[1] == S_v && src_state->ne[2] == H);
    GGML_ASSERT(src_factors->ne[0] == H*(2*S_v + 1) && src_factors->ne[1] == K);

    if (n_tokens == 0 || n_seqs == 0) {
        return;
    }

    const int64_t sq1 = src_q->nb[1]/sizeof(float);
    const int64_t sq2 = src_q->nb[2]/sizeof(float);
    const int64_t sq3 = src_q->nb[3]/sizeof(float);
    const int64_t sk1 = src_k->nb[1]/sizeof(float);
    const int64_t sk2 = src_k->nb[2]/sizeof(float);
    const int64_t sk3 = src_k->nb[3]/sizeof(float);
    const int64_t sv1 = src_v->nb[1]/sizeof(float);
    const int64_t sv2 = src_v->nb[2]/sizeof(float);
    const int64_t sv3 = src_v->nb[3]/sizeof(float);
    const int64_t sb1 = src_beta->nb[1]/sizeof(float);
    const int64_t sb2 = src_beta->nb[2]/sizeof(float);
    const int64_t sb3 = src_beta->nb[3]/sizeof(float);
    const int64_t sg1 = src_g->nb[1]/sizeof(float);
    const int64_t sg2 = src_g->nb[2]/sizeof(float);
    const int64_t sg3 = src_g->nb[3]/sizeof(float);

    // rows of one (head, sequence) are spread over warps; keep the CTA narrow so that a small
    // head count still fills the device
    const int nwarps = 4;
    const dim3 grid((S_v + nwarps - 1)/nwarps, H, n_seqs);
    const dim3 block(WARP_SIZE*nwarps, 1, 1);

    const float   * q_d       = (const float   *) src_q->data;
    const float   * k_d       = (const float   *) src_k->data;
    const float   * v_d       = (const float   *) src_v->data;
    const float   * g_d       = (const float   *) src_g->data;
    const float   * b_d       = (const float   *) src_beta->data;
    const float   * s_d       = (const float   *) src_state->data;
    const float   * f_d       = (const float   *) src_factors->data;
    const int32_t * n_apply_d = (const int32_t *) src_n_apply->data;
    float         * dst_d     = (float         *) dst->data;

    switch (S_v/WARP_SIZE) {
        case 1:
            gated_delta_net_lazy_f32<1><<<grid, block, 0, ctx.stream()>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, f_d, n_apply_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sk1, sk2, sk3, sv1, sv2, sv3,
                sb1, sb2, sb3, sg1, sg2, sg3, src_q->ne[1], src_k->ne[1], K);
            break;
        case 2:
            gated_delta_net_lazy_f32<2><<<grid, block, 0, ctx.stream()>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, f_d, n_apply_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sk1, sk2, sk3, sv1, sv2, sv3,
                sb1, sb2, sb3, sg1, sg2, sg3, src_q->ne[1], src_k->ne[1], K);
            break;
        case 4:
            gated_delta_net_lazy_f32<4><<<grid, block, 0, ctx.stream()>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, f_d, n_apply_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sk1, sk2, sk3, sv1, sv2, sv3,
                sb1, sb2, sb3, sg1, sg2, sg3, src_q->ne[1], src_k->ne[1], K);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}
