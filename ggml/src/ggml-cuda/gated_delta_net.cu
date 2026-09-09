#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     const int32_t * state_idx,
                                     int64_t       state_row_stride,
                                     int64_t       n_state_rows,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     float         qk_norm_eps) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // Ordinary mode consumes a packed [S_v,S_v,H,n_seqs] state tensor. Indexed mode
    // reads one persistent [D] cache row selected at runtime, avoiding a GET_ROWS copy.
    int64_t state_in_offset;
    if (state_idx != nullptr) {
        const int32_t row = state_idx[sequence];
        if (row < 0 || row >= n_state_rows) {
            return;
        }
        state_in_offset = (int64_t) row * state_row_stride + h_idx * S_v * S_v;
    } else {
        state_in_offset = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    }
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if (qk_norm_eps >= 0.0f) {
            float q_ss = 0.0f;
            float k_ss = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; ++r) {
                q_ss += q_reg[r] * q_reg[r];
                k_ss += k_reg[r] * k_reg[r];
            }
            q_ss = warp_reduce_sum<warp_size>(q_ss);
            k_ss = warp_reduce_sum<warp_size>(k_ss);
            const float inv_n = 1.0f / (float) S_v;
            const float post  = rsqrtf((float) S_v);
            const float q_scale = rsqrtf(q_ss * inv_n + qk_norm_eps * inv_n) * post;
            const float k_scale = rsqrtf(k_ss * inv_n + qk_norm_eps * inv_n) * post;
#pragma unroll
            for (int r = 0; r < rows_per_lane; ++r) {
                q_reg[r] *= q_scale;
                k_reg[r] *= k_scale;
            }
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
template <bool keep_rs_t>
__global__ void __launch_bounds__(ggml_cuda_get_physical_warp_size() * 4, 2)
gated_delta_net_cuda_128x4_volta(const float * q,
                                 const float * k,
                                 const float * v,
                                 const float * g,
                                 const float * beta,
                                 const float * curr_state,
                                 float *       dst,
                                 float *       state,
                                 int64_t       H,
                                 int64_t       n_tokens,
                                 int64_t       n_seqs,
                                 int64_t       sq1,
                                 int64_t       sq2,
                                 int64_t       sq3,
                                 int64_t       sv1,
                                 int64_t       sv2,
                                 int64_t       sv3,
                                 int64_t       sb1,
                                 int64_t       sb2,
                                 int64_t       sb3,
                                 const uint3   neqk1_magic,
                                 const uint3   rq3_magic,
                                 float         scale,
                                 int64_t       state_slot_stride,
                                 int           K,
                                 float         qk_norm_eps) {
    (void) qk_norm_eps;
    constexpr int S_v = 128;
    constexpr int cols_per_warp = 4;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(warp_size == 32, "Volta path requires 32-thread warps");
    constexpr int rows_per_lane = S_v / warp_size;

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int lane = threadIdx.x;
    const int warp_col_base = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t state_in_offset  = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset;
    dst += (sequence * n_tokens * H + h_idx) * S_v;

    // Reuse q/k/g/beta across four independent state columns.
    float s_shard[cols_per_warp][rows_per_lane];

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int c = 0; c < cols_per_warp; ++c) {
        const int col = warp_col_base + c;
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int i = r * warp_size + lane;
            s_shard[c][r] = curr_state[col * S_v + i];
        }
    }

    for (int t = 0; t < n_tokens; ++t) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float beta_val = beta[gb_offset];
        const float g_val    = expf(g[gb_offset]);

        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        float kv_shard[cols_per_warp] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
#pragma unroll
            for (int c = 0; c < cols_per_warp; ++c) {
                kv_shard[c] += s_shard[c][r] * k_reg[r];
            }
        }

        const float2 kv01 = warp_reduce_sum<warp_size>(make_float2(kv_shard[0], kv_shard[1]));
        const float2 kv23 = warp_reduce_sum<warp_size>(make_float2(kv_shard[2], kv_shard[3]));
        const float kv_col[cols_per_warp] = { kv01.x, kv01.y, kv23.x, kv23.y };

        float delta_col[cols_per_warp];
#pragma unroll
        for (int c = 0; c < cols_per_warp; ++c) {
            const int col = warp_col_base + c;
            delta_col[c] = (v_t[col] - g_val * kv_col[c]) * beta_val;
        }

        float attn_partial[cols_per_warp] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
#pragma unroll
            for (int c = 0; c < cols_per_warp; ++c) {
                s_shard[c][r] = g_val * s_shard[c][r] + k_reg[r] * delta_col[c];
                attn_partial[c] += s_shard[c][r] * q_reg[r];
            }
        }

        const float2 attn01 = warp_reduce_sum<warp_size>(make_float2(attn_partial[0], attn_partial[1]));
        const float2 attn23 = warp_reduce_sum<warp_size>(make_float2(attn_partial[2], attn_partial[3]));
        if (lane == 0) {
            dst[warp_col_base + 0] = attn01.x * scale;
            dst[warp_col_base + 1] = attn01.y * scale;
            dst[warp_col_base + 2] = attn23.x * scale;
            dst[warp_col_base + 3] = attn23.y * scale;
        }

        dst += S_v * H;

        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * snapshot = state + target_slot * state_slot_stride;
#pragma unroll
                for (int c = 0; c < cols_per_warp; ++c) {
                    const int col = warp_col_base + c;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; ++r) {
                        const int i = r * warp_size + lane;
                        snapshot[col * S_v + i] = s_shard[c][r];
                    }
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < cols_per_warp; ++c) {
            const int col = warp_col_base + c;
#pragma unroll
            for (int r = 0; r < rows_per_lane; ++r) {
                const int i = r * warp_size + lane;
                state[col * S_v + i] = s_shard[c][r];
            }
        }
    }
}
#endif

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, float qk_norm_eps,
        const int32_t * state_idx, int64_t state_row_stride, int64_t n_state_rows, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, state_idx, state_row_stride, n_state_rows, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, qk_norm_eps);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, state_idx, state_row_stride, n_state_rows, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, qk_norm_eps);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, state_idx, state_row_stride, n_state_rows, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, qk_norm_eps);
            break;
        }
        case 128: {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            if constexpr (!KDA) {
                const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
                if ((cc == GGML_CUDA_CC_VOLTA || cc == GGML_CUDA_CC_TURING) && n_tokens > 1 && qk_norm_eps < 0.0f && state_idx == nullptr) {
                    dim3 volta_grid(H, n_seqs, 8); // 4 warps * 4 columns = 16 columns/CTA
                    const ggml_cuda_kernel_launch_params volta_params =
                        ggml_cuda_kernel_launch_params(volta_grid, block_dims, 0, stream);
                    ggml_cuda_kernel_launch(gated_delta_net_cuda_128x4_volta<keep_rs_t>, volta_params,
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, qk_norm_eps);
                    break;
                }
            }
#endif
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, state_idx, state_row_stride, n_state_rows, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, qk_norm_eps);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta      = dst->src[4];
    ggml_tensor * src_state     = dst->src[5];
    ggml_tensor * src_state_idx = dst->src[6];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d = (const float *) src_state->data;
    const int32_t * state_idx_d = src_state_idx ? (const int32_t *) src_state_idx->data : nullptr;
    const int64_t state_row_stride = src_state_idx ? (int64_t)(src_state->nb[1] / sizeof(float)) : 0;
    const int64_t n_state_rows = src_state_idx ? src_state->ne[1] : 0;
    float * dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    if (src_state_idx) {
        GGML_ASSERT(src_state_idx->type == GGML_TYPE_I32 && ggml_is_contiguous(src_state_idx));
        GGML_ASSERT(src_state_idx->ne[0] == n_seqs);
        GGML_ASSERT(src_state->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src_state));
        GGML_ASSERT(src_state->ne[0] == S_v*S_v*H);
    } else {
        GGML_ASSERT(ggml_is_contiguous(src_state));
    }

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const float qk_norm_eps = ggml_get_op_params_f32(dst, 1);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, qk_norm_eps, state_idx_d, state_row_stride, n_state_rows, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, qk_norm_eps, state_idx_d, state_row_stride, n_state_rows, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, qk_norm_eps, state_idx_d, state_row_stride, n_state_rows, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, qk_norm_eps, state_idx_d, state_row_stride, n_state_rows, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
