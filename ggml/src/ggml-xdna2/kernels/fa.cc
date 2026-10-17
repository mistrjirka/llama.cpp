// Flash-attention prefill core kernel (AIE2P), head dim D (-DD=128 or 256, default 256): one core = 16 queries (2 tiles of 8) of one q head, K/V streamed in 16-key blocks.
// Both matmuls in bfp16ebs8 (mac_8x8_8x8T, fp32 accumulate); the softmax works in the log2 domain (the host pre-scales Q by log2(e) * scale).
// Lazy max: the running max m is only raised when a row max exceeds it by more than TH, so P <= 2^TH.  The row sums l are a matmul of the
// same bfp16 P with a ones tile (LT, every column = l), so O and l see identical P and identical rescales; the host divides O by l.
// m only has to be used consistently, so it is kept in bf16 (row max and row broadcast use the bf16 8x8 transpose).
//
// Inputs are host-encoded bfp16ebs8 (72-byte 8x8 tiles = 8 rows of [shared exponent][8 x int8]), in consumption order:
//   Q object  : [dt D/8][qt 2] tiles  [8 q][8 d]          (D * 18 B)
//   K block   : [dt D/8][kt 2] tiles  [8 keys][8 d]       (16 keys; the host's bfp16 shadow of the KV cache)
//   V block   : [dt D/8][kt 2] tiles  [8 d][8 keys]       (V^T)
//   O (f32)   : [qt 2][dt D/8][8 q][8 d] unnormalized, then l: [qt 2][64] (the LT tiles)
// Header (first object of the V stream, i32): 0 nb (blocks in the stream), 1 d0 (block index of the q tile of the qi = 0 cores)
// Identity (per core, i32): 0 me (index of this core's Q in the Q prefix of the K stream), 1 qi (q tile of the pair: 0 or 1)
// Causality: the q tile of a core is block d = d0 + qi; blocks j < d are fully visible, j == d is the diagonal (masked per 8x8 tile),
// j > d is skipped (still consumed: the K/V streams are broadcast to the 8 cores of a kv head).
#include <aie_api/aie.hpp>
#include "aie_kernel_utils.h"
#include <stdint.h>

#ifndef D
#define D 256
#endif
#define NDT (D / 8)
#define OBJB (NDT * 2 * 72)
#define TH 8.0f
#define NEG (-200.0f)                               // masked score / initial max (log2 units); the bf16 exp2 of <= -126 is 0

alignas(64) static int8_t QB[OBJB];                  // bfp16 Q tiles [dt][qt]
alignas(64) static int8_t ONES[72];                  // bfp16 tile of 1.0
alignas(64) static int8_t PB[4 * 72];                // bfp16 P tiles [qt][kt]
alignas(64) static float LT[2][64];                  // row sums, broadcast over the 8 columns
alignas(64) static float TRI[64];                    // 0 on/below the diagonal, NEG above (lane r*8+c, c > r)
alignas(64) static float SF[2][2][64];               // S tiles [qt][kt]
alignas(64) static float MB[2][64], ABC[2][64];      // row-broadcast max, rescale factors
alignas(64) static bfloat16 MX[2][16];               // running max per q tile (8 used)
static int st_q, st_j;
static bool st_fresh;                                // nothing accumulated yet (the first rescale multiplies zeros)

using bvec = aie::block_vector<bfp16ebs8, 64>;
using facc = aie::accum<accfloat, 64>;
using v64f = aie::vector<float, 64>;
using v64b = aie::vector<bfloat16, 64>;
using v16f = aie::vector<float, 16>;
using v16b = aie::vector<bfloat16, 16>;

// 2^x: the bf16 exp2 instruction is a linear approximation (2^f ~ 1 + f, up to 6% off) but exact at integers, so
// 2^x = exp2(n) * poly(f), n = round(x) (fp32 magic-number rounding), f = x - n in [-1/2, 1/2], cubic (Taylor, |err| < 1e-3) in bf16.
// Inputs are >= NEG - max (about -250): the bf16 exp2 underflows to 0 there.  64 lanes: the fp32 ops run on 64-lane accumulators anyway.
static inline facc exp2t(facc x) {
  const v64f magic = aie::broadcast<float, 64>(12582912.0f);         // 1.5 * 2^23
  const facc n = aie::sub(aie::add(x, magic), magic);
  const v64b f = aie::sub(x, n.to_vector<float>()).to_vector<bfloat16>();
  facc a; a.from_vector(aie::broadcast<float, 64>(0.240227f));
  a = aie::mac(a, f, aie::broadcast<bfloat16, 64>(0.0555041f));
  facc b; b.from_vector(aie::broadcast<float, 64>(0.693147f));
  b = aie::mac(b, f, a.to_vector<bfloat16>());
  facc c; c.from_vector(aie::broadcast<float, 64>(1.0f));
  c = aie::mac(c, f, b.to_vector<bfloat16>());
  return aie::mul(c.to_vector<bfloat16>(), aie::exp2<bfloat16>(n.to_vector<float>()));
}
static inline v16f b2f(v16b x) { aie::accum<accfloat, 16> a; a.from_vector(x); return a.to_vector<float>(); }
static inline v16b f2b(v16f x) { aie::accum<accfloat, 16> a; a.from_vector(x); return a.to_vector<bfloat16>(); }
// per-row max of an 8x8 tile (lane r*8+c), in bf16, lanes 0..7 = r
static inline v16b row_max(v64f v) {
  facc a; a.from_vector(v);
  const v64b t = aie::transpose(a.to_vector<bfloat16>(), 8, 8);
  const aie::vector<bfloat16, 32> x = aie::max(t.extract<32>(0), t.extract<32>(1));
  const v16b y = aie::max(x.extract<16>(0), x.extract<16>(1));
  return aie::max(y, aie::shuffle_down_fill(y, y, 8));
}
// lane r*8+c = m[r]   (m: bf16, lanes 0..7)
static inline v64f bcast_rows(v16b m) {
  facc a; a.from_vector(aie::transpose(m.extract<8>(0).grow_replicate<64>(), 8, 8)); return a.to_vector<float>();
}

extern "C" void fa_hdr(const int8_t *__restrict hv, int32_t *__restrict hdr, const int32_t *__restrict id) {
  const int32_t *h = (const int32_t *)hv;
  for (int i = 0; i < 4; i++) hdr[i] = h[i];
  st_q = 0; st_j = 0;
}

// one object of the Q prefix: keep it if it is ours
extern "C" void fa_q(const int8_t *__restrict kq, const int32_t *__restrict id) {
  if (st_q++ != id[0]) return;
  for (int i = 0; i < OBJB; i += 64) aie::store_v(QB + i, aie::load_v<64>(kq + i));
}

extern "C" void fa_init(float *__restrict o) {
  for (int i = 0; i < 2 * NDT * 64; i += 16) aie::store_v(o + i, aie::zeros<float, 16>());
  for (int i = 0; i < 128; i += 16) aie::store_v(&LT[0][0] + i, aie::zeros<float, 16>());
  aie::store_v(&MX[0][0], f2b(aie::broadcast<float, 16>(NEG)));
  aie::store_v(&MX[1][0], f2b(aie::broadcast<float, 16>(NEG)));
  for (int r = 0; r < 8; r++)
    for (int c = 0; c < 8; c++) TRI[r * 8 + c] = c > r ? NEG : 0.0f;
  facc one; one.from_vector(aie::broadcast<bfloat16, 64>(1.0f));
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)ONES);
  out.push(bvec(::to_v64bfp16ebs8(one)));
  st_fresh = true;
}

extern "C" void fa_blk(const int8_t *__restrict k, const int8_t *__restrict v, float *__restrict o, const int32_t *__restrict hdr,
                       const int32_t *__restrict id) {
  const int j = st_j++;
  const int d = hdr[1] + id[1];
  if (j > d) return;
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);

  // S[q][key] = Q K^T (log2 units)
  facc c00 = aie::zeros<accfloat, 64>(), c01 = c00, c10 = c00, c11 = c00;
  {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pq((const bfp16ebs8 *)QB), pk((const bfp16ebs8 *)k);
    for (int dt = 0; dt < NDT; dt++) {
      const bvec q0 = pq.pop(), q1 = pq.pop();
      const bvec k0 = pk.pop(), k1 = pk.pop();
      c00 = mac_8x8_8x8T(q0, k0, c00); c01 = mac_8x8_8x8T(q0, k1, c01);
      c10 = mac_8x8_8x8T(q1, k0, c10); c11 = mac_8x8_8x8T(q1, k1, c11);
    }
  }

  // online softmax per 8-query tile, through static buffers to keep the stack small
  aie::store_v(SF[0][0], c00.to_vector<float>()); aie::store_v(SF[0][1], c01.to_vector<float>());
  aie::store_v(SF[1][0], c10.to_vector<float>()); aie::store_v(SF[1][1], c11.to_vector<float>());
  if (j == d) {                                      // diagonal block: key tile kt vs query tile qt
    aie::store_v(SF[0][0], aie::add(aie::load_v<64>(SF[0][0]), aie::load_v<64>(TRI)));
    aie::store_v(SF[0][1], aie::broadcast<float, 64>(NEG));
    aie::store_v(SF[1][1], aie::add(aie::load_v<64>(SF[1][1]), aie::load_v<64>(TRI)));
  }
  bool resc[2];
  for (int qt = 0; qt < 2; qt++) {
    const v16b mob = aie::load_v<16>(MX[qt]);
    const v16f mo = b2f(mob), m8 = b2f(row_max(aie::max(aie::load_v<64>(SF[qt][0]), aie::load_v<64>(SF[qt][1]))));
    v16b mnb = mob;
    resc[qt] = !aie::gt(m8, aie::add(mo, aie::broadcast<float, 16>(TH))).empty();
    if (resc[qt]) {
      mnb = f2b(aie::max(mo, m8));
      aie::store_v(MX[qt], mnb);
      facc dx; dx.from_vector(bcast_rows(f2b(aie::sub(mo, b2f(mnb)))));   // the differences are bf16-exact (both maxima are bf16)
      aie::store_v(ABC[qt], exp2t(dx).to_vector<float>());
    }
    aie::store_v(MB[qt], bcast_rows(mnb));
  }
  // P = 2^(S - m) per 8x8 tile, straight into the bfp16 P tiles
  {
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)PB);
    for (int t = 0; t < 4; t++) {
      facc x; x.from_vector(aie::load_v<64>(&SF[0][0][0] + t * 64));
      x = aie::sub(x, aie::load_v<64>(MB[t >> 1]));
      out.push(bvec(::to_v64bfp16ebs8(exp2t(x))));
    }
  }

  // rescale (rare, never on the first block), then l += P 1 and O[q][d] += P V
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pps((const bfp16ebs8 *)PB), p1s((const bfp16ebs8 *)ONES);
  const bvec p00 = pps.pop(), p01 = pps.pop(), p10 = pps.pop(), p11 = pps.pop(), ones = p1s.pop();
  for (int qt = 0; qt < 2 && !st_fresh; qt++) {
    if (!resc[qt]) continue;
    const v64f ab = aie::load_v<64>(ABC[qt]);
    float *ob = o + qt * NDT * 64;
    for (int dt = 0; dt < NDT; dt++) aie::store_v(ob + dt * 64, aie::mul(aie::load_v<64>(ob + dt * 64), ab).to_vector<float>());
    aie::store_v(LT[qt], aie::mul(aie::load_v<64>(LT[qt]), ab).to_vector<float>());
  }
  st_fresh = false;
  {
    facc l0(aie::load_v<64>(LT[0])), l1(aie::load_v<64>(LT[1]));
    l0 = mac_8x8_8x8T(p00, ones, l0); l0 = mac_8x8_8x8T(p01, ones, l0);
    l1 = mac_8x8_8x8T(p10, ones, l1); l1 = mac_8x8_8x8T(p11, ones, l1);
    aie::store_v(LT[0], l0.to_vector<float>()); aie::store_v(LT[1], l1.to_vector<float>());
  }
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pv((const bfp16ebs8 *)v);
  AIE_LOOP_UNROLL(2)
  for (int dt = 0; dt < NDT; dt++) {
    const bvec v0 = pv.pop(), v1 = pv.pop();
    float *o0 = o + dt * 64, *o1 = o + (NDT + dt) * 64;
    facc a0(aie::load_v<64>(o0)), a1(aie::load_v<64>(o1));
    a0 = mac_8x8_8x8T(p00, v0, a0); a0 = mac_8x8_8x8T(p01, v1, a0);
    a1 = mac_8x8_8x8T(p10, v0, a1); a1 = mac_8x8_8x8T(p11, v1, a1);
    aie::store_v(o0, a0.to_vector<float>()); aie::store_v(o1, a1.to_vector<float>());
  }
  aie::set_rounding(saved);
}

// the row sums follow O in the output object
extern "C" void fa_end(float *__restrict o) {
  for (int i = 0; i < 128; i += 16) aie::store_v(o + 2 * NDT * 64 + i, aie::load_v<16>(&LT[0][0] + i));
}
