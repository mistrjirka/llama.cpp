// Universal prefill core kernel (AIE2P), layout v2.  64-token tile x (16-row record) per step: expand the record to bfp16ebs8 in L1
// (w = a*q - b), bfp16 mmul against the resident bfp16 activation slab (64 tok x 256 k), fp32 C accumulate in the output object.
//
// Control (see genp.py): the first A object carries the header (64 B):
//   i32 0 nsb (superblocks per core), 1 nbk (K/256), 2 qtype (4,5,6,8), 3 G (groups per core)
// per slab (sb, kb): pf_slab_begin -> number of 1 KB W chunks of the slab; pf_chunk per chunk assembles records in wrec, processes complete records.
#include <aie_api/aie.hpp>
#include "aie_kernel_utils.h"
#include <stdint.h>

#ifndef PF_S
#define PF_S 1024
#endif
#define GSB 5
#define MT 64                       // tokens per tile
#define MB (MT / 8)                 // 8-token blocks
// per-record scale tables, overlaid by qtype (only one record is in flight):
//   Q4_K/Q5_K : SV[j][nb][64] bf16 (2 KB), BV[j][nb][64] bf16 (2 KB)          j = 32-k sub-block, nb = 8-row half
//   Q6_K      : SV6[s][nb][64] bf16 (4 KB)                                    s = 16-k sub-block
//   Q8_0      : SVH[b][nb][64], SVL[b][nb][64] bf16 (2 KB each)               b = 32-k block (d split in two bf16)
alignas(64) static uint8_t TBL[6144];
#define SV  ((bfloat16 (*)[2][64])(TBL))
#define BV  ((bfloat16 (*)[2][64])(TBL + 2048))
#define SV6 ((bfloat16 (*)[2][64])(TBL))
#define SVH ((bfloat16 (*)[2][64])(TBL))
#define SVL ((bfloat16 (*)[2][64])(TBL + 2048))
alignas(64) static bfloat16 SCB[256], MNB[128], ABF[256], BBF[128];

// fp16 (uint16 x16) -> f32 x16 (normal, zero, denormal)
static inline aie::vector<float, 16> h2f(const uint16_t *p) {
  aie::vector<uint16, 16> h = aie::load_v<16>(p);
  aie::vector<uint32, 16> h32 = aie::unpack(h);
  aie::mask<16> isneg = aie::neq(aie::bit_and(h32, aie::broadcast<uint32, 16>(0x8000)), aie::zeros<uint32, 16>());
  aie::vector<uint32, 16> em = aie::bit_and(h32, aie::broadcast<uint32, 16>(0x7FFF));
  aie::vector<uint32, 16> f = aie::add(aie::upshift(em, 13), aie::broadcast<uint32, 16>(0x38000000u));
  aie::mask<16> isz = aie::eq(em, aie::zeros<uint32, 16>());
  f = aie::select(f, aie::zeros<uint32, 16>(), isz);
  aie::vector<float, 16> r = f.cast_to<float>();
  aie::mask<16> isden = aie::lt(em, aie::broadcast<uint32, 16>(0x400));
  aie::vector<float, 16> den = aie::mul(aie::to_float(em.cast_to<int32>()), aie::broadcast<float, 16>(5.9604644775390625e-08f)).to_vector<float>();
  r = aie::select(r, den, isden);
  return aie::select(r, aie::neg(r), isneg);
}

// SV[j][nb] / BV[j][nb] (64 bf16, [n][k] replicated) from ABF / BBF [j][n]
__attribute__((noinline)) static void fill_sv_bv() {
  AIE_LOOP_UNROLL(4)
  for (int i = 0; i < 16; i++) {           // i = j*2 + nb
    aie::vector<bfloat16, 8> a8 = aie::load_v<8>(ABF + i * 8), b8 = aie::load_v<8>(BBF + i * 8);
    aie::vector<bfloat16, 64> av = aie::transpose(a8.grow_replicate<64>(), 8, 8);
    aie::vector<bfloat16, 64> bv = aie::transpose(b8.grow_replicate<64>(), 8, 8);
    aie::store_v(&SV[0][0][0] + i * 64, av);
    aie::store_v(&BV[0][0][0] + i * 64, bv);
  }
}

// ---- scale prep v4: SV[j][nb] = bf16(d*sc_j) replicated [n][k]; BV[j][nb] = bf16(-dmin*m_j) replicated (the expand forms -b - 128*a)
template <int SCO, int DO, int DMO>
static inline void prep_scales_k(const int8_t *__restrict rec) {
  const uint8_t *scp = (const uint8_t *)(rec + SCO);
  aie::vector<uint16, 32> Va = aie::unpack(aie::load_v<32>(scp)), Vb = aie::unpack(aie::load_v<32>(scp + 32)),
                          Vc = aie::unpack(aie::load_v<32>(scp + 64)), Vd = aie::unpack(aie::load_v<32>(scp + 96)),
                          Ve = aie::unpack(aie::load_v<32>(scp + 128)), Vf = aie::unpack(aie::load_v<32>(scp + 160));
  const aie::vector<uint16, 32> m63 = aie::broadcast<uint16, 32>(63), m15 = aie::broadcast<uint16, 32>(15),
                                mF0 = aie::broadcast<uint16, 32>(0xF0), mC0 = aie::broadcast<uint16, 32>(0xC0);
  aie::vector<int16, 32> scp2[4], mnp2[4];
  scp2[0] = aie::bit_and(Va, m63).cast_to<int16>();
  scp2[1] = aie::bit_and(Vb, m63).cast_to<int16>();
  scp2[2] = aie::add(aie::bit_and(Ve, m15), aie::downshift(aie::bit_and(Va, mC0), 2)).cast_to<int16>();
  scp2[3] = aie::add(aie::bit_and(Vf, m15), aie::downshift(aie::bit_and(Vb, mC0), 2)).cast_to<int16>();
  mnp2[0] = aie::bit_and(Vc, m63).cast_to<int16>();
  mnp2[1] = aie::bit_and(Vd, m63).cast_to<int16>();
  mnp2[2] = aie::add(aie::downshift(aie::bit_and(Ve, mF0), 4), aie::downshift(aie::bit_and(Vc, mC0), 2)).cast_to<int16>();
  mnp2[3] = aie::add(aie::downshift(aie::bit_and(Vf, mF0), 4), aie::downshift(aie::bit_and(Vd, mC0), 2)).cast_to<int16>();
  // d, dmin as exact bf16 hi/lo pieces
  aie::vector<float, 16> d_f = h2f((const uint16_t *)(rec + DO));
  aie::vector<float, 16> dmin_f = h2f((const uint16_t *)(rec + DMO));
  aie::accum<accfloat, 16> t, u;
  t.from_vector(d_f); aie::vector<bfloat16, 16> d_hi = t.to_vector<bfloat16>();
  u.from_vector(d_hi);
  t.from_vector(aie::sub(d_f, u.to_vector<float>()));    // exact remainder
  aie::vector<bfloat16, 16> d_lo = t.to_vector<bfloat16>();
  t.from_vector(aie::neg(dmin_f)); aie::vector<bfloat16, 16> n_hi = t.to_vector<bfloat16>();
  u.from_vector(n_hi);
  t.from_vector(aie::sub(aie::neg(dmin_f), u.to_vector<float>()));
  aie::vector<bfloat16, 16> n_lo = t.to_vector<bfloat16>();
  for (int jp = 0; jp < 4; jp++) {
    aie::store_v(SCB + jp * 32, aie::to_float<bfloat16>(scp2[jp]));
    aie::store_v(MNB + jp * 32, aie::to_float<bfloat16>(mnp2[jp]));
  }
  AIE_LOOP_UNROLL_FULL
  for (int j = 0; j < 8; j++) {
    aie::vector<bfloat16, 16> sc16 = aie::load_v<16>(SCB + j * 16), mn16 = aie::load_v<16>(MNB + j * 16);
    aie::accum<accfloat, 16> aa = aie::mul(d_hi, sc16);
    aa = aie::mac(aa, d_lo, sc16);
    aie::vector<bfloat16, 16> a_bf = aa.to_vector<bfloat16>();
    aie::accum<accfloat, 16> bb = aie::mul(n_hi, mn16);
    bb = aie::mac(bb, n_lo, mn16);
    aie::vector<bfloat16, 16> b_bf = bb.to_vector<bfloat16>();
    aie::store_v(ABF + j * 16, a_bf); aie::store_v(BBF + j * 16, b_bf);
  }
  fill_sv_bv();
}

// ---- Q4_0 / Q4_1 scale prep: the Q4_K form w = a*q + b per 32-k block j with a = d, b = -8 d (Q4_0) or m (Q4_1)
__attribute__((noinline)) static void prep_q40(const int8_t *__restrict rec, bool q41) {
  // i < 8: a_j = d_j (fp16 at 2048);  i >= 8: b_j = -8 d_j (Q4_0, from 2048) or m_j (Q4_1, fp16 at 2304).  One fp16 decode (program memory).
  const float sb = q41 ? 1.0f : -8.0f;
  const int ob = q41 ? 2304 : 2048;
  AIE_LOOP_NO_UNROLL
  for (int i = 0; i < 16; i++) {
    const int j = i & 7, hi = i >> 3;
    aie::vector<float, 16> v = h2f((const uint16_t *)(rec + (hi ? ob : 2048) + j * 32));
    v = aie::mul(v, aie::broadcast<float, 16>(hi ? sb : 1.0f)).to_vector<float>();
    aie::accum<accfloat, 16> t;
    t.from_vector(v);
    aie::store_v((hi ? BBF : ABF) + j * 16, t.to_vector<bfloat16>());
  }
  fill_sv_bv();
}

// ---- expand v2: x = bf16(128+q) built by byte-zip with 0x43 (no int->float convert); w = a*x - 128*a - b via 2 macs
#ifndef EXP_STAGE
#define EXP_STAGE 4
#endif
static inline void expand_q4(const int8_t *__restrict rec, int8_t *__restrict Bt) {
  const aie::vector<uint8, 64> c43 = aie::broadcast<uint8, 64>(0x43);
  const aie::vector<bfloat16, 64> m128 = aie::broadcast<bfloat16, 64>(-128.0f);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)Bt);
  aie::vector<uint8, 64> sink = aie::zeros<uint8, 64>();
  for (int j = 0; j < 8; j++) {
    const aie::vector<bfloat16, 64> sv0 = aie::load_v<64>(SV[j][0]), sv1 = aie::load_v<64>(SV[j][1]);
    AIE_LOOP_UNROLL_FULL
    for (int t = 0; t < 4; t++) {                 // (hf, h) -> kb = 4j + t
      aie::vector<uint4, 128> q4 = aie::load_v<128>((const uint4 *)(rec + (2 * j + (t >> 1)) * 128 + (t & 1) * 64));
      aie::vector<uint8, 128> q = q4.unpack();
      auto [e0, e1] = aie::interleave_unzip(q.extract<64>(0), q.extract<64>(1), 8);
      aie::vector<uint8, 64> t0 = aie::transpose(e0, 8, 8), t1 = aie::transpose(e1, 8, 8);
#if EXP_STAGE == 1
      sink = aie::bit_xor(sink, aie::bit_xor(t0, t1));
#else
      auto [z00, z01] = aie::interleave_zip(t0, c43, 1);
      auto [z10, z11] = aie::interleave_zip(t1, c43, 1);
      aie::vector<bfloat16, 64> x0 = aie::concat(z00.cast_to<bfloat16>(), z01.cast_to<bfloat16>());
      aie::vector<bfloat16, 64> x1 = aie::concat(z10.cast_to<bfloat16>(), z11.cast_to<bfloat16>());
#if EXP_STAGE == 2
      sink = aie::bit_xor(sink, aie::bit_xor(x0.cast_to<uint8>().extract<64>(0), x1.cast_to<uint8>().extract<64>(1)));
#else
      aie::accum<accfloat, 64> acc0, acc1;
      acc0.from_vector(aie::load_v<64>(BV[j][0])); acc1.from_vector(aie::load_v<64>(BV[j][1]));
      acc0 = aie::mac(acc0, m128, sv0); acc1 = aie::mac(acc1, m128, sv1);     // -b - 128 a (as the former AC0 table: same op order)
      acc0 = aie::mac(acc0, x0, sv0); acc1 = aie::mac(acc1, x1, sv1);
      out.push(aie::block_vector<bfp16ebs8, 64>(::to_v64bfp16ebs8(acc0)));
      out.push(aie::block_vector<bfp16ebs8, 64>(::to_v64bfp16ebs8(acc1)));
#endif
#endif
    }
  }
  if (EXP_STAGE < 3) aie::store_v((uint8_t *)Bt, sink);
}


// ================= Q5_K (REC 2816): lo nibbles B-tiles [0,2048), hi bits [2048,2560), scale planes 2560, d 2752, dmin 2784 =================
static inline void expand_q5(const int8_t *__restrict rec, int8_t *__restrict Bt) {
  const aie::vector<uint8, 64> c43 = aie::broadcast<uint8, 64>(0x43);
  const aie::vector<uint32, 16> m10 = aie::broadcast<uint32, 16>(0x10101010u);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)Bt);
  AIE_LOOP_NO_UNROLL
  for (int j = 0; j < 8; j++) {
    const aie::vector<uint32, 16> V = aie::load_v<64>((const uint8 *)(rec + 2048 + j * 64)).cast_to<uint32>();
    AIE_LOOP_UNROLL_FULL
    for (int h = 0; h < 2; h++) {                     // tile t = 2j + h: bit (c + 4h) of the hi byte
      aie::vector<uint8, 64> T[4];
      AIE_LOOP_UNROLL_FULL
      for (int c = 0; c < 4; c++) {
        aie::vector<uint32, 16> sh = (h == 0) ? aie::upshift(V, 4 - c) : (c == 0 ? V : aie::downshift(V, c));
        aie::vector<uint32, 16> hb = aie::bit_and(sh, m10);
        aie::vector<uint4, 64> l4 = aie::load_v<64>((const uint4 *)(rec + (2 * j + h) * 128 + c * 32));
        aie::vector<uint8, 64> w = aie::add(aie::vector<uint8, 64>(l4.unpack()), hb.cast_to<uint8>());
        T[c] = aie::transpose(w, 8, 8);
      }
      AIE_LOOP_UNROLL_FULL
      for (int oi = 0; oi < 4; oi++) {
        const int c = (oi == 1) ? 2 : (oi == 2) ? 1 : oi;
        const int nb = c >> 1;
        auto [z0, z1] = aie::interleave_zip(T[c], c43, 1);
        aie::vector<bfloat16, 64> x = aie::concat(z0.cast_to<bfloat16>(), z1.cast_to<bfloat16>());
        aie::accum<accfloat, 64> acc;
        const aie::vector<bfloat16, 64> sv = aie::load_v<64>(SV[j][nb]);
        acc.from_vector(aie::load_v<64>(BV[j][nb]));
        acc = aie::mac(acc, aie::broadcast<bfloat16, 64>(-128.0f), sv);
        acc = aie::mac(acc, x, sv);
        out.push(aie::block_vector<bfp16ebs8, 64>(::to_v64bfp16ebs8(acc)));
      }
    }
  }
}

// ================= Q6_K (REC 3360): lo nibbles B-tiles [0,2048), hi2 [2048,3072), int8 scales [t][n] 3072, d 3328 =================
static inline void split_hilo(aie::vector<float, 16> f, aie::vector<bfloat16, 16> &hi, aie::vector<bfloat16, 16> &lo) {
  aie::accum<accfloat, 16> t, u;
  t.from_vector(f); hi = t.to_vector<bfloat16>();
  u.from_vector(hi);
  t.from_vector(aie::sub(f, u.to_vector<float>()));
  lo = t.to_vector<bfloat16>();
}
static inline void prep_q6(const int8_t *__restrict rec) {
  aie::vector<bfloat16, 16> d_hi, d_lo;
  split_hilo(h2f((const uint16_t *)(rec + 3328)), d_hi, d_lo);
  for (int sp = 0; sp < 8; sp++) {
    aie::vector<int16, 32> sc16 = aie::unpack(aie::load_v<32>(rec + 3072 + sp * 32));
    aie::store_v(SCB + sp * 32, aie::to_float<bfloat16>(sc16));
  }
  AIE_LOOP_NO_UNROLL
  for (int s = 0; s < 16; s++) {
    aie::vector<bfloat16, 16> sc = aie::load_v<16>(SCB + s * 16);
    aie::accum<accfloat, 16> aa = aie::mul(d_hi, sc);
    aa = aie::mac(aa, d_lo, sc);
    aie::store_v(ABF + s * 16, aa.to_vector<bfloat16>());
  }
  AIE_LOOP_NO_UNROLL
  for (int i = 0; i < 32; i++) {            // i = s*2 + nb
    aie::vector<bfloat16, 8> a8 = aie::load_v<8>(ABF + i * 8);
    aie::vector<bfloat16, 64> av = aie::transpose(a8.grow_replicate<64>(), 8, 8);
    aie::store_v(&SV6[0][0][0] + i * 64, av);
  }
}
static inline void expand_q6(const int8_t *__restrict rec, int8_t *__restrict Bt) {
  const aie::vector<uint8, 64> c43 = aie::broadcast<uint8, 64>(0x43);
  const aie::vector<uint32, 16> m30 = aie::broadcast<uint32, 16>(0x30303030u);
  const aie::vector<bfloat16, 64> m160 = aie::broadcast<bfloat16, 64>(-160.0f);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)Bt);
  AIE_LOOP_NO_UNROLL
  for (int s = 0; s < 16; s++) {
    aie::vector<uint32, 16> H = aie::load_v<64>((const uint8 *)(rec + 2048 + s * 64)).cast_to<uint32>();
    aie::vector<uint32, 16> hh[4];
    hh[0] = aie::bit_and(aie::upshift(H, 4), m30);
    hh[1] = aie::bit_and(aie::upshift(H, 2), m30);
    hh[2] = aie::bit_and(H, m30);
    hh[3] = aie::bit_and(aie::downshift(H, 2), m30);
    aie::vector<uint8, 64> T[4];
    AIE_LOOP_UNROLL_FULL
    for (int c = 0; c < 4; c++) {
      aie::vector<uint4, 64> l4 = aie::load_v<64>((const uint4 *)(rec + s * 128 + c * 32));
      aie::vector<uint8, 64> w = aie::add(aie::vector<uint8, 64>(l4.unpack()), hh[c].cast_to<uint8>());
      T[c] = aie::transpose(w, 8, 8);
    }
    AIE_LOOP_UNROLL_FULL
    for (int oi = 0; oi < 4; oi++) {
      const int c = (oi == 1) ? 2 : (oi == 2) ? 1 : oi;
      const int nb = c >> 1;
      auto [z0, z1] = aie::interleave_zip(T[c], c43, 1);
      aie::vector<bfloat16, 64> x = aie::concat(z0.cast_to<bfloat16>(), z1.cast_to<bfloat16>());
      const aie::vector<bfloat16, 64> sv = aie::load_v<64>(SV6[s][nb]);
      aie::accum<accfloat, 64> acc = aie::mul(m160, sv);
      acc = aie::mac(acc, x, sv);
      out.push(aie::block_vector<bfp16ebs8, 64>(::to_v64bfp16ebs8(acc)));
    }
  }
}

// ================= Q8_0 (REC 4352): int8 [b][kc][nh][k16][n8] (b = 32-k block) [0,4096), fp16 scales [b][n] at 4096 =================
static inline void prep_q8(const int8_t *__restrict rec) {
  AIE_LOOP_NO_UNROLL
  for (int b = 0; b < 8; b++) {
    aie::vector<bfloat16, 16> d_hi, d_lo;
    split_hilo(h2f((const uint16_t *)(rec + 4096 + b * 32)), d_hi, d_lo);
    aie::store_v(ABF + b * 16, d_hi);
    aie::store_v(SCB + b * 16, d_lo);
  }
  AIE_LOOP_NO_UNROLL
  for (int i = 0; i < 16; i++) {            // i = b*2 + nb
    aie::vector<bfloat16, 8> h8 = aie::load_v<8>(ABF + i * 8), l8 = aie::load_v<8>(SCB + i * 8);
    aie::store_v(&SVH[0][0][0] + i * 64, aie::transpose(h8.grow_replicate<64>(), 8, 8));
    aie::store_v(&SVL[0][0][0] + i * 64, aie::transpose(l8.grow_replicate<64>(), 8, 8));
  }
}
static inline void expand_q8(const int8_t *__restrict rec, int8_t *__restrict Bt) {
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)Bt);
  AIE_LOOP_NO_UNROLL
  for (int b = 0; b < 8; b++) {
    AIE_LOOP_UNROLL_FULL
    for (int kc = 0; kc < 2; kc++) {
      AIE_LOOP_UNROLL_FULL
      for (int hf = 0; hf < 2; hf++) {           // kb32 = 4b + 2kc + hf
        AIE_LOOP_UNROLL_FULL
        for (int nb = 0; nb < 2; nb++) {
          aie::vector<int8, 64> q = aie::load_v<64>(rec + b * 512 + kc * 256 + nb * 128 + hf * 64);
          aie::vector<int8, 64> qt = aie::transpose(q, 8, 8);
          aie::vector<int16, 64> q16 = aie::unpack(qt);
          aie::vector<bfloat16, 64> x = aie::concat(aie::to_float<bfloat16>(q16.extract<32>(0)), aie::to_float<bfloat16>(q16.extract<32>(1)));
          aie::accum<accfloat, 64> acc = aie::mul(x, aie::load_v<64>(SVH[b][nb]));
          acc = aie::mac(acc, x, aie::load_v<64>(SVL[b][nb]));
          out.push(aie::block_vector<bfp16ebs8, 64>(::to_v64bfp16ebs8(acc)));
        }
      }
    }
  }
}

// ---- mmul: C[mb][nb] (8x8 f32 blocks) += A * B^T.  A stream layout [z 4][kb 32][2 mb] , B stream layout [kb 32][nb 2]
static inline void mm_blocks(const int8_t *__restrict A, const int8_t *__restrict Bt, float *__restrict C) {
  for (unsigned z = 0; z < MB; z += 2) {
    float *c0 = C + (z * 2) * 64, *c1 = C + ((z + 1) * 2) * 64;
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA((const bfp16ebs8 *)A), pB((const bfp16ebs8 *)Bt);
    pA.seek(z * 32);                    // z/2*64 blocks
    aie::accum<accfloat, 64> c00(aie::load_v<64>(c0)), c01(aie::load_v<64>(c0 + 64));
    aie::accum<accfloat, 64> c10(aie::load_v<64>(c1)), c11(aie::load_v<64>(c1 + 64));
    AIE_PREPARE_FOR_PIPELINING
    for (unsigned i = 0; i < 32; i += 4) {
      const auto A00 = pA.pop(); const auto A10 = pA.pop(); const auto B00 = pB.pop(); const auto B10 = pB.pop();
      const auto A01 = pA.pop(); const auto A11 = pA.pop(); const auto B01 = pB.pop(); const auto B11 = pB.pop();
      const auto A02 = pA.pop(); const auto A12 = pA.pop(); const auto B02 = pB.pop(); const auto B12 = pB.pop();
      const auto A03 = pA.pop(); const auto A13 = pA.pop(); const auto B03 = pB.pop(); const auto B13 = pB.pop();
      c00 = mac_8x8_8x8T(A00, B00, c00); c01 = mac_8x8_8x8T(A00, B10, c01); c10 = mac_8x8_8x8T(A10, B00, c10); c11 = mac_8x8_8x8T(A10, B10, c11);
      c00 = mac_8x8_8x8T(A01, B01, c00); c01 = mac_8x8_8x8T(A01, B11, c01); c10 = mac_8x8_8x8T(A11, B01, c10); c11 = mac_8x8_8x8T(A11, B11, c11);
      c00 = mac_8x8_8x8T(A02, B02, c00); c01 = mac_8x8_8x8T(A02, B12, c01); c10 = mac_8x8_8x8T(A12, B02, c10); c11 = mac_8x8_8x8T(A12, B12, c11);
      c00 = mac_8x8_8x8T(A03, B03, c00); c01 = mac_8x8_8x8T(A03, B13, c01); c10 = mac_8x8_8x8T(A13, B03, c10); c11 = mac_8x8_8x8T(A13, B13, c11);
    }
    aie::store_v(c0, c00.to_vector<float>()); aie::store_v(c0 + 64, c01.to_vector<float>());
    aie::store_v(c1, c10.to_vector<float>()); aie::store_v(c1 + 64, c11.to_vector<float>());
  }
}


static inline int rec_bytes(int qt) { return qt == 4 ? 2304 : qt == 5 ? 2816 : qt == 6 ? 3360 : qt == 40 ? 2304 : qt == 41 ? 2560 : 4352; }

static int st_kb = 0, st_sb = 0, st_gl = 0, st_gc = 0, st_fill = 0;

extern "C" void pf_hdr(const int8_t *__restrict a, int32_t *__restrict hdr) {
  aie::store_v(hdr, aie::load_v<16>((const int32_t *)a));
  st_kb = -1; st_sb = 0; st_fill = 0; st_gl = 0;
}
extern "C" void pf_zero(float *__restrict c) {
  for (int i = 0; i < GSB * 1024; i += 16) aie::store_v(c + i, aie::zeros<float, 16>());
}
// called at the start of each slab: chunk count into ctl[0]
extern "C" void pf_slab_begin(const int32_t *__restrict hdr, int32_t *__restrict ctl) {
  const int G = hdr[3], qt = hdr[2];
  if (++st_kb == hdr[1]) { st_kb = 0; st_sb++; }
  int rem = G - st_sb * GSB;
  st_gc = rem < GSB ? rem : GSB;
  st_gl = 0; st_fill = 0;
  ctl[0] = (st_gc * rec_bytes(qt) + PF_S - 1) / PF_S;
  // the chunks of the first record are consumed (and the record expanded) before the A slab is acquired, so the A transfer overlaps that work
  ctl[1] = st_gc > 0 ? (rec_bytes(qt) + PF_S - 1) / PF_S : 0;
  if (ctl[1] > ctl[0]) ctl[1] = ctl[0];
  ctl[2] = ctl[0] - ctl[1];
}
static inline void copy_bytes64(int8_t *__restrict dst, const int8_t *__restrict src, int n) {
  // the pipeliner is off for this kernel: unrolled by hand so the loads issue back to back
  int i = 0;
  for (; i + 512 <= n; i += 512) {
    const auto v0 = aie::load_v<64>(src + i),       v1 = aie::load_v<64>(src + i + 64),  v2 = aie::load_v<64>(src + i + 128), v3 = aie::load_v<64>(src + i + 192);
    const auto v4 = aie::load_v<64>(src + i + 256), v5 = aie::load_v<64>(src + i + 320), v6 = aie::load_v<64>(src + i + 384), v7 = aie::load_v<64>(src + i + 448);
    aie::store_v(dst + i, v0);       aie::store_v(dst + i + 64, v1);  aie::store_v(dst + i + 128, v2); aie::store_v(dst + i + 192, v3);
    aie::store_v(dst + i + 256, v4); aie::store_v(dst + i + 320, v5); aie::store_v(dst + i + 384, v6); aie::store_v(dst + i + 448, v7);
  }
  for (; i < n; i += 64) aie::store_v(dst + i, aie::load_v<64>(src + i));
}
static inline void copy_bytes32(int8_t *__restrict dst, const int8_t *__restrict src, int n) {
  for (int i = 0; i < n; i += 32) aie::store_v(dst + i, aie::load_v<32>(src + i));
}
static inline void expand_rec(const int8_t *__restrict rec, int8_t *__restrict Bt, int qt) {
  if (qt == 4 || qt == 40 || qt == 41) {          // one inlined copy of expand_q4 (program memory)
    if (qt == 4) prep_scales_k<2048, 2240, 2272>(rec); else prep_q40(rec, qt == 41);
    expand_q4(rec, Bt);
  }
  else if (qt == 5) { prep_scales_k<2560, 2752, 2784>(rec); expand_q5(rec, Bt); }
  else if (qt == 6) { prep_q6(rec); expand_q6(rec, Bt); }
  else { prep_q8(rec); expand_q8(rec, Bt); }
}
static inline void process(const int8_t *__restrict rec, const int8_t *__restrict A, float *__restrict C, int8_t *__restrict Bt, int qt) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  expand_rec(rec, Bt, qt);
  mm_blocks(A, Bt, C);
  aie::set_rounding(saved);
}
static int st_pend = -1;                     // record expanded before the A slab was acquired, waiting for its mmul
extern "C" void pf_chunk(const int8_t *__restrict chunk, const int8_t *__restrict A, float *__restrict C, int8_t *__restrict wrec,
                         int8_t *__restrict Bt, const int32_t *__restrict hdr) {
  const int qt = hdr[2];
  const int REC = rec_bytes(qt);
  int pos = 0;
  while (pos < PF_S && st_gl < st_gc) {
    int n = REC - st_fill; if (n > PF_S - pos) n = PF_S - pos;
    if (((st_fill | pos | n) & 63) == 0) copy_bytes64(wrec + st_fill, chunk + pos, n); else copy_bytes32(wrec + st_fill, chunk + pos, n);
    st_fill += n; pos += n;
    if (st_fill == REC) {
      st_fill = 0;
      process(wrec, A, C + st_gl * 1024, Bt, qt);
      st_gl++;
    }
  }
}

// phase 1 of a slab: like pf_chunk, but the record that completes is only expanded (its mmul waits for the A slab: pf_flush)
extern "C" void pf_chunk_pre(const int8_t *__restrict chunk, int8_t *__restrict wrec, int8_t *__restrict Bt, const int32_t *__restrict hdr) {
  const int qt = hdr[2];
  const int REC = rec_bytes(qt);
  int pos = 0;
  while (pos < PF_S && st_gl < st_gc) {
    int n = REC - st_fill; if (n > PF_S - pos) n = PF_S - pos;
    if (((st_fill | pos | n) & 63) == 0) copy_bytes64(wrec + st_fill, chunk + pos, n); else copy_bytes32(wrec + st_fill, chunk + pos, n);
    st_fill += n; pos += n;
    if (st_fill == REC) {
      st_fill = 0;
      const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
      expand_rec(wrec, Bt, qt);
      aie::set_rounding(saved);
      st_pend = st_gl;
      st_gl++;
    }
  }
}
// the A slab is here: mmul of the record expanded in phase 1
extern "C" void pf_flush(const int8_t *__restrict A, float *__restrict C, int8_t *__restrict Bt) {
  if (st_pend < 0) return;
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  mm_blocks(A, Bt, C + st_pend * 1024);
  aie::set_rounding(saved);
  st_pend = -1;
}
