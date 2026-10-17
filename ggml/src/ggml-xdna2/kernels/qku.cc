// Unified decode GEMV core kernel (AIE2P): Q4_K, Q5_K, Q6_K, Q8_0 in ONE core program, K and the qtype are RUNTIME values read from
// the activation header.  One xclbin serves every (qtype, K<=KMAX, N).   M = 1 (one activation vector).
//
// ABI (see RESULTS.md "Unified kernel ABI"):
//   The core receives ONE byte stream through a single ObjectFifo of CHUNK = QK_S byte objects:
//     [act chunks: n_act][weight chunks: n_w]      (act is padded to whole chunks, the weight record stream is padded to whole chunks)
//   header (first 64 B of the activation = hdr[16] i32):
//     0 G (16-row groups per core)  1 K  2 qtype (4,5,6,8)  3 n_act-1 (act chunks after the first)  4 n_w (weight chunks)  5 total records (G*K/256)
//   activation (after the 64 B header): Q4_K/Q5_K/Q6_K : q[K] | f32 d[K/256] | i32 bsum32[K/32];  Q8_0: q[K] | f32 d[K/32]
//   weight records (REC bytes, 16 rows x 256 weights) are a CONTIGUOUS per-core byte stream (record order: group g, k-block kb), cut into chunks;
//   a record may straddle a chunk boundary (copied into wrec) -- chunks need not be record aligned.
//   Outputs: one 16 x f32 object per group (qk_flush), staged in ost[GMAX][16] while streaming.
//   ALL buffers must be multiples of 64 B and 64 B aligned: 512-bit vector loads/stores silently round the address DOWN to 64 B.
#include <aie_api/aie.hpp>
#include <stdint.h>
#ifndef QK_S
#define QK_S 1024
#endif
#define GSB 5     // groups per superblock (layout v2, see ggml-xdna2-layout.h); must equal layout::GSB

// LAYOUT v2: the per-core weight stream is the sequence of slabs (sb, kb): gc = min(GSB, G - sb*GSB) records (gl = 0..gc-1), padded to a whole
// number of QK_S chunks.  yst[gl] holds the running sums of the gc groups of the superblock.
alignas(64) static float yst[GSB][16];
static int blk = 0, recs_done = 0, sb = 0, gl = 0, gc = 0, fill = 0, act_got = 0, fl = 0;
static bool slab_done = false;

static inline void copy_bytes64(int8_t *__restrict dst, const int8_t *__restrict src, int n) {   // n multiple of 64, 64-aligned
  for (int i = 0; i < n; i += 64) aie::store_v(dst + i, aie::load_v<64>(src + i));
}
static inline void copy_bytes32(int8_t *__restrict dst, const int8_t *__restrict src, int n) {   // n multiple of 32, 32-aligned
  for (int i = 0; i < n; i += 32) aie::store_v(dst + i, aie::load_v<32>(src + i));
}

extern "C" void qk_first(const int8_t *__restrict chunk, int8_t *__restrict act, int32_t *__restrict hdr) {
  copy_bytes64(act, chunk, QK_S);
  aie::store_v(hdr, aie::load_v<16>((const int32_t *)chunk));
  const int G0 = ((const int32_t *)chunk)[0];
  blk = 0; recs_done = 0; sb = 0; gl = 0; gc = G0 < GSB ? G0 : GSB; fill = 0; fl = 0; act_got = 1; slab_done = false;
}
extern "C" void qk_act(const int8_t *__restrict chunk, int8_t *__restrict act) {
  copy_bytes64(act + act_got * QK_S, chunk, QK_S);
  act_got++;
}

// fp16 (as uint16 x16) -> f32 x16 (normal, zero, denormal)
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

// ---- accumulate helper: y(stage) += t * dact
static inline void fin(float *out, aie::vector<float, 16> t, float da, bool last) {
  aie::vector<float, 16> y = (blk == 0) ? aie::zeros<float, 16>() : aie::load_v<16>(yst[gl]);
  y = aie::add(y, aie::mul(t, aie::broadcast<float, 16>(da)).to_vector<float>());
  if (last) aie::store_v(out, y); else aie::store_v(yst[gl], y);
}

// ================= Q4_K (REC 2304) =================
static void rec_q4(const int8_t *__restrict rec, float *__restrict out, const int8_t *__restrict act_blob_, int K, bool last) {
  const int NBK = K >> 8;
  const int8_t *act_blob = act_blob_ + 64;
  const uint8_t *scp = (const uint8_t *)(rec + 2048);
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
  aie::vector<float, 16> d_f = h2f((const uint16_t *)(rec + 2048 + 192));
  aie::vector<float, 16> dmin_f = h2f((const uint16_t *)(rec + 2048 + 224));
  aie::accum<acc32, 16> sumi, summs;
  const int8_t *aq = act_blob;
  const float *dact = (const float *)(act_blob + K);
  const int32_t *bsm = (const int32_t *)(act_blob + K + NBK * 4);
  for (int j = 0; j < 8; j++) {
    aie::vector<int16, 16> scj = scp2[j >> 1].extract<16>(j & 1), mnj = mnp2[j >> 1].extract<16>(j & 1);
    aie::mmul<4, 16, 16, int8, uint4> mm0, mm1;
    for (int hf = 0; hf < 2; hf++) {
      int off = blk * 256 + j * 32 + hf * 16;
      aie::vector<int8, 16> a16 = aie::load_v<16>(aq + off);
      aie::vector<int8, 64> A = aie::concat(aie::concat(a16, a16), aie::concat(a16, a16));
      aie::vector<uint4, 256> B = aie::load_v<256>((const uint4 *)(rec + (2 * j + hf) * 128));
      if (hf == 0) mm0.mul(A, B); else mm1.mul(A, B);
    }
    aie::vector<int16, 64> c0 = mm0.to_vector<int16>(), c1 = mm1.to_vector<int16>();
    aie::vector<int16, 16> bsv = aie::broadcast<int16, 16>((int16_t)bsm[blk * 8 + j]);
    if (j == 0) { sumi = aie::mul(c0.extract<16>(0), scj); summs = aie::mul(mnj, bsv); }
    else { sumi = aie::mac(sumi, c0.extract<16>(0), scj); summs = aie::mac(summs, mnj, bsv); }
    sumi = aie::mac(sumi, c1.extract<16>(0), scj);
  }
  aie::vector<float, 16> si = aie::to_float(sumi.to_vector<int32>(0));
  aie::vector<float, 16> ss = aie::to_float(summs.to_vector<int32>(0));
  aie::vector<float, 16> t1 = aie::mul(d_f, si).to_vector<float>();
  aie::vector<float, 16> t2 = aie::mul(dmin_f, ss).to_vector<float>();
  fin(out, aie::sub(t1, t2), dact[blk], last);
}

// ================= Q5_K (REC 2816) =================
static void rec_q5(const int8_t *__restrict rec, float *__restrict out, const int8_t *__restrict act_, int K, bool last) {
  const int NBK = K >> 8;
  const int8_t *aq = act_ + 64;
  const float *dact = (const float *)(aq + K);
  const int32_t *bsm = (const int32_t *)(aq + K + NBK * 4);
  const uint8_t *scp = (const uint8_t *)(rec + 2560);
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
  aie::vector<float, 16> d_f = h2f((const uint16_t *)(rec + 2752));
  aie::vector<float, 16> dmin_f = h2f((const uint16_t *)(rec + 2784));
  aie::accum<acc64, 16> sumi;
  aie::accum<acc32, 16> summs;
  const aie::vector<uint32, 16> m10 = aie::broadcast<uint32, 16>(0x10101010u);
  for (int j = 0; j < 8; j++) {
    aie::vector<uint32, 16> V = aie::load_v<64>((const uint8 *)(rec + 2048 + j * 64)).cast_to<uint32>();
    aie::vector<int16, 16> scj = scp2[j >> 1].extract<16>(j & 1), mnj = mnp2[j >> 1].extract<16>(j & 1);
    aie::mmul<4, 16, 8, int8, int8> m0, m1;
    for (int hf = 0; hf < 2; hf++) {
      aie::vector<int8, 16> a16 = aie::load_v<16>(aq + blk * 256 + j * 32 + hf * 16);
      aie::vector<int8, 32> a32 = aie::concat(a16, a16);
      aie::vector<int8, 64> A = aie::concat(a32, a32);
      aie::vector<int8, 64> w[4];
      for (int c = 0; c < 4; c++) {
        aie::vector<uint4, 64> l = aie::load_v<64>((const uint4 *)(rec + (2 * j + hf) * 128 + c * 32));
        aie::vector<uint8, 64> lo8 = aie::unpack(l);
        aie::vector<uint32, 16> sh = (hf == 0) ? aie::upshift(V, 4 - c) : (c == 0 ? V : aie::downshift(V, c));
        aie::vector<uint32, 16> hb = aie::bit_and(sh, m10);
        w[c] = aie::add(lo8, hb.cast_to<uint8>()).cast_to<int8>();
      }
      if (hf == 0) { m0.mul(A, aie::concat(w[0], w[1])); m1.mul(A, aie::concat(w[2], w[3])); }
      else { m0.mac(A, aie::concat(w[0], w[1])); m1.mac(A, aie::concat(w[2], w[3])); }
    }
    aie::vector<int32, 16> cv = aie::concat(m0.to_vector<int32>().extract<8>(0), m1.to_vector<int32>().extract<8>(0));
    aie::vector<int32, 16> sc32 = aie::unpack(scj);
    aie::vector<int16, 16> bsv = aie::broadcast<int16, 16>((int16_t)bsm[blk * 8 + j]);
    if (j == 0) { sumi = aie::mul(cv, sc32); summs = aie::mul(mnj, bsv); }
    else { sumi = aie::mac(sumi, cv, sc32); summs = aie::mac(summs, mnj, bsv); }
  }
  aie::vector<float, 16> si = aie::to_float(sumi.to_vector<int32>(0));
  aie::vector<float, 16> ss = aie::to_float(summs.to_vector<int32>(0));
  aie::vector<float, 16> t1 = aie::mul(d_f, si).to_vector<float>();
  aie::vector<float, 16> t2 = aie::mul(dmin_f, ss).to_vector<float>();
  fin(out, aie::sub(t1, t2), dact[blk], last);
}

// ================= Q6_K (REC 3360) =================
static void rec_q6(const int8_t *__restrict rec, float *__restrict out, const int8_t *__restrict act_blob_, int K, bool last) {
  const int8_t *aq = act_blob_ + 64;
  const float *dact = (const float *)(aq + K);
  const int8_t *sp = rec + 3072;
  aie::vector<float, 16> d_f = h2f((const uint16_t *)(rec + 3328));
  aie::accum<acc64, 16> sumi;
  const aie::vector<uint32, 16> m30 = aie::broadcast<uint32, 16>(0x30303030u);
  const aie::vector<int8, 64> c32 = aie::broadcast<int8, 64>(32);
  for (int s = 0; s < 16; s++) {
    aie::vector<uint8, 64> H8 = aie::load_v<64>((const uint8 *)(rec + 2048 + s * 64));
    aie::vector<uint32, 16> H = H8.cast_to<uint32>();
    aie::vector<uint32, 16> h0 = aie::bit_and(aie::upshift(H, 4), m30);
    aie::vector<uint32, 16> h1 = aie::bit_and(aie::upshift(H, 2), m30);
    aie::vector<uint32, 16> h2 = aie::bit_and(H, m30);
    aie::vector<uint32, 16> h3 = aie::bit_and(aie::downshift(H, 2), m30);
    aie::vector<uint32, 16> hh[4] = {h0, h1, h2, h3};
    aie::vector<int8, 64> wc[4];
    for (int c = 0; c < 4; c++) {
      aie::vector<uint4, 64> l = aie::load_v<64>((const uint4 *)(rec + s * 128 + c * 32));
      aie::vector<uint8, 64> lo8 = aie::unpack(l);
      aie::vector<uint8, 64> w = aie::add(lo8, hh[c].cast_to<uint8>());
      wc[c] = aie::sub(w.cast_to<int8>(), c32);
    }
    aie::vector<int8, 16> a16 = aie::load_v<16>(aq + blk * 256 + s * 16);
    aie::vector<int8, 32> a32 = aie::concat(a16, a16);
    aie::vector<int8, 64> A = aie::concat(a32, a32);
    aie::mmul<4, 16, 8, int8, int8> m0, m1;
    m0.mul(A, aie::concat(wc[0], wc[1]));
    m1.mul(A, aie::concat(wc[2], wc[3]));
    aie::vector<int32, 16> cv = aie::concat(m0.to_vector<int32>().extract<8>(0), m1.to_vector<int32>().extract<8>(0));
    aie::vector<int32, 16> sc = aie::unpack(aie::unpack(aie::load_v<16>(sp + s * 16)));
    if (s == 0) sumi = aie::mul(cv, sc); else sumi = aie::mac(sumi, cv, sc);
  }
  aie::vector<float, 16> si = aie::to_float(sumi.to_vector<int32>(0));
  aie::vector<float, 16> t1 = aie::mul(d_f, si).to_vector<float>();
  fin(out, t1, dact[blk], last);
}

// ================= Q8_0 (REC 4352) =================
static void rec_q8(const int8_t *__restrict rec, float *__restrict out, const int8_t *__restrict act_, int K, bool last) {
  const int8_t *aq = act_ + 64;
  const float *dact = (const float *)(aq + K);       // f32 d per 32
  aie::vector<float, 16> y = (blk == 0) ? aie::zeros<float, 16>() : aie::load_v<16>(yst[gl]);
  for (int b = 0; b < 8; b++) {
    aie::mmul<4, 16, 8, int8, int8> m0, m1;
    for (int kc = 0; kc < 2; kc++) {
      aie::vector<int8, 16> a16 = aie::load_v<16>(aq + blk * 256 + b * 32 + kc * 16);
      aie::vector<int8, 32> a32 = aie::concat(a16, a16);
      aie::vector<int8, 64> A = aie::concat(a32, a32);
      const int8_t *t0 = rec + b * 512 + (kc * 2) * 128;
      aie::vector<int8, 128> B0 = aie::concat(aie::load_v<64>(t0), aie::load_v<64>(t0 + 64));
      aie::vector<int8, 128> B1 = aie::concat(aie::load_v<64>(t0 + 128), aie::load_v<64>(t0 + 192));
      if (kc == 0) { m0.mul(A, B0); m1.mul(A, B1); } else { m0.mac(A, B0); m1.mac(A, B1); }
    }
    aie::vector<int32, 16> cv = aie::concat(m0.to_vector<int32>().extract<8>(0), m1.to_vector<int32>().extract<8>(0));
    aie::vector<float, 16> dw = h2f((const uint16_t *)(rec + 4096 + b * 32));
    aie::vector<float, 16> t1 = aie::mul(aie::to_float(cv), dw).to_vector<float>();
    y = aie::add(y, aie::mul(t1, aie::broadcast<float, 16>(dact[blk * 8 + b])).to_vector<float>());
  }
  if (last) aie::store_v(out, y); else aie::store_v(yst[gl], y);
}

// ================= Q4_0 (REC 2304) / Q4_1 (REC 2560): nibbles in the Q4_K tiles, fp16 d [b][n] at 2048, Q4_1 fp16 m [b][n] at 2304 =================
// Q8_0-style activations (int8, f32 d per 32).  Q4_0: w = (q - 8) d;  Q4_1: w = q d + m.
static void rec_q40(const int8_t *__restrict rec, float *__restrict out, const int8_t *__restrict act_, int K, bool last, bool q41) {
  const int8_t *aq = act_ + 64;
  const float *dact = (const float *)(aq + K);
  aie::vector<float, 16> y = (blk == 0) ? aie::zeros<float, 16>() : aie::load_v<16>(yst[gl]);
  for (int b = 0; b < 8; b++) {
    aie::mmul<4, 16, 16, int8, uint4> mm0, mm1;
    aie::vector<int8, 32> a32 = aie::load_v<32>(aq + blk * 256 + b * 32);
    for (int hf = 0; hf < 2; hf++) {
      aie::vector<int8, 16> a16 = a32.extract<16>(hf);
      aie::vector<int8, 64> A = aie::concat(aie::concat(a16, a16), aie::concat(a16, a16));
      aie::vector<uint4, 256> B = aie::load_v<256>((const uint4 *)(rec + (2 * b + hf) * 128));
      if (hf == 0) mm0.mul(A, B); else mm1.mul(A, B);
    }
    const int asum = aie::reduce_add(aie::vector<int16, 32>(aie::unpack(a32)));
    aie::vector<int32, 16> dot = aie::add(aie::unpack(mm0.to_vector<int16>().extract<16>(0)), aie::unpack(mm1.to_vector<int16>().extract<16>(0)));
    aie::vector<float, 16> dw = h2f((const uint16_t *)(rec + 2048 + b * 32));
    aie::vector<float, 16> t1;
    if (q41) {
      aie::vector<float, 16> mw = h2f((const uint16_t *)(rec + 2304 + b * 32));
      t1 = aie::add(aie::mul(aie::to_float(dot), dw).to_vector<float>(), aie::mul(mw, aie::broadcast<float, 16>((float)asum)).to_vector<float>());
    } else {
      dot = aie::sub(dot, aie::broadcast<int32, 16>(8 * asum));
      t1 = aie::mul(aie::to_float(dot), dw).to_vector<float>();
    }
    y = aie::add(y, aie::mul(t1, aie::broadcast<float, 16>(dact[blk * 8 + b])).to_vector<float>());
  }
  if (last) aie::store_v(out, y); else aie::store_v(yst[gl], y);
}

static inline int rec_bytes(int qt) { return qt == 4 ? 2304 : qt == 5 ? 2816 : qt == 6 ? 3360 : qt == 40 ? 2304 : qt == 41 ? 2560 : 4352; }

static inline void process(const int8_t *rec, const int8_t *act, float *ost, int qt, int K, int G) {
  const int NBK = K >> 8;
  const bool last = (blk == NBK - 1);
  float *out = ost + (sb * GSB + gl) * 16;
  if (qt == 4) rec_q4(rec, out, act, K, last);
  else if (qt == 5) rec_q5(rec, out, act, K, last);
  else if (qt == 6) rec_q6(rec, out, act, K, last);
  else if (qt == 40 || qt == 41) rec_q40(rec, out, act, K, last, qt == 41);
  else rec_q8(rec, out, act, K, last);
  recs_done++;
  if (++gl == gc) {                     // slab complete (the rest of its last chunk is padding)
    gl = 0; slab_done = true;
    if (++blk == NBK) { blk = 0; sb++; const int rem = G - sb * GSB; gc = rem < GSB ? rem : GSB; }
  }
}

extern "C" void qk_w(const int8_t *__restrict chunk, const int8_t *__restrict act, int8_t *__restrict wrec, float *__restrict ost,
                     const int32_t *__restrict hdr) {
  const int G = hdr[0], K = hdr[1], qt = hdr[2], total = hdr[5];
  const int REC = rec_bytes(qt);
  int pos = 0;
  while (pos < QK_S && recs_done < total) {
    int n = REC - fill; if (n > QK_S - pos) n = QK_S - pos;       // records straddle chunks: assemble in wrec
    if (((fill | pos | n) & 63) == 0) copy_bytes64(wrec + fill, chunk + pos, n); else copy_bytes32(wrec + fill, chunk + pos, n);
    fill += n; pos += n;
    if (fill == REC) {
      fill = 0;
      process(wrec, act, ost, qt, K, G);
      if (slab_done) { slab_done = false; break; }                // skip the slab padding
    }
  }
}

extern "C" void qk_flush(float *__restrict o, const float *__restrict ost) {
  aie::store_v(o, aie::load_v<16>(ost + fl * 16));
  fl++;
}
