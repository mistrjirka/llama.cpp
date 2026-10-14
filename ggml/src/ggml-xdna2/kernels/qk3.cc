// K-quant / Q8_0 GEMV core kernels (AIE2P), activation in its own BO (header 64 B + M Q8 vectors).
// -DQK_T=4|5|6|8 -DQK_K=<K> -DQK_M=<#act vectors>.  Record = 16 rows x 256 weights; output object per 16-row group = M x 16 f32.
#include <aie_api/aie.hpp>
#include <stdint.h>
#ifndef QK_T
#define QK_T 4
#endif
#ifndef QK_K
#define QK_K 4096
#endif
#ifndef QK_M
#define QK_M 1
#endif
#define NBK (QK_K / 256)
#if QK_T == 8
#define SV (QK_K + QK_K / 8)                     // qs + f32 d per 32
#define REC 4352
#else
#define SV (QK_K + NBK * 4 + QK_K / 8)           // qs, f32 d per 256, i32 bsum per 32
#if QK_T == 4
#define REC 2304
#elif QK_T == 5
#define REC 2816
#else
#define REC 3360
#endif
#endif
#define ACT_BYTES ((64 + QK_M * SV + 63) & ~63)
#define ACT_RECS ((ACT_BYTES + REC - 1) / REC)
alignas(64) static float yst[QK_M][16];
static int act_cnt = 0, blk = 0;

extern "C" void qk_act_rec(const int8_t *__restrict rec, int8_t *__restrict act, int32_t *__restrict hdr) {
  int8_t *dst = act + act_cnt * REC;
  int rem = ACT_BYTES - act_cnt * REC; int n = (rem < REC ? rem : REC) / 32;
  for (int i = 0; i < n; i++) aie::store_v(dst + i * 32, aie::load_v<32>(rec + i * 32));
  if (act_cnt == 0) { hdr[0] = ((const int32_t *)act)[0]; }
  act_cnt = (act_cnt + 1 == ACT_RECS) ? 0 : act_cnt + 1;
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

#if QK_T == 4
extern "C" void qk_w_rec(const int8_t *__restrict rec, float *__restrict out, const int8_t *__restrict act_blob_) {
  const int8_t *act_blob = act_blob_ + 64;
  const uint8_t *scp = (const uint8_t *)(rec + 2048);
  aie::vector<uint16, 32> Va = aie::unpack(aie::load_v<32>(scp)), Vb = aie::unpack(aie::load_v<32>(scp + 32)),
                          Vc = aie::unpack(aie::load_v<32>(scp + 64)), Vd = aie::unpack(aie::load_v<32>(scp + 96)),
                          Ve = aie::unpack(aie::load_v<32>(scp + 128)), Vf = aie::unpack(aie::load_v<32>(scp + 160));
  const aie::vector<uint16, 32> m63 = aie::broadcast<uint16, 32>(63), m15 = aie::broadcast<uint16, 32>(15),
                                mF0 = aie::broadcast<uint16, 32>(0xF0), mC0 = aie::broadcast<uint16, 32>(0xC0);
  aie::vector<int16, 32> scp2[4], mnp2[4];     // pair vectors: [j even | j odd]
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
  aie::accum<acc32, 16> sumi[QK_M], summs[QK_M];
  const int32_t *bsm[QK_M]; const float *dact[QK_M]; const int8_t *aq[QK_M];
  for (int m = 0; m < QK_M; m++) {
    const int8_t *base = act_blob + (size_t)m * SV;
    aq[m] = base; dact[m] = (const float *)(base + QK_K); bsm[m] = (const int32_t *)(base + QK_K + NBK * 4);
  }
  for (int mg = 0; mg < (QK_M + 3) / 4; mg++) {
    for (int j = 0; j < 8; j++) {
      aie::vector<int16, 16> scj = scp2[j >> 1].extract<16>(j & 1), mnj = mnp2[j >> 1].extract<16>(j & 1);
      aie::mmul<4, 16, 16, int8, uint4> mm0, mm1;
      for (int hf = 0; hf < 2; hf++) {
        int off = blk * 256 + j * 32 + hf * 16;
        int r0 = (4 * mg + 0 < QK_M) ? 4 * mg + 0 : QK_M - 1, r1 = (4 * mg + 1 < QK_M) ? 4 * mg + 1 : QK_M - 1,
            r2 = (4 * mg + 2 < QK_M) ? 4 * mg + 2 : QK_M - 1, r3 = (4 * mg + 3 < QK_M) ? 4 * mg + 3 : QK_M - 1;
        aie::vector<int8, 64> A = aie::concat(aie::concat(aie::load_v<16>(aq[r0] + off), aie::load_v<16>(aq[r1] + off)),
                                              aie::concat(aie::load_v<16>(aq[r2] + off), aie::load_v<16>(aq[r3] + off)));
        aie::vector<uint4, 256> B = aie::load_v<256>((const uint4 *)(rec + (2 * j + hf) * 128));
        if (hf == 0) mm0.mul(A, B); else mm1.mul(A, B);
      }
      aie::vector<int16, 64> c0 = mm0.to_vector<int16>(), c1 = mm1.to_vector<int16>();
      for (int r = 0; r < 4 && 4 * mg + r < QK_M; r++) {
        int m = 4 * mg + r;
        aie::vector<int16, 16> bsv = aie::broadcast<int16, 16>((int16_t)bsm[m][blk * 8 + j]);
        if (j == 0) { sumi[m] = aie::mul(c0.extract<16>(r), scj); summs[m] = aie::mul(mnj, bsv); }
        else { sumi[m] = aie::mac(sumi[m], c0.extract<16>(r), scj); summs[m] = aie::mac(summs[m], mnj, bsv); }
        sumi[m] = aie::mac(sumi[m], c1.extract<16>(r), scj);
      }
    }
  }
  for (int m = 0; m < QK_M; m++) {
    aie::vector<float, 16> y = (blk == 0) ? aie::zeros<float, 16>() : aie::load_v<16>(yst[m]);
    aie::vector<float, 16> si = aie::to_float(sumi[m].to_vector<int32>(0));
    aie::vector<float, 16> ss = aie::to_float(summs[m].to_vector<int32>(0));
    aie::vector<float, 16> t1 = aie::mul(d_f, si).to_vector<float>();
    aie::vector<float, 16> t2 = aie::mul(dmin_f, ss).to_vector<float>();
    aie::vector<float, 16> t = aie::sub(t1, t2);
    y = aie::add(y, aie::mul(t, aie::broadcast<float, 16>(dact[m][blk])).to_vector<float>());
    if (blk == NBK - 1) aie::store_v(out + m * 16, y); else aie::store_v(yst[m], y);
  }
  blk++;
  if (blk == NBK) blk = 0;
}
#endif

#if QK_T == 6
static_assert(QK_M == 1, "Q6_K kernel is M=1");
extern "C" void qk_w_rec(const int8_t *__restrict rec, float *__restrict out, const int8_t *__restrict act_blob_) {
  const int8_t *act_blob = act_blob_ + 64;
  const int8_t *aq = act_blob;
  const float *dact = (const float *)(act_blob + QK_K);
  const int8_t *sp = rec + 3072;                     // int8 scales [s][row]
  aie::vector<float, 16> d_f = h2f((const uint16_t *)(rec + 3328));
  aie::accum<acc64, 16> sumi;
  const aie::vector<uint32, 16> m30 = aie::broadcast<uint32, 16>(0x30303030u);
  const aie::vector<int8, 64> c32 = aie::broadcast<int8, 64>(32);
  for (int s = 0; s < 16; s++) {
    aie::vector<uint8, 64> H8 = aie::load_v<64>((const uint8 *)(rec + 2048 + s * 64));
    aie::vector<uint32, 16> H = H8.cast_to<uint32>();
    aie::vector<uint32, 16> h0 = aie::bit_and(aie::upshift(H, 4), m30);     // bits 0,1 -> 4,5
    aie::vector<uint32, 16> h1 = aie::bit_and(aie::upshift(H, 2), m30);     // bits 2,3 -> 4,5
    aie::vector<uint32, 16> h2 = aie::bit_and(H, m30);                      // bits 4,5
    aie::vector<uint32, 16> h3 = aie::bit_and(aie::downshift(H, 2), m30);   // bits 6,7 -> 4,5
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
  aie::vector<float, 16> y = (blk == 0) ? aie::zeros<float, 16>() : aie::load_v<16>(yst[0]);
  aie::vector<float, 16> si = aie::to_float(sumi.to_vector<int32>(0));
  aie::vector<float, 16> t1 = aie::mul(d_f, si).to_vector<float>();
  y = aie::add(y, aie::mul(t1, aie::broadcast<float, 16>(dact[blk])).to_vector<float>());
  if (blk == NBK - 1) aie::store_v(out, y); else aie::store_v(yst[0], y);
  blk++;
  if (blk == NBK) blk = 0;
}
#endif

#if QK_T == 5
static_assert(QK_M == 1, "Q5_K kernel is M=1");
extern "C" void qk_w_rec(const int8_t *__restrict rec, float *__restrict out, const int8_t *__restrict act_) {
  const int8_t *aq = act_ + 64;
  const float *dact = (const float *)(aq + QK_K);
  const int32_t *bsm = (const int32_t *)(aq + QK_K + NBK * 4);
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
  aie::vector<float, 16> y = (blk == 0) ? aie::zeros<float, 16>() : aie::load_v<16>(yst[0]);
  aie::vector<float, 16> si = aie::to_float(sumi.to_vector<int32>(0));
  aie::vector<float, 16> ss = aie::to_float(summs.to_vector<int32>(0));
  aie::vector<float, 16> t1 = aie::mul(d_f, si).to_vector<float>();
  aie::vector<float, 16> t2 = aie::mul(dmin_f, ss).to_vector<float>();
  aie::vector<float, 16> t = aie::sub(t1, t2);
  y = aie::add(y, aie::mul(t, aie::broadcast<float, 16>(dact[blk])).to_vector<float>());
  if (blk == NBK - 1) aie::store_v(out, y); else aie::store_v(yst[0], y);
  blk++;
  if (blk == NBK) blk = 0;
}
#endif

#if QK_T == 8
static_assert(QK_M == 1, "Q8_0 kernel is M=1");
extern "C" void qk_w_rec(const int8_t *__restrict rec, float *__restrict out, const int8_t *__restrict act_) {
  const int8_t *aq = act_ + 64;
  const float *dact = (const float *)(aq + QK_K);       // f32 d per 32
  aie::vector<float, 16> y = (blk == 0) ? aie::zeros<float, 16>() : aie::load_v<16>(yst[0]);
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
  if (blk == NBK - 1) aie::store_v(out, y); else aie::store_v(yst[0], y);
  blk++;
  if (blk == NBK) blk = 0;
}
#endif
