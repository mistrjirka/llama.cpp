#pragma once
#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "../ggml/src/ggml-common.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>
// Diagnostic mirror of the Volta D4 activation quantizer and complete-K MMQ.
inline float lf_half(ggml_half h) { return ggml_fp16_to_fp32(h); }
inline std::vector<float> lf_cpu_mmq(ggml_type type,const void * data,int width,int height,const std::vector<float>& x) {
    if(width%32 || x.size()!=size_t(width))throw std::runtime_error("MMQ CPU shape");
    std::vector<int8_t> q(width);std::vector<float> scales(width/32);
    for(int off=0;off<width;off+=32) {
        float amax=0;for(int j=0;j<32;++j)amax=std::max(amax,std::abs(x[off+j]));
        float inv=amax ? 127.0f/amax : 0;scales[off/32]=amax ? 1.0f/inv : 0;
        for(int j=0;j<32;++j)q[off+j]=int8_t(std::round(x[off+j]*inv));
    }
    std::vector<float> y(height);
    for(int r=0;r<height;++r) {
        float sum=0;
        for(int b=0;b<width/32;++b) {
            int dot=0;float d=0;
            if(type==GGML_TYPE_Q8_0) {
                auto & w=static_cast<const block_q8_0 *>(data)[r*(width/32)+b];d=lf_half(w.d);
                for(int j=0;j<32;++j)dot+=int(w.qs[j])*int(q[32*b+j]);
            } else if(type==GGML_TYPE_IQ4_NL) {
                auto & w=static_cast<const block_iq4_nl *>(data)[r*(width/32)+b];d=lf_half(w.d);
                for(int j=0;j<16;++j) {
                    dot+=int(kvalues_iq4nl[w.qs[j]&15])*int(q[32*b+j]);
                    dot+=int(kvalues_iq4nl[w.qs[j]>>4])*int(q[32*b+j+16]);
                }
            } else if(type==GGML_TYPE_IQ4_XS) {
                auto & w=static_cast<const block_iq4_xs *>(data)[r*(width/256)+b/8];int sb=b%8;
                int sc=((w.scales_l[sb/2]>>(4*(sb%2)))&15) | (((w.scales_h>>(2*sb))&3)<<4);
                d=float(sc-32)*lf_half(w.d);
                for(int j=0;j<16;++j) {
                    int v=w.qs[16*sb+j];
                    dot+=int(kvalues_iq4nl[v&15])*int(q[32*b+j]);
                    dot+=int(kvalues_iq4nl[v>>4])*int(q[32*b+j+16]);
                }
            } else if(type==GGML_TYPE_IQ3_S) {
                auto & w=static_cast<const block_iq3_s *>(data)[r*(width/256)+b/8];int sb=b%8;
                d=float(1+2*((w.scales[sb/2]>>(4*(sb%2)))&15))*lf_half(w.d);
                for(int g=0;g<8;++g) {
                    uint32_t grid=iq3s_grid[w.qs[sb*8+g] | (((w.qh[sb]>>g)&1)<<8)];
                    for(int j=0;j<4;++j) {
                        int idx=g*4+j;int sign=(w.signs[sb*4+idx/8]>>(idx%8))&1;
                        int v=int((grid>>(8*j))&255);if(sign)v=-v;
                        dot+=v*int(q[32*b+idx]);
                    }
                }
            } else throw std::runtime_error("unsupported CPU MMQ type");
            sum=std::fma(d*scales[b],float(dot),sum);
        }
        y[r]=sum;
    }
    return y;
}
