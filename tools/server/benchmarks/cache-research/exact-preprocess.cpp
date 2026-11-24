// Exact byte transforms for offline codec screening. No floating-point arithmetic.
// c++ -O3 -march=native -std=c++17 -shared -fPIC exact-preprocess.cpp -o /tmp/libexact-preprocess.so
#include <cstddef>
#include <cstdint>
#include <cstring>

static uint32_t load32(const uint8_t * p) { uint32_t v; std::memcpy(&v,p,4); return v; }
static void store32(uint8_t * p,uint32_t v) { std::memcpy(p,&v,4); }
static uint8_t zigzag(uint8_t v) { const int x=v<128?v:int(v)-256; return uint8_t(x<0?-2*x-1:2*x); }
static uint8_t unzigzag(uint8_t v) { return uint8_t(v&1?-int(v/2)-1:int(v/2)); }

static void bitshuffle(const uint8_t * src,uint8_t * dst,size_t bytes,bool inverse) {
    const size_t groups=bytes/8;
    if (inverse) std::memset(dst,0,groups*8);
    for(size_t g=0;g<groups;++g) {
        for(unsigned bit=0;bit<8;++bit) {
            if(inverse) {
                const uint8_t bits=src[bit*groups+g];
                for(unsigned j=0;j<8;++j) dst[g*8+j]|=((bits>>j)&1)<<bit;
            } else {
                uint8_t bits=0;
                for(unsigned j=0;j<8;++j) bits|=((src[g*8+j]>>bit)&1)<<j;
                dst[bit*groups+g]=bits;
            }
        }
    }
    std::memcpy(dst+groups*8,src+groups*8,bytes-groups*8);
}

extern "C" int exact_preprocess(int mode,int inverse,const uint8_t * src,uint8_t * dst,size_t bytes,uint8_t * scratch) {
    if(!src||!dst||!scratch||mode<0||mode>8) return -1;
    if(mode==0) { std::memcpy(dst,src,bytes); return 0; }
    if(mode>=1&&mode<=4) {
        const size_t words=bytes/4;
        if(mode==4) {
            // Shuffle F32 byte planes, then bit planes. Invert in reverse order.
            if(inverse) {
                bitshuffle(src,scratch,words*4,true);
                for(size_t i=0;i<words;++i) for(size_t b=0;b<4;++b) dst[4*i+b]=scratch[b*words+i];
            } else {
                for(size_t i=0;i<words;++i) for(size_t b=0;b<4;++b) scratch[b*words+i]=src[4*i+b];
                bitshuffle(scratch,dst,words*4,false);
            }
        } else {
            uint32_t previous=0;
            for(size_t i=0;i<words;++i) {
                if(inverse) {
                    uint32_t u;
                    if(mode==2) {
                        const uint32_t rest=uint32_t(src[words+i])|(uint32_t(src[2*words+i])<<8)|(uint32_t(src[3*words+i])<<16);
                        u=((rest>>23)<<31)|(uint32_t(src[i])<<23)|(rest&0x7fffff);
                    } else {
                        u=uint32_t(src[i])|(uint32_t(src[words+i])<<8)|(uint32_t(src[2*words+i])<<16)|(uint32_t(src[3*words+i])<<24);
                        if(mode==3) { u^=previous; previous=u; }
                    }
                    store32(dst+4*i,u);
                } else {
                    uint32_t u=load32(src+4*i);
                    if(mode==3) { const auto original=u;u^=previous;previous=original; }
                    if(mode==2) {
                        const uint32_t rest=(u&0x7fffff)|((u>>31)<<23);
                        dst[i]=uint8_t(u>>23);dst[words+i]=uint8_t(rest);dst[2*words+i]=uint8_t(rest>>8);dst[3*words+i]=uint8_t(rest>>16);
                    } else {
                        for(size_t b=0;b<4;++b) dst[b*words+i]=uint8_t(u>>(8*b));
                    }
                }
            }
        }
        std::memcpy(dst+words*4,src+words*4,bytes-words*4);
        return 0;
    }
    const size_t blocks=bytes/34;
    const bool use_zigzag=mode==6||mode==8;
    const bool use_bits=mode==7||mode==8;
    if(inverse) {
        const uint8_t * quants=src+2*blocks;
        if(use_bits) { bitshuffle(quants,scratch,blocks*32,true);quants=scratch; }
        for(size_t i=0;i<blocks;++i) {
            dst[34*i]=src[i];dst[34*i+1]=src[blocks+i];
            for(size_t j=0;j<32;++j) dst[34*i+2+j]=use_zigzag?unzigzag(quants[32*i+j]):quants[32*i+j];
        }
    } else {
        uint8_t * quants=use_bits?scratch:dst+2*blocks;
        for(size_t i=0;i<blocks;++i) {
            dst[i]=src[34*i];dst[blocks+i]=src[34*i+1];
            for(size_t j=0;j<32;++j) quants[32*i+j]=use_zigzag?zigzag(src[34*i+2+j]):src[34*i+2+j];
        }
        if(use_bits) bitshuffle(quants,dst+2*blocks,blocks*32,false);
    }
    std::memcpy(dst+blocks*34,src+blocks*34,bytes-blocks*34);
    return 0;
}
