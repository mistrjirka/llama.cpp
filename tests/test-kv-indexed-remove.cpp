// Differential test: preserve every cell, sequence set, position and free-head result.
#include "llama-kv-cells.h"
#include <algorithm>
#include <cstdio>
#include <random>
#include <stdexcept>
static uint64_t checks=0;
static void check(bool v) {++checks;if(!v)throw std::runtime_error("indexed removal differs from scan");}
static uint32_t scan(llama_kv_cells &c,int s,llama_pos a,llama_pos b) {
 uint32_t first=c.size();
 for(uint32_t i=0;i<c.size();++i)if(c.pos_in(i,a,b)&&c.seq_has(i,s)&&c.seq_rm(i,s))first=std::min(first,i);
 return first;
}
static void equal(const llama_kv_cells&a,const llama_kv_cells&b){
 check(a.size()==b.size());check(a.get_used()==b.get_used());check(a.used_min()==b.used_min());check(a.used_max_p1()==b.used_max_p1());check(a.get_has_shift()==b.get_has_shift());
 for(uint32_t i=0;i<a.size();++i){check(a.is_empty(i)==b.is_empty(i));check(a.seq_get_all(i)==b.seq_get_all(i));if(!a.is_empty(i)){
  check(a.pos_get(i)==b.pos_get(i));check(a.get_shift(i)==b.get_shift(i));auto x=a.ext_get(i),y=b.ext_get(i);check(x.x==y.x&&x.y==y.y&&x.tok==y.tok);
 }}
 for(int s=0;s<8;++s){check(a.seq_pos_min(s)==b.seq_pos_min(s));check(a.seq_pos_max(s)==b.seq_pos_max(s));
 for(int p:{0,1,17,31,63,100}){check(a.seq_pos_idx(s,p)==b.seq_pos_idx(s,p));check(a.seq_pos_tok_le(s,p)==b.seq_pos_tok_le(s,p));}}
}
int main(){try{
 std::mt19937 rng(617314);uint64_t removes=0;
 for(int trial=0;trial<100;++trial){llama_kv_cells a,b;const unsigned N=32+rng()%225;a.resize(N);b.resize(N);
 for(int step=0;step<1000;++step){unsigned i=rng()%N;int s=rng()%8;unsigned op=rng()%8;
  if(op<2){llama_pos lo=(int)(rng()%90),hi=(rng()%4==0)?std::numeric_limits<llama_pos>::max():(int)(rng()%90);
   auto x=scan(a,s,lo,hi);auto y=b.seq_rm_range(s,lo,hi);check(x==y);++removes;
  }else if(a.is_empty(i)){auto p=(llama_pos)(rng()%64);llama_kv_cell_ext ext;ext.x=rng()%7;ext.y=rng()%9;ext.tok=rng()%1000;
   a.pos_set(i,p);b.pos_set(i,p);a.seq_add(i,s);b.seq_add(i,s);a.ext_set(i,ext);b.ext_set(i,ext);
  }else if(op==2){if(!a.seq_has(i,s)){a.seq_add(i,s);b.seq_add(i,s);}}
  else if(op==3){a.rm(i);b.rm(i);}
  else if(op==4){int delta=(int)(rng()%9)-4;check(a.pos_add(i,delta)==b.pos_add(i,delta));}
  else if(op==5){int d=1+rng()%3;a.pos_div(i,d);b.pos_div(i,d);}
  else if(op==6){check(a.seq_keep(i,s)==b.seq_keep(i,s));}
  else {a.reset_shift();b.reset_shift();auto ca=a,cb=b;a=std::move(ca);b=std::move(cb);}
  equal(a,b);
 }
 for(int s=0;s<8;++s){check(scan(a,s,0,INT32_MAX)==b.seq_rm_range(s,0,INT32_MAX));equal(a,b);}
 }
 std::printf("PASS checks=%llu removals=%llu trials=100 randomized_steps=100000\n",(unsigned long long)checks,(unsigned long long)removes);return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
