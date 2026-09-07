#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
static uint32_t u32(std::ifstream& in){uint32_t x;in.read((char*)&x,4);if(!in)throw std::runtime_error("truncated header");return x;}
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("usage: compare-target reference candidate");
 std::ifstream a(argv[1],std::ios::binary),b(argv[2],std::ios::binary);
 if(u32(a)!=0x56455231u || u32(b)!=0x56455231u)throw std::runtime_error("bad magic");
 const uint32_t n=u32(a);if(u32(b)!=n)throw std::runtime_error("vocab mismatch");
 const uint32_t count=u32(a);if(u32(b)!=count)throw std::runtime_error("case mismatch");
 std::vector<float>x(n),y(n);std::cout<<std::setprecision(12)<<"[";
 for(uint32_t c=0;c<count;++c){
  std::array<uint32_t,4>lens{};for(auto& l:lens){l=u32(a);if(u32(b)!=l)throw std::runtime_error("shape mismatch");}
  const uint32_t rows=u32(a);if(u32(b)!=rows)throw std::runtime_error("rows mismatch");
  uint64_t different=0,top1=0;double max_delta=0,se=0,energy=0,max_prob=0,js_sum=0,tv_sum=0;
  for(uint32_t row=0;row<rows;++row){
   a.read((char*)x.data(),n*sizeof(float));b.read((char*)y.data(),n*sizeof(float));if(!a||!b)throw std::runtime_error("truncated logits");
   auto ax=std::max_element(x.begin(),x.end()),ay=std::max_element(y.begin(),y.end());top1+=(ax-x.begin())!=(ay-y.begin());
   double zx=0,zy=0;
   for(uint32_t i=0;i<n;++i){if(!std::isfinite(x[i])||!std::isfinite(y[i]))throw std::runtime_error("nonfinite output");
    double d=(double)x[i]-y[i];different+=x[i]!=y[i];max_delta=std::max(max_delta,std::abs(d));se+=d*d;energy+=(double)x[i]*x[i];zx+=std::exp((double)x[i]-*ax);zy+=std::exp((double)y[i]-*ay);}
   double js=0,tv=0;
   for(uint32_t i=0;i<n;++i){double p=std::exp((double)x[i]-*ax)/zx,q=std::exp((double)y[i]-*ay)/zy,m=(p+q)*.5;
    max_prob=std::max(max_prob,std::abs(p-q));tv+=std::abs(p-q)*.5;
    if(p>0)js+=.5*p*std::log(p/m);if(q>0)js+=.5*q*std::log(q/m);}
   js_sum+=js;tv_sum+=tv;
  }
  if(c)std::cout<<",";
  std::cout<<"{\"lengths\":["<<lens[0]<<","<<lens[1]<<","<<lens[2]<<","<<lens[3]<<"],\"rows\":"<<rows
   <<",\"different_float_values\":"<<different<<",\"top1_changes\":"<<top1<<",\"max_logit_delta\":"<<max_delta
   <<",\"rms_logit_delta\":"<<std::sqrt(se/(rows*n))<<",\"normalized_mse\":"<<se/std::max(energy,1e-30)
   <<",\"max_probability_delta\":"<<max_prob<<",\"mean_total_variation\":"<<tv_sum/rows<<",\"mean_js_divergence\":"<<js_sum/rows<<"}";
 }
 std::cout<<"]\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
