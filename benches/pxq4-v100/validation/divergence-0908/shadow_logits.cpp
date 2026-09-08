// Compare one forward from the SAME reference state at every position.
// Prevent cumulative numerical drift from contaminating decode attribution.
#include "llama.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using json=nlohmann::ordered_json;
static void quiet(enum ggml_log_level level,const char*s,void*){if(level<=GGML_LOG_LEVEL_WARN)std::cerr<<s;}
static void mode(int n,int f){setenv("GGML_CUDA_PXQ4_NATIVE",n?"1":"0",1);setenv("GGML_CUDA_PXQ4_MMV_F32",f?"1":"0",1);}
static json compare(const std::vector<float>&a,const float*b,int target){
 double am=*std::max_element(a.begin(),a.end()),bm=*std::max_element(b,b+a.size()),as=0,bs=0;
 int at=std::max_element(a.begin(),a.end())-a.begin(),bt=std::max_element(b,b+a.size())-b;
 for(size_t i=0;i<a.size();++i){if(!std::isfinite(a[i])||!std::isfinite(b[i]))throw std::runtime_error("nonfinite");as+=exp(a[i]-am);bs+=exp(b[i]-bm);}
 double al=am+log(as),bl=bm+log(bs),kl=0,tv=0,sq=0,mx=0;
 bool exact=true;
 for(size_t i=0;i<a.size();++i){double ap=exp(a[i]-al),bp=exp(b[i]-bl),e=double(a[i])-b[i];kl+=ap*((a[i]-al)-(b[i]-bl));tv+=fabs(ap-bp);sq+=e*e;mx=std::max(mx,fabs(e));exact&=a[i]==b[i];}
 return {{"top1_reference",at},{"top1_candidate",bt},{"exact",exact},{"kl",kl},{"tv",tv/2},{"logit_rmse",sqrt(sq/a.size())},{"logit_maxabs",mx},{"nll_reference",al-a[target]},{"nll_candidate",bl-b[target]}};
}
int main(int argc,char**argv){
 try {
  if(argc<4)throw std::runtime_error("model fixtures output.json [scores_per_sample]");
  int scores=argc>4?std::stoi(argv[4]):32;
  setenv("GGML_CUDA_DISABLE_GRAPHS","1",1);setenv("GGML_CUDA_PXQ4_PREFILL","0",1);setenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE","f32",1);mode(0,0);
  llama_log_set(quiet,nullptr);llama_backend_init();
  auto mp=llama_model_default_params();mp.n_gpu_layers=99;mp.split_mode=LLAMA_SPLIT_MODE_NONE;
  auto model=std::unique_ptr<llama_model,decltype(&llama_model_free)>(llama_model_load_from_file(argv[1],mp),llama_model_free);if(!model)throw std::runtime_error("model load");
  const int V=llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
  json samples;std::ifstream(argv[2])>>samples;
  json rows=json::array();
  auto cp=llama_context_default_params();cp.n_ctx=2048;cp.n_batch=256;cp.n_ubatch=256;cp.n_seq_max=1;cp.n_threads=cp.n_threads_batch=16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;cp.type_k=cp.type_v=GGML_TYPE_F16;cp.no_perf=true;
  for(auto&sample:samples){
   auto ref=std::unique_ptr<llama_context,decltype(&llama_free)>(llama_init_from_model(model.get(),cp),llama_free);
   auto cand=std::unique_ptr<llama_context,decltype(&llama_free)>(llama_init_from_model(model.get(),cp),llama_free);
   if(!ref||!cand)throw std::runtime_error("context");
   auto tokens=sample["tokens"].get<std::vector<llama_token>>();std::string name=sample["name"];
   const int start=384,stop=std::min(start+scores,512);
   auto batch=llama_batch_init(256,0,1);
   auto decode=[&](llama_context*c,int pos,int n,bool logits){batch.n_tokens=n;for(int j=0;j<n;++j){batch.token[j]=tokens[pos+j];batch.pos[j]=pos+j;batch.n_seq_id[j]=1;batch.seq_id[j][0]=0;batch.logits[j]=logits;}if(llama_decode(c,batch)!=0)throw std::runtime_error("decode");};
   mode(0,0);for(int pos=0;pos<start;){int n=std::min(256,start-pos);decode(ref.get(),pos,n,false);pos+=n;}
   for(int pos=start;pos<stop;++pos){
    const size_t size=llama_state_seq_get_size(ref.get(),0);std::vector<uint8_t>state(size);
    if(llama_state_seq_get_data(ref.get(),state.data(),size,0)!=size)throw std::runtime_error("state export");
    mode(0,0);decode(ref.get(),pos,1,true);auto*z=llama_get_logits_ith(ref.get(),0);std::vector<float>reference(z,z+V);
    for(int variant=0;variant<3;++variant){
     mode(variant!=0,variant==1);
     llama_memory_clear(llama_get_memory(cand.get()),true);
     if(llama_state_seq_set_data(cand.get(),state.data(),size,0)!=size)throw std::runtime_error("state restore");
     decode(cand.get(),pos,1,true);
     auto row=compare(reference,llama_get_logits_ith(cand.get(),0),tokens[pos+1]);row["sample"]=name;row["position"]=pos;row["mode"]=variant==0?"restore-control":variant==1?"direct-f32":"integer";
     if(variant==0&&!row["exact"].get<bool>())throw std::runtime_error("restored reference differs");
     rows.push_back(row);
    }
    if((pos-start)%8==0)std::cout<<name<<" pos="<<pos<<std::endl;
   }
   llama_batch_free(batch);std::ofstream(argv[3])<<json({{"method","same-state single-step"},{"rows",rows}}).dump(2)<<"\n";
  }
  model.reset();llama_backend_free();return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}
}
