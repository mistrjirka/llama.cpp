// Teacher forcing: all engines receive identical token IDs and score the same rows.
#include "llama.h"
#include "ggml-backend.h"
#include <filesystem>
#include <cstdlib>
#include <regex>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstdint>
using json=nlohmann::json;
static void logfn(enum ggml_log_level level,const char*text,void*){if(level>=GGML_LOG_LEVEL_WARN)std::cerr<<text;}
// Optional diagnostic callback. It forces graph boundaries: use only for
// localization, never for reporting production throughput.
struct Trace {
 std::string prefix; int position=-1, wanted=-1, dataset=0, index=0;
 std::regex match; json rows=json::array();
 Trace(const std::string &p):prefix(p),match(std::getenv("PXQ_TRACE_MATCH")?std::getenv("PXQ_TRACE_MATCH"):".*") {
   if(const char *v=std::getenv("PXQ_TRACE_POS"))wanted=std::atoi(v);
 }
};
static bool trace_cb(ggml_tensor*t,bool ask,void*opaque){
 auto &s=*static_cast<Trace*>(opaque);
 bool take=s.dataset==0 && s.position==s.wanted && s.wanted>=0 &&
   (t->type==GGML_TYPE_F32 || t->type==GGML_TYPE_F16 || t->type==GGML_TYPE_I32) &&
   ggml_nbytes(t)<=40000000 && std::regex_search(t->name,s.match);
 if(ask)return take;
 if(!take)return true;
 std::vector<char>buf(ggml_nbytes(t));ggml_backend_tensor_get(t,buf.data(),0,buf.size());
 std::string filename=s.prefix+".tensor-"+std::to_string(s.index++)+".bin";
 std::ofstream f(filename,std::ios::binary);f.write(buf.data(),buf.size());
 json sources=json::array(); for(auto*src:t->src){if(src)sources.push_back({{"name",src->name},{"type",ggml_type_name(src->type)}});}
 s.rows.push_back({{"name",t->name},{"type",ggml_type_name(t->type)},{"ne",{t->ne[0],t->ne[1],t->ne[2],t->ne[3]}},{"nb",{t->nb[0],t->nb[1],t->nb[2],t->nb[3]}},{"op",ggml_op_name(t->op)},{"file",filename},{"sources",sources}});
 std::ofstream(s.prefix+".trace.json")<<s.rows.dump(2)<<"\n";
 return true;
}
int main(int argc,char**argv){
 try {
  if(argc<6)throw std::runtime_error("model fixture.json prefix prefill_chunk score_chunk [dataset_limit]");
  const std::string modelpath=argv[1],fixture=argv[2],prefix=argv[3];int chunk=std::stoi(argv[4]),step=std::stoi(argv[5]);
  if(chunk<1||step<1)throw std::runtime_error("positive chunks required");
  std::ifstream in(fixture);json data;in>>data;
  llama_log_set(logfn,nullptr);llama_backend_init();
  auto mp=llama_model_default_params();mp.n_gpu_layers=99;mp.split_mode=LLAMA_SPLIT_MODE_NONE;
  auto *model=llama_model_load_from_file(modelpath.c_str(),mp);if(!model)throw std::runtime_error("load failed");
  const auto*vocab=llama_model_get_vocab(model);int V=llama_vocab_n_tokens(vocab);
  std::ofstream output(prefix+".f32",std::ios::binary);if(!output)throw std::runtime_error("output open failed");
  json meta={{"n_vocab",V},{"chunk",chunk},{"step",step},{"rows",json::array()},{"datasets",json::array()}};
  #ifdef PXA_HEADER
   const auto *tokenizer=model;
#else
   const auto *tokenizer=vocab;
#endif
   Trace trace(prefix);
   int count=0;for(auto &sample:data){if(argc>6&&count>=std::stoi(argv[6]))break;++count;
   std::string name=sample.at("name");std::vector<llama_token>tokens;
   if(sample.contains("tokens"))tokens=sample.at("tokens").get<std::vector<llama_token>>();
   else{std::string text=sample.at("text");tokens.resize(text.size()+16);int n=llama_tokenize(tokenizer,text.data(),text.size(),tokens.data(),tokens.size(),true,true);if(n<0)throw std::runtime_error("tokenizer size");tokens.resize(n);}
   const int total=sample.value("total",512),scores=sample.value("scores",128),start=total-scores;
   if((int)tokens.size()<total+1)throw std::runtime_error("too few tokens for "+name);
   tokens.resize(total+1);
   auto cp=llama_context_default_params();cp.n_ctx=std::max(2048,total+16);cp.n_batch=std::max(chunk,step);cp.n_ubatch=cp.n_batch;cp.n_seq_max=1;cp.n_threads=16;cp.n_threads_batch=16;cp.type_k=cp.type_v=GGML_TYPE_F16;
#ifdef PXA_HEADER
   cp.flash_attn=true;
#else
   cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;cp.no_perf=true;
#endif
   trace.dataset=count-1;
   if(trace.wanted>=0){cp.cb_eval=trace_cb;cp.cb_eval_user_data=&trace;}
   auto*ctx=llama_init_from_model(model,cp);if(!ctx)throw std::runtime_error("context failed");
   auto batch=llama_batch_init(std::max(chunk,step),0,1);double nll=0;int nr=0;
   int begin=0;
#ifndef PXA_HEADER
   const std::string cache=sample.value("cache_file",std::string());
   const int cache_n=sample.value("cache_n",0);
   if(!cache.empty() && std::ifstream(cache).good()) {
       std::vector<llama_token> restored(cache_n);size_t n=0;
       if(!llama_state_seq_load_file(ctx,cache.c_str(),0,restored.data(),restored.size(),&n) || int(n)!=cache_n ||
               !std::equal(restored.begin(),restored.end(),tokens.begin()))throw std::runtime_error("cache restore mismatch");
       begin=cache_n;
   }
#endif
   for(int pos=begin;pos<total;){bool scoring=pos>=start;int n=std::min(scoring?step:chunk,(scoring?total:start)-pos);
#ifndef PXA_HEADER
    if(!cache.empty() && pos<cache_n)n=std::min(n,cache_n-pos);
#endif
    batch.n_tokens=n;
    for(int j=0;j<n;++j){batch.token[j]=tokens[pos+j];batch.pos[j]=pos+j;batch.n_seq_id[j]=1;batch.seq_id[j][0]=0;batch.logits[j]=scoring;}
    trace.position=pos;
     if(llama_decode(ctx,batch)!=0)throw std::runtime_error("decode failed "+name+" at "+std::to_string(pos));
    if(scoring)for(int j=0;j<n;++j){float*z=llama_get_logits_ith(ctx,j);if(!z)throw std::runtime_error("missing logits");
     double mx=-INFINITY;int top=0;for(int v=0;v<V;++v){if(!std::isfinite(z[v]))throw std::runtime_error("nonfinite logits");if(z[v]>mx){mx=z[v];top=v;}}
     double sum=0;for(int v=0;v<V;++v)sum+=exp(z[v]-mx);double loss=log(sum)+mx-z[tokens[pos+j+1]];nll+=loss;++nr;
     output.write(reinterpret_cast<char*>(z),V*sizeof(float));if(!output)throw std::runtime_error("write failed");
     meta["rows"].push_back({{"sample",name},{"position",pos+j},{"target",tokens[pos+j+1]},{"top1",top},{"nll",loss}});
    }pos+=n;
#ifndef PXA_HEADER
    if(!cache.empty() && pos==cache_n && !std::ifstream(cache).good()) {
        if(!llama_state_seq_save_file(ctx,cache.c_str(),0,tokens.data(),cache_n))throw std::runtime_error("cache save failed");
        std::cout<<"CACHE_SAVED "<<cache_n<<std::endl;
    }
#endif
   }
   meta["datasets"].push_back({{"name",name},{"tokens",tokens},{"scores",nr},{"nll",nll/nr},{"ppl",exp(nll/nr)}});
   std::ofstream(prefix+".json")<<meta.dump(2)<<"\n";
   std::cout<<name<<" rows="<<nr<<" nll="<<nll/nr<<" ppl="<<exp(nll/nr)<<std::endl;
   llama_batch_free(batch);llama_free(ctx);
  }
  std::ofstream(prefix+".json")<<meta.dump(2)<<"\n";
#ifndef PXA_HEADER
  llama_model_free(model);
#else
  llama_free_model(model);
#endif
  llama_backend_free();return 0;
 }catch(const std::exception&e){std::cerr<<"VALIDATION_FAILED: "<<e.what()<<std::endl;return 1;}
}
