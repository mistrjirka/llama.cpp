// Fixed input batches remove request-arrival timing and token choices as confounders.
#include "arg.h"
#include "common.h"
#include "llama.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
static void require(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
static void write_u32(std::ofstream &o, uint32_t n) {o.write((const char *)&n,sizeof(n));}
int main(int argc, char **argv) {
 try {
  common_params params; common_init();
  if (!common_params_parse(argc,argv,params,LLAMA_EXAMPLE_SERVER)) return 2;
  ggml_backend_load_all();
  auto init=common_init_from_params(params);
  llama_model *model=init->model();llama_context *ctx=init->context();
  require(model && ctx,"model initialization");
  const char *dir=std::getenv("VERIFY_FIXTURE");const char *output=std::getenv("VERIFY_OUTPUT");
  require(dir && output,"missing VERIFY_FIXTURE or VERIFY_OUTPUT");
  std::ifstream inputs(std::string(dir)+"/teacher-inputs.bin",std::ios::binary);
  std::array<std::array<llama_token,64>,4> tokens{};
  inputs.read((char *)tokens.data(),sizeof(tokens));require(inputs.good(),"teacher inputs");
  const uint32_t vocab=llama_vocab_n_tokens(llama_model_get_vocab(model));
  std::ofstream out(output,std::ios::binary);require(out.good(),"open output");
  write_u32(out,0x56455231u);write_u32(out,vocab);write_u32(out,4);
  const std::array<std::array<int,4>,4> shapes={{{4,0,0,0},{1,1,1,1},{4,4,4,4},{1,2,3,4}}};
  for (const auto &lengths:shapes) {
   llama_memory_clear(llama_get_memory(ctx),false);
   for (int seq=0;seq<4;++seq) {
    const std::string file=std::string(dir)+"/real100k-"+std::to_string(seq)+".bin";
    size_t count=0;require(llama_state_seq_load_file_tokens(file.c_str(),nullptr,0,&count)>0,"metadata");
    std::vector<llama_token> saved(std::max<size_t>(count,1));
    require(llama_state_seq_load_file(ctx,file.c_str(),seq,saved.data(),saved.size(),&count)>0,"restore");
    // Slot files wrap token lists in versioned server metadata. Verify
    // the actual position count independently of that packed envelope.
    require(llama_memory_seq_pos_max(llama_get_memory(ctx),seq)==99999,"wrong saved history position");
    require(count==100000 || (count==100004 && saved[0]==LLAMA_TOKEN_NULL && saved[1]==1 && saved[2]==100000),"unexpected token envelope");
   }
   llama_batch b=llama_batch_init(32,0,1);
   for (int seq:{2,0,3,1}) for(int j=0;j<lengths[seq];++j)
    common_batch_add(b,tokens[seq][j],100000+j,{seq},true);
   require(llama_decode(ctx,b)==0,"fixed batch decode");
   for (int n:lengths) write_u32(out,(uint32_t)n);
   write_u32(out,(uint32_t)b.n_tokens);
   for (int row=0;row<b.n_tokens;++row) {
    const float *logits=llama_get_logits_ith(ctx,row);require(logits!=nullptr,"output logits");
    out.write((const char*)logits,(size_t)vocab*sizeof(float));
   }
   out.flush();require(out.good(),"write logits");
   std::cout<<"verified shape "<<lengths[0]<<","<<lengths[1]<<","<<lengths[2]<<","<<lengths[3]
            <<" outputs="<<b.n_tokens<<" rollback="<<llama_n_rs_seq(ctx)<<std::endl;
   llama_batch_free(b);
  }
  return 0;
 } catch (const std::exception &e) {std::cerr<<"ERROR "<<e.what()<<std::endl;return 1;}
}
