// Teacher-forced target verification: every arm sees identical tokens and positions.
// No sampling or timing claims are made by this validation executable.
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

static std::string required_env(const char * name) {
    const char * value = std::getenv(name);
    if (!value || !*value) { throw std::runtime_error(std::string("Missing ") + name); }
    return value;
}
int main(int argc, char ** argv) {
    try {
        common_init();
        common_params params;
        if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) { return 1; }
        params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        params.speculative.draft.n_max = 3;
        params.n_outputs_max = 64;
        params.n_outputs_max_per_seq = 16;
        llama_backend_init();
        auto init = common_init_from_params(params);
        auto * ctx = init->context();
        if (!ctx) { throw std::runtime_error("Target load failed"); }
        std::ifstream fixture_stream(required_env("GROUP_FIXTURE"));
        nlohmann::json fixture; fixture_stream >> fixture;
        const auto prompts = fixture.at("prompts").get<std::vector<std::vector<llama_token>>>();
        const auto forced = fixture.at("forced").get<std::vector<std::vector<llama_token>>>();
        if (prompts.size() != 4 || forced.size() != 4) { throw std::runtime_error("Expected four sequences"); }
        const auto snapshot_dir = required_env("GROUP_SNAPSHOTS");
        std::vector<llama_token> packed(110000);
        for (int seq = 0; seq < 4; ++seq) {
            size_t count = 0;
            const auto path = snapshot_dir + "/real100k-" + std::to_string(seq) + ".bin";
            if (!llama_state_seq_load_file(ctx, path.c_str(), seq, packed.data(), packed.size(), &count)) {
                throw std::runtime_error("Restore failed: " + path);
            }
            if (seq && !llama_memory_seq_share_prefix(llama_get_memory(ctx), 0, seq, 98000)) {
                throw std::runtime_error("Prefix deduplication failed");
            }
        }
        llama_batch batch = llama_batch_init(2048, 0, 1);
        for (int seq = 0; seq < 4; ++seq) {
            for (size_t i = 100000; i < prompts[seq].size(); ++i) {
                common_batch_add(batch, prompts[seq][i], i, {seq}, i+1 == prompts[seq].size());
            }
        }
        if (llama_decode(ctx, batch)) { throw std::runtime_error("Suffix prefill failed"); }
        llama_synchronize(ctx);
        const uint32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(init->model()));
        constexpr uint32_t steps = 6, tokens_per_seq = 4, n_rows = steps*4*tokens_per_seq;
        std::ofstream output(required_env("GROUP_LOGITS"), std::ios::binary);
        output.write(reinterpret_cast<const char *>(&n_rows), sizeof(n_rows));
        output.write(reinterpret_cast<const char *>(&vocab), sizeof(vocab));
        for (uint32_t step = 0; step < steps; ++step) {
            common_batch_clear(batch);
            for (int seq = 0; seq < 4; ++seq) {
                for (uint32_t k = 0; k < tokens_per_seq; ++k) {
                    const auto offset = step*tokens_per_seq+k;
                    common_batch_add(batch, forced[seq].at(offset), prompts[seq].size()+offset, {seq}, true);
                }
            }
            if (llama_decode(ctx, batch)) { throw std::runtime_error("Verification decode failed"); }
            llama_synchronize(ctx);
            for (uint32_t row = 0; row < 4*tokens_per_seq; ++row) {
                const auto * logits = llama_get_logits_ith(ctx, row);
                if (!logits) { throw std::runtime_error("Missing output row"); }
                output.write(reinterpret_cast<const char *>(logits), vocab*sizeof(float));
            }
        }
        if (!output) { throw std::runtime_error("Output write failed"); }
        llama_batch_free(batch);
        std::cout << "teacher_rows=" << n_rows << " vocab=" << vocab << '\n';
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "teacher-groups: " << e.what() << '\n'; return 1;
    }
}
