// Writes the hidden states a DFlash draft would read from a target model, for given token sequences.
//
// A DFlash draft is conditioned on the target's layer inputs at a few layers ("taps"). Training one needs
// those states for the training text, and they must be the ones the engine produces at run time: the same
// quantized weights, and for sensenova_loop the same reduction of the two residual streams. This tool
// decodes each sequence with teacher forcing and writes, for every token, the tapped layer inputs through
// llama_get_embeddings_layer_inp, the call the DFlash driver itself uses.
//
// input  (--tokens):  u32 "LTK1", u32 n_seq, then per sequence u32 n, i32 token[n]
//        (--first):   optional, u32 per sequence of the token file: the first token whose states are written
// output (--out):     raw fp16, for each sequence in order, for each written token, n_taps * n_embd values
//        (--out).json: n_embd, taps, the first sequence, and per sequence its length and its first written token
//        (--topk-out): optional, for each written token the --topk largest logits of the next token,
//                      as (i32 token, f32 logit), largest first
// without --out and --taps only the logits are written, and the index goes to (--topk-out).json

#include "llama.h"
#include "../../src/llama-ext.h"
#include "ggml.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char * prog) {
    fprintf(stderr, "usage: %s -m model.gguf --tokens seqs.bin [--taps 2,30,34 --out taps.f16] [--first first.bin] [--topk K --topk-out file] [--start N] [--count N] [-ngl N] [--n-batch N]\n", prog);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path, tokens_path, out_path, taps_arg, first_path, topk_path;
    int ngl = 99;
    int n_batch = 4096;
    int topk = 0;
    long start = 0;
    long count = -1;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool has_value = i + 1 < argc;
        if      (arg == "-m"        && has_value) { model_path  = argv[++i]; }
        else if (arg == "--tokens"  && has_value) { tokens_path = argv[++i]; }
        else if (arg == "--out"     && has_value) { out_path    = argv[++i]; }
        else if (arg == "--taps"    && has_value) { taps_arg    = argv[++i]; }
        else if (arg == "--first"   && has_value) { first_path  = argv[++i]; }
        else if (arg == "--topk"    && has_value) { topk        = atoi(argv[++i]); }
        else if (arg == "--topk-out" && has_value) { topk_path  = argv[++i]; }
        else if (arg == "--start"   && has_value) { start       = atol(argv[++i]); }
        else if (arg == "--count"   && has_value) { count       = atol(argv[++i]); }
        else if (arg == "-ngl"      && has_value) { ngl         = atoi(argv[++i]); }
        else if (arg == "--n-batch" && has_value) { n_batch     = atoi(argv[++i]); }
        else { usage(argv[0]); return 1; }
    }
    const bool write_taps = !out_path.empty();
    if (model_path.empty() || tokens_path.empty() || write_taps != !taps_arg.empty() || (topk > 0) != !topk_path.empty() || (!write_taps && topk == 0)) {
        usage(argv[0]);
        return 1;
    }

    std::vector<uint32_t> taps;
    for (size_t pos = 0; write_taps && pos < taps_arg.size(); ) {
        const size_t comma = taps_arg.find(',', pos);
        taps.push_back((uint32_t) atoi(taps_arg.substr(pos, comma - pos).c_str()));
        if (comma == std::string::npos) { break; }
        pos = comma + 1;
    }

    // the sequences
    FILE * fin = fopen(tokens_path.c_str(), "rb");
    if (!fin) { fprintf(stderr, "cannot open %s\n", tokens_path.c_str()); return 1; }
    uint32_t header[2];
    if (fread(header, sizeof(uint32_t), 2, fin) != 2 || memcmp(&header[0], "LTK1", 4) != 0) {
        fprintf(stderr, "%s is not a token file\n", tokens_path.c_str());
        return 1;
    }
    const long n_seq_total = header[1];
    std::vector<std::vector<llama_token>> seqs;
    for (long s = 0; s < n_seq_total; ++s) {
        uint32_t n = 0;
        if (fread(&n, sizeof(n), 1, fin) != 1) { fprintf(stderr, "truncated token file\n"); return 1; }
        std::vector<llama_token> tokens(n);
        if (n > 0 && fread(tokens.data(), sizeof(llama_token), n, fin) != n) { fprintf(stderr, "truncated token file\n"); return 1; }
        if (s >= start && (count < 0 || s < start + count)) {
            seqs.push_back(std::move(tokens));
        }
    }
    fclose(fin);

    // optionally, per sequence of the token file, the first token whose states are written: a long shared
    // prompt (a system prompt, tool definitions) is decoded but need not be stored for every sequence
    std::vector<uint32_t> first(seqs.size(), 0);
    if (!first_path.empty()) {
        FILE * ffirst = fopen(first_path.c_str(), "rb");
        if (!ffirst) { fprintf(stderr, "cannot open %s\n", first_path.c_str()); return 1; }
        std::vector<uint32_t> all((size_t) n_seq_total);
        if (fread(all.data(), sizeof(uint32_t), all.size(), ffirst) != all.size()) { fprintf(stderr, "%s is too short\n", first_path.c_str()); return 1; }
        fclose(ffirst);
        for (size_t s = 0; s < seqs.size(); ++s) {
            first[s] = std::min<uint32_t>(all[(size_t) start + s], (uint32_t) seqs[s].size());
        }
    }

    size_t n_longest = 0;
    for (const auto & s : seqs) { n_longest = std::max(n_longest, s.size()); }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "cannot load %s\n", model_path.c_str()); return 1; }

    const int n_embd = llama_model_n_embd(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = (uint32_t) std::max<size_t>(n_longest, 8);
    cparams.n_batch  = (uint32_t) std::min<size_t>((size_t) n_batch, cparams.n_ctx);
    cparams.n_ubatch = cparams.n_batch;
    cparams.no_perf  = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "cannot create a context\n"); return 1; }

    for (uint32_t lid : taps) {
        llama_set_embeddings_layer_inp(ctx, lid, true);
    }

    FILE * fout = nullptr;
    if (write_taps) {
        fout = fopen(out_path.c_str(), "wb");
        if (!fout) { fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }
    }

    // optionally, the target's own next-token distribution at every written token: its top logits
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    FILE * ftopk = nullptr;
    if (topk > 0) {
        ftopk = fopen(topk_path.c_str(), "wb");
        if (!ftopk) { fprintf(stderr, "cannot write %s\n", topk_path.c_str()); return 1; }
    }
    std::vector<int32_t> order((size_t) n_vocab);
    struct topk_entry { int32_t id; float logit; };
    std::vector<topk_entry> best((size_t) std::max(topk, 1));

    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    std::vector<ggml_fp16_t> row((size_t) taps.size() * n_embd);

    const int64_t t_start = ggml_time_us();
    size_t n_tokens_done = 0;

    for (size_t s = 0; s < seqs.size(); ++s) {
        const auto & tokens = seqs[s];
        llama_memory_clear(llama_get_memory(ctx), true);

        for (size_t off = 0; off < tokens.size(); off += cparams.n_batch) {
            const int32_t n = (int32_t) std::min<size_t>(cparams.n_batch, tokens.size() - off);

            llama_batch_ext_clear(batch);
            for (int32_t i = 0; i < n; ++i) {
                const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[off + i]);
                const llama_pos pos = (llama_pos) (off + i);
                llama_batch_ext_set_pos(batch, idx, &pos);
            }
            if (topk > 0) {
                for (int32_t i = 0; i < n; ++i) {
                    llama_batch_ext_set_output_logits(batch, i, true);
                }
            } else {
                llama_batch_ext_set_output_logits(batch, n - 1, true);
            }

            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
                fprintf(stderr, "decode failed at sequence %zu\n", s);
                return 1;
            }

            std::vector<const float *> layers;
            for (uint32_t lid : taps) {
                const float * layer = llama_get_embeddings_layer_inp(ctx, lid);
                if (!layer) { fprintf(stderr, "layer %u is not tapped\n", lid); return 1; }
                layers.push_back(layer);
            }
            for (int32_t i = 0; i < n; ++i) {
                if (off + i < first[s]) {
                    continue;
                }
                for (size_t k = 0; k < taps.size(); ++k) {
                    ggml_fp32_to_fp16_row(layers[k] + (size_t) i * n_embd, row.data() + k * n_embd, n_embd);
                }
                if (write_taps && fwrite(row.data(), sizeof(ggml_fp16_t), row.size(), fout) != row.size()) {
                    fprintf(stderr, "write failed\n");
                    return 1;
                }
                if (topk > 0) {
                    const float * logits = llama_get_logits_ith(ctx, i);
                    for (int32_t v = 0; v < n_vocab; ++v) { order[(size_t) v] = v; }
                    std::partial_sort(order.begin(), order.begin() + topk, order.end(), [logits](int32_t a, int32_t b) { return logits[a] > logits[b]; });
                    for (int k = 0; k < topk; ++k) { best[(size_t) k] = { order[(size_t) k], logits[order[(size_t) k]] }; }
                    if (fwrite(best.data(), sizeof(topk_entry), (size_t) topk, ftopk) != (size_t) topk) {
                        fprintf(stderr, "write failed\n");
                        return 1;
                    }
                }
            }
        }

        n_tokens_done += tokens.size();
        if ((s + 1) % 64 == 0 || s + 1 == seqs.size()) {
            const double elapsed = 1e-6 * (ggml_time_us() - t_start);
            fprintf(stderr, "%zu/%zu sequences, %zu tokens, %.0f tokens/s\n", s + 1, seqs.size(), n_tokens_done, n_tokens_done / elapsed);
        }
    }
    if (fout)  { fclose(fout); }
    if (ftopk) { fclose(ftopk); }

    FILE * fmeta = fopen(((write_taps ? out_path : topk_path) + ".json").c_str(), "w");
    if (!fmeta) { fprintf(stderr, "cannot write the index\n"); return 1; }
    fprintf(fmeta, "{\"n_embd\": %d, \"dtype\": \"float16\", \"taps\": [", n_embd);
    for (size_t k = 0; k < taps.size(); ++k) { fprintf(fmeta, "%s%u", k ? ", " : "", taps[k]); }
    fprintf(fmeta, "], \"first_sequence\": %ld, \"lengths\": [", start);
    for (size_t s = 0; s < seqs.size(); ++s) { fprintf(fmeta, "%s%zu", s ? ", " : "", seqs[s].size()); }
    fprintf(fmeta, "], \"first\": [");
    for (size_t s = 0; s < seqs.size(); ++s) { fprintf(fmeta, "%s%u", s ? ", " : "", first[s]); }
    fprintf(fmeta, "], \"topk\": %d}\n", topk);
    fclose(fmeta);

    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
