#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "common.h"

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <cerrno>
#include <chrono>

#if defined(_WIN32)
    #include <windows.h>
    #ifndef PATH_MAX
        #define PATH_MAX MAX_PATH
    #endif
    #include <io.h>
#else
    #include <sys/types.h>
    #include <unistd.h>
    #if defined(__linux__)
        #include <sys/sendfile.h>
        #include <sys/syscall.h>
        #include <fcntl.h>
    #endif
#endif

enum split_operation : uint8_t {
    OP_NONE,
    OP_SPLIT,
    OP_MERGE,
};

enum split_mode : uint8_t {
    MODE_NONE,
    MODE_TENSOR,
    MODE_SIZE,
};

struct split_params {
    split_operation operation = OP_NONE;
    split_mode mode = MODE_NONE;
    size_t n_bytes_split = 0;
    int n_split_tensors = 128;
    std::string input;
    std::string output;
    bool no_tensor_first_split = false;
    bool dry_run = false;
};

static void split_print_usage(const char * executable) {
    const split_params default_params;
    printf("\n");
    printf("usage: %s [options] GGUF_IN GGUF_OUT\n", executable);
    printf("\n");
    printf("Apply a GGUF operation on IN to OUT.");
    printf("\n");
    printf("options:\n");
    printf("  -h, --help              show this help message and exit\n");
    printf("  --version               show version and build info\n");
    printf("  --split                 split GGUF to multiple GGUF (enabled by default)\n");
    printf("  --merge                 merge multiple GGUF to a single GGUF\n");
    printf("  --split-max-tensors     max tensors in each split (default: %d)\n", default_params.n_split_tensors);
    printf("  --split-max-size N(M|G) max size per split\n");
    printf("  --no-tensor-first-split do not add tensors to the first split (disabled by default)\n");
    printf("  --dry-run               only print out a split plan and exit, without writing any new files\n");
    printf("\n");
}

// return convert string, for example "128M" or "4G" to number of bytes
static size_t split_str_to_n_bytes(std::string str) {
    size_t n_bytes = 0;
    int n;
    if (str.back() == 'M') {
        sscanf(str.c_str(), "%d", &n);
        n_bytes = (size_t)n * 1000 * 1000; // megabytes
    } else if (str.back() == 'G') {
        sscanf(str.c_str(), "%d", &n);
        n_bytes = (size_t)n * 1000 * 1000 * 1000; // gigabytes
    } else {
        throw std::invalid_argument("error: supported units are M (megabytes) or G (gigabytes), but got: " + std::string(1, str.back()));
    }
    if (n <= 0) {
        throw std::invalid_argument("error: size must be a positive value");
    }
    return n_bytes;
}

static void split_params_parse_ex(int argc, const char ** argv, split_params & params) {
    std::string arg;
    const std::string arg_prefix = "--";
    bool invalid_param = false;

    int arg_idx = 1;
    for (; arg_idx < argc && strncmp(argv[arg_idx], "--", 2) == 0; arg_idx++) {
        arg = argv[arg_idx];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        bool arg_found = false;
        if (arg == "-h" || arg == "--help") {
            split_print_usage(argv[0]);
            exit(0);
        } else if (arg == "--version") {
            fprintf(stderr, "version: %d (%s)\n", LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
            fprintf(stderr, "built with %s for %s\n", LLAMA_COMPILER, LLAMA_BUILD_TARGET);
            exit(0);
        } else if (arg == "--dry-run") {
            arg_found = true;
            params.dry_run = true;
        } else if (arg == "--no-tensor-first-split") {
            arg_found = true;
            params.no_tensor_first_split = true;
        } else if (arg == "--merge") {
            arg_found = true;
            if (params.operation != OP_NONE && params.operation != OP_MERGE) {
                throw std::invalid_argument("error: either --split or --merge can be specified, but not both");
            }
            params.operation = OP_MERGE;
        } else if (arg == "--split") {
            arg_found = true;
            if (params.operation != OP_NONE && params.operation != OP_SPLIT) {
                throw std::invalid_argument("error: either --split or --merge can be specified, but not both");
            }
            params.operation = OP_SPLIT;
        } else if (arg == "--split-max-tensors") {
            if (++arg_idx >= argc) {
                invalid_param = true;
                break;
            }
            arg_found = true;
            if (params.mode != MODE_NONE && params.mode != MODE_TENSOR) {
                throw std::invalid_argument("error: either --split-max-tensors or --split-max-size can be specified, but not both");
            }
            params.mode = MODE_TENSOR;
            params.n_split_tensors = atoi(argv[arg_idx]);
        } else if (arg == "--split-max-size") {
            if (++arg_idx >= argc) {
                invalid_param = true;
                break;
            }
            arg_found = true;
            if (params.mode != MODE_NONE && params.mode != MODE_SIZE) {
                throw std::invalid_argument("error: either --split-max-tensors or --split-max-size can be specified, but not both");
            }
            params.mode = MODE_SIZE;
            params.n_bytes_split = split_str_to_n_bytes(argv[arg_idx]);
        }

        if (!arg_found) {
            throw std::invalid_argument("error: unknown argument: " + arg);
        }
    }

    // the operation is split if not specified
    if (params.operation == OP_NONE) {
        params.operation = OP_SPLIT;
    }
    // the split mode is by tensor if not specified
    if (params.mode == MODE_NONE) {
        params.mode = MODE_TENSOR;
    }

    if (invalid_param) {
        throw std::invalid_argument("error: invalid parameter for argument: " + arg);
    }

    if (argc - arg_idx != 2) {
        throw std::invalid_argument("error: bad arguments");
    }

    params.input = argv[arg_idx++];
    params.output = argv[arg_idx++];
}

static bool split_params_parse(int argc, const char ** argv, split_params & params) {
    bool result = true;
    try {
        split_params_parse_ex(argc, argv, params);
    }
    catch (const std::invalid_argument & ex) {
        fprintf(stderr, "%s\n", ex.what());
        split_print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return result;
}

static void zeros(std::ofstream & file, size_t n) {
    if (n == 0) {
        return;
    }
    static constexpr size_t ZERO_CHUNK = 64u * 1024;
    std::vector<char> buffer(std::min(ZERO_CHUNK, n), 0);
    size_t remaining = n;
    while (remaining > 0) {
        const size_t to_write = std::min(remaining, buffer.size());
        file.write(buffer.data(), to_write);
        remaining -= to_write;
    }
}

static void zeros(FILE * file, size_t n) {
    if (n == 0) {
        return;
    }
    static constexpr size_t ZERO_CHUNK = 64u * 1024;
    std::vector<char> buffer(std::min(ZERO_CHUNK, n), 0);
    size_t remaining = n;
    while (remaining > 0) {
        const size_t to_write = std::min(remaining, buffer.size());
        if (std::fwrite(buffer.data(), 1, to_write, file) != to_write) {
            fprintf(stderr, "error: failed to write padding bytes: %s\n", std::strerror(errno));
            exit(EXIT_FAILURE);
        }
        remaining -= to_write;
    }
}

static int64_t size_to_i64(size_t value) {
    if (value > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        fprintf(stderr, "error: file offset too large: %zu\n", value);
        exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

static void file_seek_checked(FILE * file, int64_t offset, int whence) {
#if defined(_WIN32)
    if (_fseeki64(file, offset, whence) != 0) {
#else
    if (fseeko(file, static_cast<off_t>(offset), whence) != 0) {
#endif
        fprintf(stderr, "error: seek failed: %s\n", std::strerror(errno));
        exit(EXIT_FAILURE);
    }
}

static void file_seek_checked(FILE * file, size_t offset, int whence = SEEK_SET) {
    file_seek_checked(file, size_to_i64(offset), whence);
}

static int64_t file_tell_checked(FILE * file) {
#if defined(_WIN32)
    const __int64 pos = _ftelli64(file);
    if (pos == -1) {
#else
    const off_t pos = ftello(file);
    if (pos == static_cast<off_t>(-1)) {
#endif
        fprintf(stderr, "error: tell failed: %s\n", std::strerror(errno));
        exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(pos);
}

static size_t file_get_size(FILE * file) {
    const int64_t cur = file_tell_checked(file);
    file_seek_checked(file, (int64_t)0, SEEK_END);
    const int64_t size = file_tell_checked(file);
    file_seek_checked(file, cur, SEEK_SET);
    if (size < 0) {
        fprintf(stderr, "error: invalid file size\n");
        exit(EXIT_FAILURE);
    }
    return static_cast<size_t>(size);
}

struct split_strategy {
    const split_params params;
    std::ifstream & f_input;
    struct gguf_context * ctx_gguf;
    struct ggml_context * ctx_meta = NULL;
    const int n_tensors;

    // one ctx_out per one output file
    std::vector<struct gguf_context *> ctx_outs;

    // temporary buffer for reading in tensor data
    std::vector<uint8_t> read_buf;

    split_strategy(const split_params & params,
            std::ifstream & f_input,
            struct gguf_context * ctx_gguf,
            struct ggml_context * ctx_meta) :
        params(params),
        f_input(f_input),
        ctx_gguf(ctx_gguf),
        ctx_meta(ctx_meta),
        n_tensors(gguf_get_n_tensors(ctx_gguf)) {

        // because we need to know list of tensors for each file in advance, we will build all the ctx_out for all output splits
        int i_split = -1;
        struct gguf_context * ctx_out = NULL;
        auto new_ctx_out = [&](bool allow_no_tensors) {
            i_split++;
            if (ctx_out != NULL) {
                if (gguf_get_n_tensors(ctx_out) == 0 && !allow_no_tensors) {
                    fprintf(stderr, "error: one of splits have 0 tensors. Maybe size or tensors limit is too small\n");
                    exit(EXIT_FAILURE);
                }
                ctx_outs.push_back(ctx_out);
            }
            ctx_out = gguf_init_empty();
            // Save all metadata in first split only
            if (i_split == 0) {
                gguf_set_kv(ctx_out, ctx_gguf);
            }
            gguf_set_val_u16(ctx_out, LLM_KV_SPLIT_NO, i_split);
            gguf_set_val_u16(ctx_out, LLM_KV_SPLIT_COUNT, 0); // placeholder
            gguf_set_val_i32(ctx_out, LLM_KV_SPLIT_TENSORS_COUNT, n_tensors);
        };

        // initialize ctx_out for the first split
        new_ctx_out(false);

        // skip first split if no_tensor_first_split is set
        if (params.no_tensor_first_split) {
            new_ctx_out(true);
        }

        // process tensors one by one
        size_t curr_tensors_size = 0; // current size by counting only tensors size (without metadata)
        for (int i = 0; i < n_tensors; ++i) {
            struct ggml_tensor * t = ggml_get_tensor(ctx_meta, gguf_get_tensor_name(ctx_gguf, i));
            // calculate the "imaginary" size = the current size + next tensor size
            size_t n_bytes = GGML_PAD(ggml_nbytes(t), GGUF_DEFAULT_ALIGNMENT);
            size_t next_tensors_size = curr_tensors_size + n_bytes;
            if (should_split(i, next_tensors_size)) {
                new_ctx_out(false);
                curr_tensors_size = n_bytes;
            } else {
                curr_tensors_size = next_tensors_size;
            }
            gguf_add_tensor(ctx_out, t);
        }

        // push the last ctx_out
        ctx_outs.push_back(ctx_out);

        // set the correct n_split for all ctx_out
        for (auto & ctx : ctx_outs) {
            gguf_set_val_u16(ctx, LLM_KV_SPLIT_COUNT, ctx_outs.size());
        }
    }

    ~split_strategy() {
        for (auto & ctx_out : ctx_outs) {
            gguf_free(ctx_out);
        }
    }

    bool should_split(int i_tensor, size_t next_size) {
        if (params.mode == MODE_SIZE) {
            // split by max size per file
            return next_size > params.n_bytes_split;
        } else if (params.mode == MODE_TENSOR) {
            // split by number of tensors per file
            return i_tensor > 0 && i_tensor < n_tensors && i_tensor % params.n_split_tensors == 0;
        }
        // should never happen
        GGML_ABORT("invalid mode");
    }

    void print_info() {
        printf("n_split: %zu\n", ctx_outs.size());
        int i_split = 0;
        for (auto & ctx_out : ctx_outs) {
            // re-calculate the real gguf size for each split (= metadata size + total size of all tensors)
            size_t total_size = gguf_get_meta_size(ctx_out);
            for (int i = 0; i < gguf_get_n_tensors(ctx_out); ++i) {
                struct ggml_tensor * t = ggml_get_tensor(ctx_meta, gguf_get_tensor_name(ctx_out, i));
                total_size += ggml_nbytes(t);
            }
            total_size = total_size / 1000 / 1000; // convert to megabytes
            printf("split %05d: n_tensors = %" PRIi64 ", total_size = %zuM\n", i_split + 1, gguf_get_n_tensors(ctx_out), total_size);
            i_split++;
        }
    }

    void write() {
        int i_split = 0;
        int n_split = ctx_outs.size();
        for (auto & ctx_out : ctx_outs) {
            // construct file path
            char split_path[PATH_MAX] = {0};
            llama_split_path(split_path, sizeof(split_path), params.output.c_str(), i_split, n_split);

            // open the output file
            printf("Writing file %s ... ", split_path);
            fflush(stdout);
            std::ofstream fout = std::ofstream(split_path, std::ios::binary);
            fout.exceptions(std::ofstream::failbit); // fail fast on write errors

            // write metadata
            std::vector<uint8_t> data(gguf_get_meta_size(ctx_out));
            gguf_get_meta_data(ctx_out, data.data());
            fout.write((const char *)data.data(), data.size());

            // write tensors
            for (int i = 0; i < gguf_get_n_tensors(ctx_out); ++i) {
                // read tensor meta and prepare buffer
                const char * t_name = gguf_get_tensor_name(ctx_out, i);
                struct ggml_tensor * t = ggml_get_tensor(ctx_meta, t_name);
                auto n_bytes = ggml_nbytes(t);
                read_buf.resize(n_bytes);

                // calculate offset
                auto i_tensor_in = gguf_find_tensor(ctx_gguf, t_name); // idx of tensor in the input file
                auto offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i_tensor_in);

                // copy tensor from input to output file
                copy_file_to_file(f_input, fout, offset, n_bytes);
                zeros(fout, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);
            }

            printf("done\n");
            // close the file
            fout.close();
            i_split++;
        }
    }

    void copy_file_to_file(std::ifstream & f_in, std::ofstream & f_out, const size_t in_offset, const size_t len) {
        // TODO: detect OS and use copy_file_range() here for better performance
        if (read_buf.size() < len) {
            read_buf.resize(len);
        }
        f_in.seekg(in_offset);
        f_in.read((char *)read_buf.data(), len);
        f_out.write((const char *)read_buf.data(), len);
    }
};

static constexpr size_t GGUF_MERGE_COPY_CHUNK = 8u * 1024 * 1024;

#if defined(__linux__)
static size_t gguf_try_fast_copy(FILE * f_input,
        FILE * f_output,
        size_t in_offset,
        size_t out_offset,
        size_t n_bytes) {
    if (n_bytes == 0) {
        return 0;
    }

    const int fd_in = fileno(f_input);
    const int fd_out = fileno(f_output);
    if (fd_in == -1 || fd_out == -1) {
        return 0;
    }

    // Advise the kernel that we are going to read/write sequentially
    posix_fadvise(fd_in, in_offset, n_bytes, POSIX_FADV_SEQUENTIAL);
    posix_fadvise(fd_out, out_offset, n_bytes, POSIX_FADV_SEQUENTIAL);

    off_t in_off = static_cast<off_t>(size_to_i64(in_offset));
    off_t out_off = static_cast<off_t>(size_to_i64(out_offset));
    size_t remaining = n_bytes;

    while (remaining > 0) {
        // Cap the copy size to avoid overflow issues with ssize_t
        // 1GB is a safe chunk size
        const size_t chunk = std::min<size_t>(remaining, 1u << 30);
        
        ssize_t res = -1;

        // Try copy_file_range first (supports reflink)
#ifdef __NR_copy_file_range
        res = syscall(__NR_copy_file_range, fd_in, &in_off, fd_out, &out_off, chunk, 0);
#endif
        
        if (res > 0) {
            remaining -= res;
            continue;
        }

        // Fallback logging for copy_file_range
        static bool logged_cfr_fallback = false;
        if (!logged_cfr_fallback) {
#ifdef __NR_copy_file_range
             fprintf(stderr, "%s: copy_file_range failed: %s, falling back to sendfile\n", __func__, std::strerror(errno));
#endif
             logged_cfr_fallback = true;
        }

        // Fallback to sendfile
        // sendfile writes to current offset of fd_out, so we must seek
        if (lseek(fd_out, out_off, SEEK_SET) == (off_t)-1) {
            break;
        }

        res = ::sendfile(fd_out, fd_in, &in_off, chunk);
        if (res > 0) {
            out_off += res;
            remaining -= res;
            continue;
        }

        // Fallback logging for sendfile
        static bool logged_sf_fallback = false;
        if (!logged_sf_fallback) {
             fprintf(stderr, "%s: sendfile failed: %s, falling back to read/write\n", __func__, std::strerror(errno));
             logged_sf_fallback = true;
        }

        if (res == -1) {
            if (errno == EINTR || errno == EAGAIN) {
                continue;
            }
            break;
        }
        if (res == 0) {
            break;
        }
    }

    const size_t fast_copied = n_bytes - remaining;
    if (fast_copied > 0) {
        // sync stdio offsets with the descriptor positions
        file_seek_checked(f_input, in_offset + fast_copied);
        file_seek_checked(f_output, out_offset + fast_copied);
    }

    return fast_copied;
}
#else
static size_t gguf_try_fast_copy(FILE *, FILE *, size_t, size_t, size_t) {
    return 0;
}
#endif

struct merge_state {
    const split_params & params;
    struct gguf_context * ctx_out = NULL;
    std::vector<gguf_context *> ctx_ggufs;
    std::vector<ggml_context *> ctx_metas;
    char split_prefix[PATH_MAX] = {0};
    int n_split = 1;
    int total_tensors = 0;

    merge_state(const split_params & params) : params(params) {}
};

static void gguf_merge_release_split_contexts(merge_state & state) {
    for (auto & ctx : state.ctx_ggufs) {
        if (ctx != NULL) {
            gguf_free(ctx);
            ctx = NULL;
        }
    }
    for (auto & ctx : state.ctx_metas) {
        if (ctx != NULL) {
            ggml_free(ctx);
            ctx = NULL;
        }
    }
    state.ctx_ggufs.clear();
    state.ctx_metas.clear();
}

static void gguf_merge_fail(merge_state & state) {
    gguf_merge_release_split_contexts(state);
    if (state.ctx_out != NULL) {
        gguf_free(state.ctx_out);
        state.ctx_out = NULL;
    }
}

static void gguf_merge_copy_payload(FILE * f_input,
        FILE * f_output,
        size_t in_offset,
        size_t out_offset,
        size_t n_bytes,
        std::vector<uint8_t> & buffer) {

    if (n_bytes == 0 || f_output == NULL) {
        return;
    }

    const size_t chunk = std::min(n_bytes, GGUF_MERGE_COPY_CHUNK);
    if (buffer.size() < chunk) {
        buffer.resize(chunk);
    }

    if (std::fflush(f_output) != 0) {
        fprintf(stderr, "%s: failed to flush output stream: %s\n", __func__, std::strerror(errno));
        exit(EXIT_FAILURE);
    }
    file_seek_checked(f_output, out_offset);

    const size_t fast_copied = gguf_try_fast_copy(f_input, f_output, in_offset, out_offset, n_bytes);

    size_t remaining = n_bytes - fast_copied;
    size_t read_pos = in_offset + fast_copied;
    size_t write_pos = out_offset + fast_copied;

    file_seek_checked(f_input, read_pos);
    file_seek_checked(f_output, write_pos);

    while (remaining > 0) {
        const size_t to_copy = std::min(remaining, buffer.size());
        if (std::fread(buffer.data(), 1, to_copy, f_input) != to_copy) {
            fprintf(stderr, "%s: failed to read input payload: %s\n", __func__, std::strerror(errno));
            exit(EXIT_FAILURE);
        }
        if (std::fwrite(buffer.data(), 1, to_copy, f_output) != to_copy) {
            fprintf(stderr, "%s: failed to write output payload: %s\n", __func__, std::strerror(errno));
            exit(EXIT_FAILURE);
        }
        remaining -= to_copy;
    }
}

static void gguf_merge_first_pass(merge_state & state) {
    char split_path[PATH_MAX] = {0};
    strncpy(split_path, state.params.input.c_str(), sizeof(split_path) - 1);

    for (int i_split = 0; i_split < state.n_split; ++i_split) {
        struct ggml_context * ctx_meta = NULL;

        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &ctx_meta,
        };

        if (i_split > 0) {
            llama_split_path(split_path, sizeof(split_path), state.split_prefix, i_split, state.n_split);
        }
        fprintf(stderr, "%s: reading metadata %s ...", __func__, split_path);

        auto * ctx_gguf = gguf_init_from_file(split_path, params);
        if (!ctx_gguf) {
            fprintf(stderr, "\n%s:  failed to load input GGUF from %s\n", __func__, state.params.input.c_str());
            ggml_free(ctx_meta);
            gguf_merge_fail(state);
            exit(EXIT_FAILURE);
        }
        state.ctx_ggufs.push_back(ctx_gguf);
        state.ctx_metas.push_back(ctx_meta);

        if (i_split == 0) {
            auto key_n_split = gguf_find_key(ctx_gguf, LLM_KV_SPLIT_COUNT);
            if (key_n_split < 0) {
                fprintf(stderr,
                        "\n%s: input file does not contain %s metadata\n",
                        __func__,
                        LLM_KV_SPLIT_COUNT);
                gguf_merge_fail(state);
                exit(EXIT_FAILURE);
            }

            state.n_split = gguf_get_val_u16(ctx_gguf, key_n_split);
            if (state.n_split < 1) {
                fprintf(stderr,
                        "\n%s: input file does not contain a valid split count %d\n",
                        __func__,
                        state.n_split);
                gguf_merge_fail(state);
                exit(EXIT_FAILURE);
            }

            if (!llama_split_prefix(state.split_prefix, sizeof(state.split_prefix), split_path, i_split, state.n_split)) {
                fprintf(stderr, "\n%s: unexpected input file name: %s i_split=%d n_split=%d\n",
                        __func__, split_path, i_split, state.n_split);
                gguf_merge_fail(state);
                exit(EXIT_FAILURE);
            }

            gguf_set_val_u16(ctx_gguf, LLM_KV_SPLIT_COUNT, 0);
            gguf_set_kv(state.ctx_out, ctx_gguf);
        }

        auto n_tensors = gguf_get_n_tensors(ctx_gguf);
        for (int i_tensor = 0; i_tensor < n_tensors; i_tensor++) {
            const char * t_name = gguf_get_tensor_name(ctx_gguf, i_tensor);
            struct ggml_tensor * t = ggml_get_tensor(ctx_meta, t_name);
            gguf_add_tensor(state.ctx_out, t);
        }
        state.total_tensors += n_tensors;

        fprintf(stderr, "\033[3Ddone\n");
    }
}

static size_t gguf_merge_second_pass(merge_state & state, FILE * fout, std::vector<uint8_t> & buffer) {
    char split_path[PATH_MAX] = {0};
    size_t out_offset = fout != NULL ? gguf_get_meta_size(state.ctx_out) : 0;
    size_t total_bytes = 0;

    for (int i_split = 0; i_split < state.n_split; ++i_split) {
        auto * ctx_gguf = state.ctx_ggufs[i_split];
        auto * ctx_meta = state.ctx_metas[i_split];

        if (fout == NULL) {
            gguf_free(ctx_gguf);
            ggml_free(ctx_meta);
            state.ctx_ggufs[i_split] = NULL;
            state.ctx_metas[i_split] = NULL;
            continue;
        }

        llama_split_path(split_path, sizeof(split_path), state.split_prefix, i_split, state.n_split);
        FILE * f_input = ggml_fopen(split_path, "rb");
        if (f_input == NULL) {
            fprintf(stderr, "%s:  failed to open input GGUF from %s\n", __func__, split_path);
            gguf_merge_release_split_contexts(state);
            std::fclose(fout);
            gguf_free(state.ctx_out);
            exit(EXIT_FAILURE);
        }

        fprintf(stderr, "%s: writing tensors %s ...", __func__, split_path);

        const size_t file_size = file_get_size(f_input);
        const size_t data_offset = gguf_get_data_offset(ctx_gguf);
        if (file_size < data_offset) {
            fprintf(stderr, "\n%s: invalid data offset in %s\n", __func__, split_path);
            std::fclose(f_input);
            gguf_merge_release_split_contexts(state);
            std::fclose(fout);
            gguf_free(state.ctx_out);
            exit(EXIT_FAILURE);
        }

        const size_t payload = file_size - data_offset;
        gguf_merge_copy_payload(f_input, fout, data_offset, out_offset, payload, buffer);
        out_offset += payload;
        total_bytes += payload;

        gguf_free(ctx_gguf);
        ggml_free(ctx_meta);
        state.ctx_ggufs[i_split] = NULL;
        state.ctx_metas[i_split] = NULL;
        std::fclose(f_input);
        fprintf(stderr, "\033[3Ddone\n");
    }

    state.ctx_ggufs.clear();
    state.ctx_metas.clear();
    return total_bytes;
}

static void gguf_split(const split_params & split_params) {
    struct ggml_context * ctx_meta = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_meta,
    };

    std::ifstream f_input(split_params.input.c_str(), std::ios::binary);
    if (!f_input.is_open()) {
        fprintf(stderr, "%s:  failed to open input GGUF from %s\n", __func__, split_params.input.c_str());
        exit(EXIT_FAILURE);
    }

    auto * ctx_gguf = gguf_init_from_file(split_params.input.c_str(), params);
    if (!ctx_gguf) {
        fprintf(stderr, "%s:  failed to load input GGUF from %s\n", __func__, split_params.input.c_str());
        exit(EXIT_FAILURE);
    }

    // prepare the strategy
    split_strategy strategy(split_params, f_input, ctx_gguf, ctx_meta);
    int n_split = strategy.ctx_outs.size();
    strategy.print_info();

    if (!split_params.dry_run) {
        // write all output splits
        strategy.write();
    }

    // done, clean up
    gguf_free(ctx_gguf);
    f_input.close();

    fprintf(stderr, "%s: %d gguf split written with a total of %d tensors.\n",
            __func__, n_split, strategy.n_tensors);
}

static void gguf_merge(const split_params & split_params) {
    fprintf(stderr, "%s: %s -> %s\n",
            __func__, split_params.input.c_str(),
            split_params.output.c_str());
    // avoid overwriting existing output file
    if (std::ifstream(split_params.output.c_str())) {
        fprintf(stderr, "%s: output file %s already exists\n", __func__, split_params.output.c_str());
        exit(EXIT_FAILURE);
    }

    merge_state state(split_params);
    state.ctx_out = gguf_init_empty();
    if (state.ctx_out == NULL) {
        fprintf(stderr, "%s: failed to allocate merge context\n", __func__);
        exit(EXIT_FAILURE);
    }

    gguf_merge_first_pass(state);

    FILE * fout = NULL;
    const size_t meta_size = gguf_get_meta_size(state.ctx_out);
    if (!split_params.dry_run) {
        fout = ggml_fopen(split_params.output.c_str(), "wb+");
        if (fout == NULL) {
            fprintf(stderr, "%s: failed to open output file %s\n", __func__, split_params.output.c_str());
            gguf_merge_fail(state);
            exit(EXIT_FAILURE);
        }
        zeros(fout, meta_size);
    }

    std::vector<uint8_t> copy_buffer;
    auto t_start = std::chrono::high_resolution_clock::now();
    size_t total_bytes = gguf_merge_second_pass(state, fout, copy_buffer);
    auto t_end = std::chrono::high_resolution_clock::now();

    if (!split_params.dry_run) {
        if (std::fflush(fout) != 0) {
            fprintf(stderr, "%s: failed to flush output stream: %s\n", __func__, std::strerror(errno));
            std::fclose(fout);
            gguf_merge_fail(state);
            exit(EXIT_FAILURE);
        }
        file_seek_checked(fout, (int64_t)0, SEEK_SET);
        std::vector<uint8_t> data(meta_size);
        gguf_get_meta_data(state.ctx_out, data.data());
        if (std::fwrite(data.data(), 1, data.size(), fout) != data.size()) {
            fprintf(stderr, "%s: failed to finalize metadata: %s\n", __func__, std::strerror(errno));
            std::fclose(fout);
            gguf_merge_fail(state);
            exit(EXIT_FAILURE);
        }
        std::fclose(fout);
        
        double duration = std::chrono::duration<double>(t_end - t_start).count();
        fprintf(stderr, "%s: merged %.2f MB in %.2f s (%.2f MB/s)\n", 
                __func__, total_bytes / 1024.0 / 1024.0, duration, (total_bytes / 1024.0 / 1024.0) / duration);
    }

    gguf_free(state.ctx_out);

    fprintf(stderr, "%s: %s merged from %d split with %d tensors.\n",
            __func__, split_params.output.c_str(), state.n_split, state.total_tensors);
}

int main(int argc, const char ** argv) {
    split_params params;
    split_params_parse(argc, argv, params);

    switch (params.operation) {
        case OP_SPLIT: gguf_split(params);
            break;
        case OP_MERGE: gguf_merge(params);
            break;
        default: split_print_usage(argv[0]);
            exit(EXIT_FAILURE);
    }

    return 0;
}
