#include "speculative.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>

// No model or GPU is required. In particular, -1 means no override, not no draft.
int main() {
    constexpr int32_t imax = std::numeric_limits<int32_t>::max();
    int cases = 0;
    for (int32_t configured : std::array<int32_t, 6>{-1, 0, 1, 3, 16, imax}) {
        for (int32_t requested : std::array<int32_t, 8>{-2, -1, 0, 1, 2, 3, 16, imax}) {
            common_speculative_draft_params p{};
            p.n_max = requested;
            const int32_t got = p.max_draft_tokens(configured);
            const int32_t expected = configured <= 0 ? 0 :
                requested < 0 ? configured : requested < configured ? requested : configured;
            if (got != expected) {
                std::fprintf(stderr, "limit mismatch configured=%d requested=%d got=%d expected=%d\n",
                             configured, requested, got, expected);
                return 1;
            }
            ++cases;
        }
    }
    common_speculative_draft_params defaults{};
    if (defaults.max_draft_tokens(3) != 3) return 1;
    std::printf("MTP budget: %d boundary cases and default -1 override passed\n", cases);
    return 0;
}
