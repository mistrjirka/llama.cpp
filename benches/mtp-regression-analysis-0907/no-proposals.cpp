// Diagnostic only: suppress proposals, retaining all normal target/draft maintenance.
// Never linked into or enabled in the production launcher.
#include <atomic>
#include <cstdio>
struct common_speculative;
static std::atomic<unsigned long> calls{0};
void common_speculative_draft(common_speculative *) { ++calls; }
__attribute__((destructor)) static void report() {
    std::fprintf(stderr, "MTP_NO_PROPOSALS_DIAGNOSTIC calls=%lu\n", calls.load());
}
