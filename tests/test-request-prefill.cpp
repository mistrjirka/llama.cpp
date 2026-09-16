#include "../tools/server/server-request-prefill.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

static int cases = 0;
static void check(bool value, const char * what) {
    if (!value) { throw std::runtime_error(what); }
}
template<class F> static void rejected(F action, const char * name) {
    bool threw = false;
    try { action(); } catch (const std::runtime_error &) { threw = true; }
    check(threw, name);
    ++cases;
}
int main() {
    try {
        // The same complete suffix can be assembled in arbitrary transport chunks.
        // Only one complete submission is admitted, independent of those chunks.
        for (int32_t n : {1, 257, 1000, 100000}) {
            for (int32_t chunk : {1, 127, 1000, 4096}) {
                server_request_prefill p;
                p.begin(42, n);
                int32_t assembled = 0;
                while (assembled < n) { assembled += std::min(chunk, n - assembled); }
                p.validate_assembly(assembled);
                p.submit(assembled, 0, assembled);
                rejected([&] { p.submit(assembled, 0, assembled); }, "duplicate request dispatch accepted");
                p.reset();
                p.begin(43, 1);
                p.submit(1, 0, 1);
                check(p.id() == 43, "new request identity was not restored");
                ++cases;
            }
        }
        // Model-level chunks and checkpoint tails must remain inside one request.
        for (int32_t fragment : {1, 4, 128, 1000, 99999}) {
            server_request_prefill p;
            p.begin(5, 100000);
            rejected([&] { p.validate_assembly(fragment); }, "partial assembly accepted");
            rejected([&] { p.submit(100000, 0, fragment); }, "shortened request dispatch accepted");
            rejected([&] { p.submit(100000, 1, 100000); }, "offset request dispatch accepted");
            // Rejected assembly/geometry did not consume the valid submission.
            p.submit(100000, 0, 100000);
        }
        server_request_prefill p;
        rejected([&] { p.begin(1, 0); }, "empty suffix accepted");
        rejected([&] { p.begin(1, -1); }, "negative suffix accepted");
        rejected([&] { p.submit(1, 0, 1); }, "unadmitted submission accepted");
        p.begin(1, 1000); // cached prefix length is deliberately absent
        rejected([&] { p.begin(2, 1000); }, "two requests mixed in one batch");
        rejected([&] { p.validate_assembly(101000); }, "cached prefix added to uncached work");
        p.submit(1000, 0, 1000);
        std::cout << "PASS " << cases << " request-boundary/dispatch cases\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
