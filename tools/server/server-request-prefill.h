#pragma once

#include <cstdint>
#include <stdexcept>

// Admission belongs to a complete uncached request suffix. Computation begins
// after assembly, so transport/batch boundaries do not delimit weight reuse.
class server_request_prefill {
public:
    void reset() {
        request_id = -1;
        expected = 0;
        submitted = false;
    }

    void begin(int id, int32_t uncached_tokens) {
        require(!active(), "request-prefill admission already active");
        require(uncached_tokens > 0, "request-prefill needs a nonempty suffix");
        request_id = id;
        expected = uncached_tokens;
        submitted = false;
    }

    bool active() const { return expected != 0; }
    int id() const { return request_id; }

    void validate_assembly(int32_t assembled) const {
        if (active()) {
            require(assembled == expected, "request-prefill requires the complete uncached suffix");
        }
    }

    void submit(int32_t assembled, int32_t offset, int32_t count) {
        require(active(), "request-prefill was not admitted");
        validate_assembly(assembled);
        require(offset == 0 && count == expected, "request-prefill must be dispatched as one full suffix");
        require(!submitted, "request-prefill was already dispatched");
        submitted = true;
    }

private:
    static void require(bool condition, const char * message) {
        if (!condition) { throw std::runtime_error(message); }
    }
    int request_id = -1;
    int32_t expected = 0;
    bool submitted = false;
};
