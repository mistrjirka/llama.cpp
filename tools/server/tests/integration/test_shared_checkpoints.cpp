// Link against server-task.cpp and the server/common libraries.
#include "server-task.h"
#include <cassert>
#include <cstdio>
#include <type_traits>

int main() {
    constexpr size_t mib = 1024 * 1024;
    server_prompt active;
    for (int i = 0; i < 3; ++i) {
        auto checkpoint = std::make_shared<common_prompt_checkpoint>();
        checkpoint->n_tokens = 1000 * (i + 1);
        checkpoint->pos_min = checkpoint->pos_max = checkpoint->n_tokens - 1;
        checkpoint->data_tgt.assign(mib, uint8_t(i + 1));
        checkpoint->data_spec.assign(32, uint8_t(i + 4));
        active.checkpoints.push_back(std::move(checkpoint));
    }
    static_assert(std::is_const<typename decltype(active.checkpoints)::value_type::element_type>::value,
                  "published checkpoint state must be immutable");
    auto shared = active.clone();
    auto copied = active.clone(false);
    auto source = active.checkpoints.begin();
    auto clone = copied.checkpoints.begin();
    for (const auto & checkpoint : shared.checkpoints) {
        assert(checkpoint == *source);
        assert(checkpoint != *clone);
        assert(checkpoint->data_tgt == (*clone)->data_tgt);
        assert(checkpoint->data_spec == (*clone)->data_spec);
        ++source;
        ++clone;
    }
    server_prompt_cache cache(16, 0);
    cache.share_checkpoints = true;
    cache.states.push_back({ active.clone(), {}, false, 0, 512 });
    const size_t one = cache.size();
    assert(one > 3 * mib);
    assert(cache.size(&active) == one);
    cache.states.push_back({ active.clone(), {}, false, 0, 512 });
    assert(cache.size() == one + 512);
    cache.states.pop_front();
    assert(cache.size() == one);
    const size_t copy_projection = cache.size(&copied);
    assert(copy_projection == one + (one - 512));
    cache.share_checkpoints = false;
    assert(cache.size(&active) == copy_projection);
    cache.states.push_back({ std::move(copied), {}, false, 0, 512 });
    assert(cache.size() == copy_projection + 512);
    cache.states.clear();
    assert(cache.size() == 0);
    assert(shared.checkpoints.front()->data_tgt.front() == 1);
    const std::weak_ptr<const common_prompt_checkpoint> lifetime = active.checkpoints.front();
    active.clear();
    assert(!lifetime.expired());
    shared.clear();
    assert(lifetime.expired());
    std::puts("PASS: immutable sharing, exact deep-copy control, unique cache accounting, incoming-reference admission, and final-reference release");
}
