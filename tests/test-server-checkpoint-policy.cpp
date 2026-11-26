#include "../tools/server/server-checkpoint-policy.h"

#include <iostream>
#include <stdexcept>

static int assertions = 0;

static void check(bool condition, const char * message) {
    ++assertions;
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static server_checkpoint_candidate cp(int64_t pos, size_t bytes = 88, bool semantic = false,
        uint32_t hits = 0, bool device = false) {
    return {pos, device ? size_t(1) : bytes, bytes, semantic, hits, device};
}

static void apply(std::vector<server_checkpoint_candidate> & old, server_checkpoint_candidate incoming,
        const server_checkpoint_admission & plan) {
    check(plan.admit, "expected admission");
    std::vector<server_checkpoint_candidate> result;
    for (size_t i = 0; i < old.size(); ++i) {
        if (std::find(plan.evict.begin(), plan.evict.end(), i) == plan.evict.end()) {
            result.push_back(old[i]);
        }
    }
    result.push_back(incoming);
    std::sort(result.begin(), result.end(), [](const auto & a, const auto & b) { return a.pos < b.pos; });
    old = std::move(result);
}

static bool contains(const std::vector<server_checkpoint_candidate> & list, int64_t pos) {
    return std::any_of(list.begin(), list.end(), [&](const auto & c) { return c.pos == pos; });
}

int main() {
    try {
        server_checkpoint_anchor_tracker tracker;
        tracker.on_user(61860);
        tracker.on_assistant(61948);
        // Tool responses deliberately do not call on_user.
        tracker.on_assistant(84468);
        tracker.on_assistant(85349);
        tracker.on_assistant(86111);
        auto anchors = tracker.finish();
        check(anchors.turn == 61948 && anchors.edit == 85349 && anchors.latest == 86111,
              "tool cycles must not move the real turn");
        tracker.on_user(90000);
        check(tracker.finish().turn == 90000, "new user resets turn even before assistant");
        tracker.on_assistant(90030);
        check(tracker.finish().turn == 90030, "new reasoning turn");

        // Recorded November 26 loss: preserve the real turn when successive
        // tool cycles grow the same long request.
        std::vector<server_checkpoint_candidate> old{
            cp(14981), cp(61948, 88, true), cp(84468), cp(85349)};
        auto incoming = cp(86111, 88, true);
        apply(old, incoming, server_plan_checkpoint(old, incoming, 14981, anchors, 4, 384, 8192));
        check(old.size() == 4 && contains(old, 14981) && contains(old, 61948) &&
              contains(old, 85349) && contains(old, 86111), "system and turn/edit coverage");
        auto plan = server_plan_checkpoint(old, cp(86112), 14981, anchors, 4, 384, 8192);
        check(!plan.admit && plan.evict.empty(), "reject redundant incoming tail without mutation");
        incoming = cp(90500, 88, true);
        apply(old, incoming, server_plan_checkpoint(old, incoming, 14981,
              {61948, 86111, 90500}, 4, 384, 8192));
        check(contains(old, 61948) && contains(old, 86111) && contains(old, 90500),
              "old turn survives continuation");

        // Exact refresh replaces one payload, not two. Memory is host capacity.
        old = {cp(100, 100), cp(200, 100), cp(300, 100)};
        plan = server_plan_checkpoint(old, cp(300, 100), 100, {200, -1, 300}, 3, 300, 8192);
        check(plan.admit && plan.evict == std::vector<size_t>{2}, "refresh must not evict unrelated state");
        plan = server_plan_checkpoint(old, cp(300, 301), 100, {200, -1, 300}, 3, 300, 8192);
        check(!plan.admit && plan.evict.empty(), "oversized refresh is nondestructive");
        plan = server_plan_checkpoint(old, cp(400, 101), 100, {200, 300, 400}, 4, 400, 8192);
        check(!plan.admit && plan.evict.empty(), "cannot exceed host budget to add fourth anchor");
        plan = server_plan_checkpoint(old, cp(400, 100), 100, {200, 300, 400}, 4, 400, 8192);
        check(plan.admit && plan.evict.empty(), "fourth exact fit");
        check(!server_plan_checkpoint(old, cp(400), 100, {}, 0, 384, 8192).admit, "disabled count");
        check(!server_plan_checkpoint(old, cp(400), 100, {}, 4, 0, 8192).admit, "disabled budget");

        // Preserve measured reuse signals on equal semantic/gap candidates.
        old = {cp(100, 88, true, 0), cp(200, 88, true, 8)};
        plan = server_plan_checkpoint(old, cp(300, 88, true, 1), -1, {}, 2, 384, 100);
        check(plan.admit && plan.evict == std::vector<size_t>{0}, "reuse history raises retention value");

        // The physical context has only one device-backed snapshot.
        old = {cp(100, 88), cp(200, 88, true, 0, true)};
        plan = server_plan_checkpoint(old, cp(300, 88, true, 0, true),
              100, {200, -1, 300}, 4, 384, 8192);
        check(!plan.admit && plan.evict.empty(), "retain protected device turn rather than overwriting it");
        plan = server_plan_checkpoint(old, cp(300, 88, true, 0, true),
              100, {100, -1, 300}, 4, 384, 8192);
        check(plan.admit && plan.evict == std::vector<size_t>{1}, "device replacement invalidates old envelope");
        // A failed later capacity decision must not commit an earlier device removal.
        plan = server_plan_checkpoint(old, cp(300, 88, false, 0, true),
              100, {100, -1, 300}, 1, 384, 8192);
        check(!plan.admit && plan.evict.empty(), "failed multi-constraint plan is atomic");

        std::cout << assertions << " checkpoint policy assertions passed\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
