#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

// Retention metadata only. Token/KV compatibility is checked by the caller's
// existing exact-prefix and state-restore logic.
struct server_checkpoint_anchors {
    int64_t turn   = -1;
    int64_t edit   = -1;
    int64_t latest = -1;

    bool contains(int64_t pos) const {
        return pos >= 0 && (pos == turn || pos == edit || pos == latest);
    }
};

// A tool result is not a new human turn, even when rendered with a user header.
struct server_checkpoint_anchor_tracker {
    server_checkpoint_anchors anchors;
    int64_t user = -1;

    void on_user(int64_t pos) {
        user = pos;
        anchors.turn = -1;
    }

    void on_assistant(int64_t pos) {
        anchors.edit = anchors.latest;
        anchors.latest = pos;
        if (user >= 0 && anchors.turn < 0) {
            anchors.turn = pos;
        }
    }

    server_checkpoint_anchors finish() const {
        auto out = anchors;
        if (out.turn < 0) {
            out.turn = user;
        }
        return out;
    }
};

struct server_checkpoint_candidate {
    int64_t pos;
    size_t host_bytes;
    size_t logical_bytes;
    bool semantic;
    uint32_t hits;
    bool device;
};

struct server_checkpoint_admission {
    bool admit = false;
    std::vector<size_t> evict;
};

// Plan on metadata, including the proposed capture. Never modify the live list
// until admission succeeds. Device snapshots share one context-owned storage
// slot; host capacity and logical payload size are deliberately distinct.
inline server_checkpoint_admission server_plan_checkpoint(
        const std::vector<server_checkpoint_candidate> & existing,
        server_checkpoint_candidate incoming,
        int64_t system, const server_checkpoint_anchors & anchors,
        size_t max_count, size_t max_bytes, int64_t min_step) {
    if (!max_count || !max_bytes || incoming.host_bytes > max_bytes || incoming.pos < 0) {
        return {};
    }

    struct entry {
        server_checkpoint_candidate c;
        size_t index;
    };
    const size_t new_index = existing.size();
    std::vector<entry> keep;
    server_checkpoint_admission plan;
    for (size_t i = 0; i < existing.size(); ++i) {
        const auto & c = existing[i];
        if (c.pos == incoming.pos) {
            plan.evict.push_back(i);
            incoming.semantic |= c.semantic;
            incoming.hits = std::max(incoming.hits, c.hits);
        } else {
            keep.push_back({c, i});
        }
    }
    keep.push_back({incoming, new_index});
    std::stable_sort(keep.begin(), keep.end(), [](const entry & a, const entry & b) {
        return a.c.pos < b.c.pos;
    });

    const int64_t horizon = std::max<int64_t>({1, incoming.pos, anchors.latest, keep.back().c.pos});
    const int64_t semantic_span = std::max<int64_t>(1, min_step);
    for (;;) {
        size_t remaining = max_bytes;
        bool host_fits = true;
        size_t devices = 0;
        for (const auto & e : keep) {
            devices += e.c.device;
            if (e.c.host_bytes > remaining) {
                host_fits = false;
                remaining = 0;
            } else {
                remaining -= e.c.host_bytes;
            }
        }
        if (keep.size() <= max_count && host_fits && devices <= 1) {
            plan.admit = true;
            return plan;
        }

        std::vector<int> priority(keep.size(), 0);
        const int64_t targets[] = {anchors.latest, anchors.edit, anchors.turn, system};
        // Preserve the deepest recoverable prefix at or before each boundary,
        // in descending importance: system, turn, edit, latest.
        for (int role = 0; role < 4; ++role) {
            if (targets[role] < 0) {
                continue;
            }
            for (size_t i = keep.size(); i > 0; --i) {
                if (keep[i - 1].c.pos <= targets[role]) {
                    priority[i - 1] = std::max(priority[i - 1], role + 1);
                    break;
                }
            }
        }

        size_t victim = keep.size();
        long double victim_value = 0;
        for (size_t i = 0; i < keep.size(); ++i) {
            const auto & c = keep[i].c;
            if (devices > 1 && !c.device) {
                continue;
            }
            const int64_t prev = i ? keep[i - 1].c.pos : 0;
            const int64_t next = i + 1 < keep.size() ? keep[i + 1].c.pos : horizon;
            const auto left = std::max<int64_t>(1, c.pos - prev);
            const auto right = std::max<int64_t>(1, next - c.pos);
            // Marginal replay saved, with concentrated semantic reuse and
            // measured restore hits. Weight by logical bytes even on GPU.
            long double value = (long double) left * right;
            if (c.semantic) {
                value += (long double) left * semantic_span * 8.0L;
            }
            value *= 1.0L + 4.0L * c.hits;
            value /= std::max<size_t>(1, std::max(c.host_bytes, c.logical_bytes));
            if (victim == keep.size() || priority[i] < priority[victim] ||
                    (priority[i] == priority[victim] && value < victim_value)) {
                victim = i;
                victim_value = value;
            }
        }

        if (victim == keep.size() || keep[victim].index == new_index) {
            // Discard the whole plan, including any tentative duplicate/device
            // removals. Rejected capture must leave all older state intact.
            return {};
        }
        plan.evict.push_back(keep[victim].index);
        keep.erase(keep.begin() + victim);
    }
}
