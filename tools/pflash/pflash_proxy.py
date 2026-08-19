#!/usr/bin/env python3
"""Optional cache-stable PFlash proxy for OpenAI-compatible chat requests.

AUTO is cold-long only: a session whose first request is short remains byte-for-
byte pass-through for its lifetime, preserving llama.cpp prefix-cache reuse.
Cold long histories are compressed once against the current query and frozen.
Later user turns can recover omitted history; agent tool loops use cheap lexical
recovery from the newest tail. Every recovered tail augmentation is frozen and
never rewritten, so transformed request N remains a prefix of request N+1.

PFlash is approximate and disabled by default. The target server itself is not
modified by this proxy.
"""
import argparse
import hashlib
import json
import os
import struct
import subprocess
import tempfile
import threading
import time
import urllib.request
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


class PFlash:
    def __init__(self, binpath: str, gguf: str, gpu: int, anchor_hits: int | None, anchor_radius: int | None, edge_chunks: tuple[int, int] | None):
        self.lock = threading.Lock()
        self.r, w = os.pipe()
        env = os.environ.copy()
        env["DFLASH_FP_USE_BSA"] = "0"
        env["DFLASH_FP_ALPHA"] = "0.85"
        # Freeze uses Lucebox's normal anchor heuristic; recovery can disable
        # it because exact query/body anchor matching is already represented by
        # the frozen omitted store and otherwise creates a large retention floor.
        if anchor_hits is None:
            env.pop("PFLASH_COMPRESS_MAX_ANCHOR_HITS", None)
        else:
            env["PFLASH_COMPRESS_MAX_ANCHOR_HITS"] = str(anchor_hits)
        if anchor_radius is None:
            env.pop("PFLASH_COMPRESS_ANCHOR_RADIUS", None)
        else:
            env["PFLASH_COMPRESS_ANCHOR_RADIUS"] = str(anchor_radius)
        if edge_chunks is None:
            env.pop("DFLASH_COMPRESS_HEAD_CHUNKS", None)
            env.pop("DFLASH_COMPRESS_TAIL_CHUNKS", None)
        else:
            env["DFLASH_COMPRESS_HEAD_CHUNKS"] = str(edge_chunks[0])
            env["DFLASH_COMPRESS_TAIL_CHUNKS"] = str(edge_chunks[1])
        self.profile = (anchor_hits, anchor_radius, edge_chunks)
        self.p = subprocess.Popen(
            [binpath, gguf, f"--stream-fd={w}", f"--gpu={gpu}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            pass_fds=(w,),
            text=True,
            bufsize=1,
            env=env,
        )
        os.close(w)
        self.log_tail = deque(maxlen=128)
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("pflash daemon died during init")
            self.log_tail.append(line.rstrip())
            if "[pflash-daemon] ready" in line:
                break
        # The daemon writes per-call diagnostics to stdout. Drain continuously so
        # a long-lived proxy can never deadlock on a full subprocess pipe.
        self._drain_thread = threading.Thread(target=self._drain_stdout, daemon=True)
        self._drain_thread.start()

    def _drain_stdout(self):
        try:
            for line in self.p.stdout:
                self.log_tail.append(line.rstrip())
        except Exception:
            pass

    def compress_ids(self, ids: list[int], ratio: float, lookahead: int, chunk: int, pool: int):
        with self.lock:
            fd, path = tempfile.mkstemp(suffix=".bin")
            os.close(fd)
            try:
                with open(path, "wb") as f:
                    f.write(struct.pack("<I", len(ids)))
                    f.write(struct.pack(f"<{len(ids)}i", *ids))
                t0 = time.perf_counter()
                self.p.stdin.write(f"compress {round(ratio * 1000)} {lookahead} {chunk} {pool} {path}\n")
                self.p.stdin.flush()
                out: list[int] = []
                while True:
                    b = b""
                    while len(b) < 4:
                        x = os.read(self.r, 4 - len(b))
                        if not x:
                            raise RuntimeError("pflash stream eof")
                        b += x
                    v = struct.unpack("<i", b)[0]
                    if v == -1:
                        break
                    out.append(v)
                return out, time.perf_counter() - t0
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def close(self):
        try:
            if self.p.poll() is None:
                self.p.terminate()
                try:
                    self.p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.p.kill()
                    self.p.wait()
        finally:
            try:
                os.close(self.r)
            except OSError:
                pass


def recover_subsequence(full: list[int], kept: list[int]) -> list[int]:
    out: list[int] = []
    j = 0
    for i, token in enumerate(full):
        if j < len(kept) and token == kept[j]:
            out.append(i)
            j += 1
    if j != len(kept):
        raise RuntimeError(f"PFlash subsequence recovery failed: {j}/{len(kept)}")
    return out


def group_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    out: list[tuple[int, int]] = []
    a = b = indices[0]
    for x in indices[1:]:
        if x == b + 1:
            b = x
        else:
            out.append((a, b))
            a = b = x
    out.append((a, b))
    return out


def content_text(message: dict[str, Any]) -> str:
    c = message.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out: list[str] = []
        for part in c:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                out.append(part["text"])
        return "\n".join(out)
    return ""


def message_fingerprint(message: dict[str, Any]) -> str:
    raw = json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()


def session_key(body: dict[str, Any], explicit: str | None = None) -> str:
    if explicit:
        return explicit
    msgs = body.get("messages") or []
    first_user = next((m for m in msgs if m.get("role") == "user"), {})
    systems = [m for m in msgs if m.get("role") == "system"]
    seed = {
        "model": body.get("model"),
        "system": systems,
        "first_user": first_user,
    }
    raw = json.dumps(seed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()[:24]


@dataclass
class FrozenMessage:
    index: int
    fingerprint: str
    original: str
    compressed: str
    omitted: str
    original_tokens: int
    kept_tokens: int


@dataclass
class TailOverride:
    fingerprint: str
    content: Any


@dataclass
class Session:
    initialized: bool = False
    auto_passthrough_started: bool = False
    frozen: dict[int, FrozenMessage] = field(default_factory=dict)
    tail_overrides: dict[int, TailOverride] = field(default_factory=dict)
    recovered_snippets: set[str] = field(default_factory=set)
    freeze_query_hash: str = ""
    created_at: float = field(default_factory=time.time)


class Global:
    pass


S = Global()


def ensure_pflash(anchor_hits: int | None, anchor_radius: int | None = None, edge_chunks: tuple[int, int] | None = None) -> tuple[PFlash, float]:
    profile = (anchor_hits, anchor_radius, edge_chunks)
    with S.pf_init_lock:
        if S.pf is not None and S.pf.profile == profile:
            return S.pf, 0.0
        t0 = time.perf_counter()
        if S.pf is not None:
            S.pf.close()
            S.pf = None
        S.pf = PFlash(S.binpath, S.gguf, S.gpu, anchor_hits, anchor_radius, edge_chunks)
        return S.pf, time.perf_counter() - t0


def current_query(messages, last_user, exclude_indices=None):
    excluded = exclude_indices or set()
    start_i = max(0, len(messages) - S.query_hot_messages)
    pieces = []
    for i in range(start_i, len(messages)):
        if i in excluded:
            continue
        m = messages[i]
        c = content_text(m)
        tc = m.get("tool_calls")
        if not c and not tc:
            continue
        pieces.append("[" + str(m.get("role", "")) + "]\n")
        if tc:
            pieces.append(json.dumps(tc, ensure_ascii=False, separators=(",", ":")) + "\n")
        if c:
            pieces.append(c + "\n")
    if not pieces and last_user >= 0:
        pieces.append(content_text(messages[last_user]))
    text = "".join(pieces)
    enc = S.tok.encode(text, add_special_tokens=False)
    if len(enc.ids) > S.query_context_tokens:
        text = S.tok.decode(enc.ids[-S.query_context_tokens:], skip_special_tokens=False)
    return text

def _snippets(flat: str, offsets: list[tuple[int, int]], indices: list[int], lo: int, hi: int) -> str:
    parts: list[str] = []
    for ta, tb in group_runs(indices):
        a = max(lo, offsets[ta][0])
        b = min(hi, offsets[tb][1])
        if b > a:
            parts.append(flat[a:b])
    return "\n...\n".join(parts)


def score_content_segmented(content: str, query: str, ratio: float):
    """Score one candidate message in bounded PFlash windows.

    This is used only when the whole cold scorer input would exceed the
    co-resident drafter memory budget. Every window sees the same query tail;
    selected token indices are unioned back into the original message.
    """
    enc = S.tok.encode(content, add_special_tokens=False)
    ids, offsets = enc.ids, enc.offsets
    qids = S.tok.encode("\n" + query, add_special_tokens=False).ids
    body_budget = S.scorer_max_tokens - len(qids)
    if body_budget < 512:
        return None, {"fallback": "scorer-query-too-large", "query_tokens": len(qids)}
    step = max(1, body_budget - S.scorer_overlap)
    pf, switch_s = ensure_pflash(None, None, None)
    selected: set[int] = set()
    score_s = 0.0
    calls = 0
    raw_kept = 0
    starts = list(range(0, len(ids), step)) or [0]
    for start in starts:
        end = min(len(ids), start + body_budget)
        chunk = ids[start:end]
        if not chunk:
            continue
        segment = chunk + qids
        kept, dt = pf.compress_ids(segment, ratio, S.lookahead, S.chunk, S.pool)
        score_s += dt
        calls += 1
        raw_kept += len(kept)
        # Empty/tiny output means scorer failure (usually memory), not a valid
        # aggressive selection. Fail safe to verbatim instead of dropping data.
        min_plausible = max(8, int(len(chunk) * 0.03))
        if len(kept) < min_plausible:
            return None, {
                "fallback": "implausibly-small-segment-selection",
                "segment_start": start,
                "segment_tokens": len(segment),
                "raw_kept": len(kept),
                "min_plausible": min_plausible,
                "compress_s": score_s,
                "scorer_switch_s": switch_s,
                "scorer_calls": calls,
            }
        local = recover_subsequence(segment, kept)
        selected.update(start + i for i in local if i < len(chunk))
        if end >= len(ids):
            break
    # Preserve modest message edges independent of scorer segmentation.
    selected.update(range(min(16, len(ids))))
    selected.update(range(max(0, len(ids) - 16), len(ids)))
    local = sorted(selected)
    omitted = sorted(set(range(len(ids))) - selected)
    compressed = _snippets(content, offsets, local, 0, len(content))
    omitted_text = _snippets(content, offsets, omitted, 0, len(content))
    return (compressed, omitted_text, len(ids), len(local), len(omitted)), {
        "compress_s": score_s,
        "scorer_switch_s": switch_s,
        "scorer_calls": calls,
        "scorer_segmented": True,
        "scorer_raw_kept": raw_kept,
        "scorer_max_tokens": S.scorer_max_tokens,
    }


def freeze_history(body: dict[str, Any], sess: Session, ratio: float):
    messages = body.get("messages") or []
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)

    candidates: list[tuple[int, int, int]] = []
    parts: list[str] = []
    cursor = 0
    hot_cut = max(0, len(messages) - S.hot_messages)
    for i, m in enumerate(messages):
        if m.get("role") not in ("assistant", "tool", "toolResult"):
            continue
        if not isinstance(m.get("content"), str):
            continue
        c = m["content"]
        if not c or len(c) < 256 or i >= hot_cut:
            continue
        if parts:
            parts.append("\n")
            cursor += 1
        a = cursor
        parts.append(c)
        cursor += len(c)
        candidates.append((i, a, cursor))

    if not candidates:
        return body, {"skipped": "no_candidates"}

    q = current_query(messages, last_user, {mi for mi, _, _ in candidates})
    parts.append("\n")
    cursor += 1
    qa = cursor
    parts.append(q)
    cursor += len(q)
    qb = cursor
    flat = "".join(parts)
    enc = S.tok.encode(flat, add_special_tokens=False)
    ids, offsets = enc.ids, enc.offsets
    if len(ids) < S.min_tokens:
        return body, {"skipped": "short", "tokens": len(ids)}

    # The 3060 Ti scorer can coexist with the target around ~20k scorer tokens,
    # but a 40k+ monolithic drafter KV allocation cannot. Segment only the scorer
    # pass; the target still receives one coherent compressed history.
    if len(ids) > S.scorer_max_tokens:
        out_messages = [dict(m) for m in messages]
        new_frozen: dict[int, FrozenMessage] = {}
        original_tokens = frozen_tokens = omitted_tokens = 0
        score_s = switch_s = 0.0
        scorer_calls = 0
        for mi, _, _ in candidates:
            res, st = score_content_segmented(messages[mi]["content"], q, ratio)
            score_s += st.get("compress_s", 0.0)
            switch_s += st.get("scorer_switch_s", 0.0)
            scorer_calls += st.get("scorer_calls", 0)
            if res is None:
                return body, {**st, "scorer_tokens": len(ids), "segmented": True}
            compressed_text, omitted_text, n_orig, n_keep, n_omit = res
            if not compressed_text:
                compressed_text = messages[mi]["content"]
                omitted_text = ""
                n_keep, n_omit = n_orig, 0
            fm = FrozenMessage(mi, message_fingerprint(messages[mi]), messages[mi]["content"], compressed_text, omitted_text, n_orig, n_keep)
            new_frozen[mi] = fm
            out_messages[mi]["content"] = compressed_text
            original_tokens += n_orig; frozen_tokens += n_keep; omitted_tokens += n_omit
        sess.frozen = new_frozen
        sess.initialized = True
        sess.freeze_query_hash = hashlib.sha256(q.encode("utf-8", "surrogatepass")).hexdigest()[:16]
        new_body = dict(body); new_body["messages"] = out_messages
        return new_body, {
            "phase": "freeze", "segmented": True, "scorer_tokens": len(ids),
            "candidate_tokens": original_tokens, "frozen_tokens": frozen_tokens,
            "omitted_tokens": omitted_tokens, "candidate_ratio": frozen_tokens / max(1, original_tokens),
            "compress_s": score_s, "scorer_switch_s": switch_s,
            "scorer_calls": scorer_calls, "frozen_messages": len(new_frozen),
        }

    pf, switch_s = ensure_pflash(None, None, None)
    kept, score_s = pf.compress_ids(ids, ratio, S.lookahead, S.chunk, S.pool)
    min_plausible = max(32, int(len(ids) * 0.05))
    if len(kept) < min_plausible:
        return body, {
            "fallback": "implausibly-small-freeze-selection",
            "scorer_tokens": len(ids),
            "raw_kept": len(kept),
            "min_plausible": min_plausible,
            "compress_s": score_s,
            "scorer_switch_s": switch_s,
        }
    selected = set(recover_subsequence(ids, kept))
    # Query is scorer-only context and must not be emitted.
    for ti, (a, b) in enumerate(offsets):
        if a < qb and b > qa:
            selected.add(ti)

    out_messages = [dict(m) for m in messages]
    frozen_tokens = original_tokens = omitted_tokens = 0
    new_frozen: dict[int, FrozenMessage] = {}
    for mi, ca, cb in candidates:
        tis = [ti for ti, (a, b) in enumerate(offsets) if a < cb and b > ca]
        if not tis:
            continue
        local = sorted((set(tis) & selected) | set(tis[:16]) | set(tis[-16:]))
        omitted = sorted(set(tis) - set(local))
        compressed_text = _snippets(flat, offsets, local, ca, cb)
        omitted_text = _snippets(flat, offsets, omitted, ca, cb)
        if not compressed_text:
            compressed_text = messages[mi]["content"]
            omitted_text = ""
            local = tis
            omitted = []
        fm = FrozenMessage(
            index=mi,
            fingerprint=message_fingerprint(messages[mi]),
            original=messages[mi]["content"],
            compressed=compressed_text,
            omitted=omitted_text,
            original_tokens=len(tis),
            kept_tokens=len(local),
        )
        new_frozen[mi] = fm
        out_messages[mi]["content"] = compressed_text
        original_tokens += len(tis)
        frozen_tokens += len(local)
        omitted_tokens += len(omitted)

    sess.frozen = new_frozen
    sess.initialized = True
    sess.freeze_query_hash = hashlib.sha256(q.encode("utf-8", "surrogatepass")).hexdigest()[:16]
    new_body = dict(body)
    new_body["messages"] = out_messages
    return new_body, {
        "phase": "freeze",
        "scorer_tokens": len(ids),
        "raw_kept": len(kept),
        "candidate_tokens": original_tokens,
        "frozen_tokens": frozen_tokens,
        "omitted_tokens": omitted_tokens,
        "candidate_ratio": frozen_tokens / max(1, original_tokens),
        "compress_s": score_s,
        "scorer_switch_s": switch_s,
        "frozen_messages": len(new_frozen),
    }


def apply_stable_overrides(messages: list[dict[str, Any]], sess: Session):
    out = [dict(m) for m in messages]
    frozen_hits = frozen_misses = tail_hits = tail_misses = 0
    for idx, fm in sess.frozen.items():
        if idx < len(messages) and message_fingerprint(messages[idx]) == fm.fingerprint:
            out[idx]["content"] = fm.compressed
            frozen_hits += 1
        else:
            frozen_misses += 1
    for idx, ov in sess.tail_overrides.items():
        if idx < len(messages) and message_fingerprint(messages[idx]) == ov.fingerprint:
            out[idx]["content"] = ov.content
            tail_hits += 1
        else:
            tail_misses += 1
    return out, frozen_hits, frozen_misses, tail_hits, tail_misses


def inject_recovery(content: Any, recovered: str, *, prepend: bool):
    if not recovered:
        return content
    block = f"<<<PFLASH_RECOVERED_CONTEXT>>>\n{recovered}\n<<<PFLASH_RECOVERY_END>>>"
    if isinstance(content, str):
        return (block + "\n" + content) if prepend else (content + "\n" + block)
    if isinstance(content, list):
        part = {"type": "text", "text": block + "\n"}
        return ([part] + content) if prepend else (content + [part])
    text = str(content or "")
    return (block + "\n" + text) if prepend else (text + "\n" + block)


def choose_recovery_anchor(messages: list[dict[str, Any]], last_user: int, sess: Session) -> tuple[int, bool]:
    """Pick a tail message whose augmentation will become stable on the next turn.

    A genuine latest user turn can carry recovery before its query.  Agent loops
    often keep the same initial user message forever and append assistant/tool
    messages; rewriting that old user turn destroys prefix-cache reuse.  In that
    case append recovery to the newest string-content tail message instead.
    """
    if last_user >= 0 and last_user == len(messages) - 1 and last_user not in sess.tail_overrides:
        return last_user, True
    for idx in range(len(messages) - 1, -1, -1):
        if idx in sess.frozen or idx in sess.tail_overrides:
            continue
        m = messages[idx]
        if isinstance(m.get("content"), (str, list)):
            return idx, False
    return last_user, True


def _meaningful_anchor_piece(token_id: int) -> bool:
    piece = S.tok.decode([token_id], skip_special_tokens=False).strip()
    alnum = "".join(ch for ch in piece if ch.isalnum())
    return len(alnum) >= 3


def lexical_recovery_indices(body_text: str, query_text: str):
    """Find small, high-confidence exact neighborhoods in omitted history.

    This is deliberately conservative: only n-grams/unigrams that occur a few
    times in the omitted store are anchors. Generic repeated prose therefore
    does not consume recovery budget, while file names, symbols, service IDs,
    error fragments, hashes, ports, etc. typically do.
    """
    benc = S.tok.encode(body_text, add_special_tokens=False)
    qenc = S.tok.encode(query_text, add_special_tokens=False)
    bids, qids = benc.ids, qenc.ids
    selected: set[int] = set()
    anchors: list[dict[str, Any]] = []
    seen_positions: set[int] = set()
    max_hits = S.recovery_lexical_max_hits
    max_anchors = S.recovery_lexical_max_anchors

    for n in (8, 7, 6, 5, 4, 3, 2):
        if len(qids) < n or len(bids) < n:
            continue
        posmap: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for i in range(len(bids) - n + 1):
            k = tuple(bids[i:i+n])
            v = posmap[k]
            if len(v) <= max_hits:
                v.append(i)
        for qi in range(len(qids) - n + 1):
            key = tuple(qids[qi:qi+n])
            hits = posmap.get(key, [])
            if not (1 <= len(hits) <= max_hits):
                continue
            if not any(_meaningful_anchor_piece(t) for t in key):
                continue
            for pos in hits:
                if pos in seen_positions:
                    continue
                seen_positions.add(pos)
                lo = max(0, pos - S.recovery_lexical_window)
                hi = min(len(bids), pos + n + S.recovery_lexical_window)
                selected.update(range(lo, hi))
                anchors.append({"n": n, "pos": pos, "hits": len(hits)})
                if len(anchors) >= max_anchors:
                    return bids, benc.offsets, sorted(selected), anchors

    # Rare meaningful unigrams catch names such as "cobalt" even when the
    # surrounding wording is paraphrased and no 2+-token sequence survives.
    counts = Counter(bids)
    positions: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(bids):
        if counts[t] <= max_hits:
            positions[t].append(i)
    for t in qids:
        hits = positions.get(t, [])
        if not (1 <= len(hits) <= max_hits) or not _meaningful_anchor_piece(t):
            continue
        for pos in hits:
            if pos in seen_positions:
                continue
            seen_positions.add(pos)
            lo = max(0, pos - S.recovery_lexical_window)
            hi = min(len(bids), pos + 1 + S.recovery_lexical_window)
            selected.update(range(lo, hi))
            anchors.append({"n": 1, "pos": pos, "hits": len(hits)})
            if len(anchors) >= max_anchors:
                break
        if len(anchors) >= max_anchors:
            break
    return bids, benc.offsets, sorted(selected), anchors


def recover_omitted(messages: list[dict[str, Any]], sess: Session, last_user: int, ratio: float):
    """Recover only information that is absent from the frozen prefix.

    The omitted store is the sole recovery corpus.  This is important for cache
    stability and token efficiency: searching/re-emitting the original message
    would duplicate text that is already present in ``FrozenMessage.compressed``.

    Exact/rare lexical anchors are handled first because they are both cheap and
    high confidence (symbols, file names, IDs, literal error fragments, etc.).
    If no such anchor exists, policy chooses either a quality-first verbatim
    restore or a query-conditioned PFlash pass over omitted text only.
    """
    q = current_query(messages, last_user, set(sess.frozen.keys()))
    # On a genuine new user turn, exact anchors should come from that user query.
    # During an agent tool loop the original user task can be many turns old; use
    # the newest tail message instead so stale task words do not trigger the same
    # recovery on every continuation.
    if last_user >= 0 and last_user == len(messages) - 1:
        lexical_query = content_text(messages[last_user])
    else:
        lexical_query = content_text(messages[-1]) if messages else q
    frozen_items = [(idx, fm) for idx, fm in sorted(sess.frozen.items()) if fm.omitted]
    if not frozen_items:
        return "", {"recovery_skipped": "nothing_omitted"}

    omitted_body = "\n".join(fm.omitted for _, fm in frozen_items)
    omitted_enc = S.tok.encode(omitted_body, add_special_tokens=False)
    omitted_ids = omitted_enc.ids

    # Fast, bounded recovery for an exact/rare pointer in the current query.
    lex_ids, lex_offsets, lex_selected, lex_anchors = lexical_recovery_indices(omitted_body, lexical_query)
    if lex_anchors and lex_selected:
        recovered = _snippets(omitted_body, lex_offsets, lex_selected, 0, len(omitted_body))
        return recovered, {
            "recovery_mode": "lexical",
            "recovery_tokens_in": len(lex_ids),
            "recovery_tokens_kept": len(lex_selected),
            "lexical_anchors": len(lex_anchors),
            "lexical_tokens": len(lex_selected),
            "recovery_s": 0.0,
        }

    is_new_user_turn = last_user >= 0 and last_user == len(messages) - 1
    if not lex_anchors:
        if S.semantic_recovery == "never" or (S.semantic_recovery == "user" and not is_new_user_turn):
            return "", {
                "recovery_mode": "skipped-semantic-policy",
                "recovery_tokens_in": len(omitted_ids),
                "recovery_tokens_kept": 0,
                "lexical_anchors": 0,
                "recovery_s": 0.0,
            }

    if S.recovery_no_anchor_fallback == "full":
        # Quality-first policy: if the query gives us no reliable pointer, make
        # every omitted token reachable rather than trusting an uncertain prune.
        return omitted_body, {
            "recovery_mode": "verbatim-no-strong-anchor",
            "recovery_tokens_in": len(omitted_ids),
            "recovery_tokens_kept": len(omitted_ids),
            "lexical_anchors": 0,
            "recovery_s": 0.0,
        }

    if len(omitted_ids) < S.recovery_min_tokens:
        return omitted_body, {
            "recovery_mode": "verbatim-small",
            "recovery_tokens_in": len(omitted_ids),
            "recovery_tokens_kept": len(omitted_ids),
            "recovery_s": 0.0,
        }

    # Query-conditioned fallback.  Deliberately use *no synthetic marker text*:
    # Lucebox scans repeated query/body n-grams as retrieval anchors, and repeated
    # PFLASH_* delimiters can accidentally force a large fraction of the corpus.
    flat = omitted_body + "\n" + q
    body_end = len(omitted_body)
    qa = body_end + 1
    qb = len(flat)
    enc = S.tok.encode(flat, add_special_tokens=False)
    ids, offsets = enc.ids, enc.offsets

    # Recovery already operates on the omitted store, so disable Lucebox's
    # forced head/tail/anchor neighborhoods.  Selection is then governed by the
    # query-conditioned scores and requested keep ratio instead of an anchor floor.
    pf, switch_s = ensure_pflash(0, 0, (0, 0))
    kept, score_s = pf.compress_ids(ids, ratio, S.lookahead, S.chunk, S.pool)
    if not kept:
        return omitted_body, {
            "recovery_mode": "verbatim-scorer-fallback",
            "recovery_tokens_in": len(omitted_ids),
            "recovery_tokens_kept": len(omitted_ids),
            "recovery_s": score_s,
            "recovery_scorer_switch_s": switch_s,
            "lexical_anchors": 0,
        }

    selected = set(recover_subsequence(ids, kept))
    # Never emit scorer-only query tokens.
    body_tis = [ti for ti, (a, b) in enumerate(offsets) if a < body_end and b > 0]
    local = sorted(ti for ti in body_tis if ti in selected)
    recovered = _snippets(flat, offsets, local, 0, body_end)
    if not recovered:
        # Recovery is a safety net; an empty reconstruction must not silently
        # make omitted information permanently unreachable.
        return omitted_body, {
            "recovery_mode": "verbatim-empty-reconstruction-fallback",
            "recovery_tokens_in": len(omitted_ids),
            "recovery_tokens_kept": len(omitted_ids),
            "recovery_s": score_s,
            "recovery_scorer_switch_s": switch_s,
            "lexical_anchors": 0,
        }

    return recovered, {
        "recovery_mode": "pflash",
        "recovery_tokens_in": len(omitted_ids),
        "recovery_scorer_tokens": len(ids),
        "recovery_raw_kept": len(kept),
        "recovery_tokens_kept": len(local),
        "recovery_s": score_s,
        "recovery_scorer_switch_s": switch_s,
        "recovery_anchor_hits": 0,
        "recovery_anchor_radius": 0,
        "recovery_head_chunks": 0,
        "recovery_tail_chunks": 0,
        "lexical_anchors": 0,
    }


def dedupe_recovery(sess: Session, recovered: str) -> tuple[str, int, int]:
    """Drop recovery spans that are already frozen into earlier tail messages."""
    if not recovered:
        return "", 0, 0
    parts = [p.strip() for p in recovered.split("\n...\n") if p.strip()]
    fresh = []
    dup = 0
    for part in parts:
        key = hashlib.sha256(part.encode("utf-8", "surrogatepass")).hexdigest()
        if key in sess.recovered_snippets:
            dup += 1
            continue
        sess.recovered_snippets.add(key)
        fresh.append(part)
    return "\n...\n".join(fresh), len(fresh), dup


def transform_body(body: dict[str, Any], explicit_session: str | None, mode: str, ratio: float, recovery_ratio: float):
    key = session_key(body, explicit_session)
    with S.sessions_lock:
        sess = S.sessions.setdefault(key, Session())

    messages = body.get("messages") or []
    trigger_text = "\n".join(content_text(m) for m in messages)
    trigger_tokens = len(S.tok.encode(trigger_text, add_special_tokens=False).ids)

    if mode == "off":
        return body, {"mode": "off", "passthrough": True, "session": key, "tokens": trigger_tokens}

    if not sess.initialized:
        if mode == "auto":
            # AUTO is deliberately cold-long only. Once we have observed a short
            # active conversation, do not retrofit compression when it later grows:
            # the server already owns a valuable stable prefix cache for that branch.
            if sess.auto_passthrough_started:
                return body, {"mode": "auto", "passthrough": True, "reason": "active-session", "session": key, "tokens": trigger_tokens}
            if trigger_tokens < S.threshold_tokens:
                sess.auto_passthrough_started = True
                return body, {"mode": "auto", "passthrough": True, "reason": "cold-short", "session": key, "tokens": trigger_tokens}
        out, stat = freeze_history(body, sess, ratio)
        if mode == "auto" and not sess.initialized:
            # Compression was not applicable/safe on the first request. Keep this
            # active session verbatim rather than attempting a disruptive later rewrite.
            sess.auto_passthrough_started = True
            stat["passthrough"] = True
            stat["reason"] = "cold-compression-not-applicable"
        stat.update({"mode": mode, "session": key, "trigger_tokens": trigger_tokens})
        return out, stat

    out_messages, fh, fm, th, tm = apply_stable_overrides(messages, sess)
    # If the frozen base no longer matches, fail safe: do not silently apply a stale compression.
    if fm:
        return body, {
            "mode": mode,
            "session": key,
            "fallback": "frozen-prefix-mismatch",
            "frozen_hits": fh,
            "frozen_misses": fm,
        }

    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    rec, recstat = recover_omitted(messages, sess, last_user, recovery_ratio) if last_user >= 0 else ("", {"recovery_skipped": "no_user"})
    if rec:
        rec, fresh_spans, duplicate_spans = dedupe_recovery(sess, rec)
        recstat["recovery_fresh_spans"] = fresh_spans
        recstat["recovery_duplicate_spans"] = duplicate_spans
    if rec and messages:
        anchor_idx, prepend = choose_recovery_anchor(messages, last_user, sess)
        if anchor_idx >= 0:
            original_anchor = messages[anchor_idx]
            transformed = inject_recovery(out_messages[anchor_idx].get("content"), rec, prepend=prepend)
            out_messages[anchor_idx]["content"] = transformed
            # Freeze this one new tail augmentation. Older augmented turns are never
            # rewritten, so each transformed request remains a prefix of the next.
            sess.tail_overrides[anchor_idx] = TailOverride(message_fingerprint(original_anchor), transformed)
            recstat["recovery_anchor_index"] = anchor_idx
            recstat["recovery_anchor_role"] = original_anchor.get("role")
            recstat["recovery_anchor_prepend"] = prepend

    new_body = dict(body)
    new_body["messages"] = out_messages
    prefix_material = json.dumps(out_messages[:-1], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    stat = {
        "mode": mode,
        "phase": "reuse+recovery",
        "session": key,
        "trigger_tokens": trigger_tokens,
        "frozen_hits": fh,
        "frozen_misses": fm,
        "tail_hits": th,
        "tail_misses": tm,
        "frozen_messages": len(sess.frozen),
        "tail_overrides": len(sess.tail_overrides),
        "stable_prefix_sha256": hashlib.sha256(prefix_material.encode("utf-8", "surrogatepass")).hexdigest(),
        **recstat,
    }
    return new_body, stat


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path == "/health":
            b = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        self._proxy_get()

    def _proxy_get(self):
        try:
            with urllib.request.urlopen(S.target + self.path, timeout=30) as r:
                b = r.read()
                self.send_response(r.status)
                self.send_header("Content-Type", r.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
        except Exception as e:
            self.send_error(502, str(e))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        original = self.rfile.read(n)
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            return self.forward(original)
        raw = original
        try:
            body = json.loads(raw)
            mode = self.headers.get("X-PFlash-Mode", S.mode).lower()
            if mode not in ("off", "auto", "frozen"):
                mode = S.mode
            ratio = min(1.0, max(0.01, float(self.headers.get("X-PFlash-Keep-Ratio", S.ratio))))
            rr = min(1.0, max(0.01, float(self.headers.get("X-PFlash-Recovery-Ratio", S.recovery_ratio))))
            sid = self.headers.get("X-PFlash-Session")
            with S.transform_lock:
                body2, stat = transform_body(body, sid, mode, ratio, rr)
            with open(S.log, "a") as f:
                f.write(json.dumps({"t": time.time(), **stat}, ensure_ascii=False) + "\n")
            raw = original if stat.get("passthrough") else json.dumps(body2, ensure_ascii=False, separators=(",", ":")).encode()
        except Exception as e:
            with open(S.log, "a") as f:
                f.write(json.dumps({"t": time.time(), "error": repr(e), "fallback": "verbatim"}) + "\n")
            raw = original
        self.forward(raw)

    def forward(self, raw: bytes):
        try:
            req = urllib.request.Request(
                S.target + self.path,
                data=raw,
                headers={"Content-Type": "application/json", "Authorization": self.headers.get("Authorization", "Bearer local")},
                method="POST",
            )
            r = urllib.request.urlopen(req, timeout=1200)
            self.send_response(r.status)
            ct = r.headers.get("Content-Type", "application/json")
            self.send_header("Content-Type", ct)
            self.send_header("Connection", "close")
            self.end_headers()
            if "text/event-stream" in ct:
                for b in r:
                    self.wfile.write(b)
                    self.wfile.flush()
            else:
                while True:
                    b = r.read(65536)
                    if not b:
                        break
                    self.wfile.write(b)
                    self.wfile.flush()
            r.close()
        except Exception as e:
            self.send_error(502, str(e))


def configure(a):
    S.target = a.target
    S.mode = a.mode
    S.threshold_tokens = max(1, a.threshold_tokens)
    S.ratio = a.ratio
    S.recovery_ratio = a.recovery_ratio
    S.min_tokens = max(1, a.min_tokens)
    S.recovery_min_tokens = max(1, a.recovery_min_tokens)
    S.hot_messages = max(1, a.hot_messages)
    S.lookahead = max(1, a.lookahead)
    S.chunk = max(1, a.chunk)
    S.pool = max(1, a.pool)
    S.log = a.log
    S.binpath = a.bin
    S.gguf = a.gguf
    S.gpu = a.gpu
    S.recovery_lexical_window = a.recovery_lexical_window
    S.recovery_lexical_max_hits = a.recovery_lexical_max_hits
    S.recovery_lexical_max_anchors = a.recovery_lexical_max_anchors
    S.recovery_no_anchor_fallback = a.recovery_no_anchor_fallback
    S.semantic_recovery = a.semantic_recovery
    S.query_hot_messages = max(1, a.query_hot_messages)
    S.query_context_tokens = max(8, a.query_context_tokens)
    S.scorer_max_tokens = max(1024, a.scorer_max_tokens)
    S.scorer_overlap = max(0, min(a.scorer_overlap, S.scorer_max_tokens // 4))
    S.pf = None
    S.pf_init_lock = threading.Lock()
    S.sessions: dict[str, Session] = {}
    S.sessions_lock = threading.Lock()
    S.transform_lock = threading.Lock()
    S.tok = Tokenizer.from_file(a.tokenizer)
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    Path(a.log).write_text("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=18321)
    ap.add_argument("--target", default="http://127.0.0.1:18320")
    ap.add_argument("--bin", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--mode", choices=["off", "auto", "frozen"], default="off")
    ap.add_argument("--threshold-tokens", type=int, default=32000)
    ap.add_argument("--ratio", type=float, default=0.70)
    ap.add_argument("--recovery-ratio", type=float, default=0.01)
    ap.add_argument("--recovery-lexical-window", type=int, default=24, help="token radius around rare exact recovery anchors")
    ap.add_argument("--recovery-lexical-max-hits", type=int, default=4, help="maximum body frequency for a lexical anchor")
    ap.add_argument("--recovery-lexical-max-anchors", type=int, default=8, help="maximum rare anchor occurrences recovered per turn")
    ap.add_argument("--recovery-no-anchor-fallback", choices=["full", "pflash"], default="pflash", help="quality-first fallback when no strong lexical anchor exists")
    ap.add_argument("--semantic-recovery", choices=["user", "always", "never"], default="user", help="when to run semantic PFlash recovery if no exact/rare lexical anchor exists")
    ap.add_argument("--query-hot-messages", type=int, default=4, help="recent non-frozen messages used as scorer query context")
    ap.add_argument("--query-context-tokens", type=int, default=2048, help="maximum scorer query-tail token budget")
    ap.add_argument("--scorer-max-tokens", type=int, default=22000, help="maximum PFlash tokens per scorer call; larger cold histories are segmented")
    ap.add_argument("--scorer-overlap", type=int, default=64, help="body-token overlap between segmented cold scorer windows")
    ap.add_argument("--min-tokens", type=int, default=8000)
    ap.add_argument("--recovery-min-tokens", type=int, default=512)
    ap.add_argument("--hot-messages", type=int, default=4)
    ap.add_argument("--lookahead", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--pool", type=int, default=13)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    configure(a)
    ThreadingHTTPServer(("127.0.0.1", a.listen), H).serve_forever()


if __name__ == "__main__":
    main()
