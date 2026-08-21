import os
import tempfile
import pytest
from utils import *

server = ServerPreset.tinyllama2()

class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0
    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.cache_ram = 100
    server.kv_unified = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


LONG_PROMPT = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
    "He met many creatures along the way including dragons and fairies "
    "and wizards who helped him on his noble quest to save the kingdom."
)


# idle slot cleared on launch should restore from cache-ram
def test_clear_and_restore():
    global server
    server.start()
    log = LogReader(server.log_path)

    # verify feature is enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    original_prompt_n = res.body["timings"]["prompt_n"]

    # Slot 0 is the only slot with KV — should NOT be cleared
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()

    # Launching slot 1 clears idle slot 0
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in log.drain()

    # Re-send same prompt — should restore from cache-ram
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "updating prompt cache" in log.drain()
    assert res.body["timings"]["cache_n"] > 0
    assert res.body["timings"]["prompt_n"] < original_prompt_n

    # Follow-up — slot 0 kept its KV, no clearing needed
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT + " The knight finally reached the castle gates.",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


def test_disabled_with_flag():
    global server
    server.no_cache_idle_slots = True
    server.start()
    log = LogReader(server.log_path)

    # Feature should not be enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" not in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # Request on different slot — should NOT trigger clearing
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


def test_prompt_cache_prefers_deeper_absolute_prefix():
    """A tiny live prefix must not block a much deeper cached branch."""
    global server
    server.n_slots = 1
    server.n_ctx = 1024
    server.start()

    shared = LONG_PROMPT * 4
    prompt_a = shared + " Cached branch A ends here."
    prompt_b = shared + " Cached branch B ends here."
    # Keep the live prompt below the server's 0.1 similarity threshold for B,
    # while making it almost entirely a prefix of B (high f_keep).
    prompt_short = LONG_PROMPT[:40]

    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_a,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    n_long = res.body["timings"]["cache_n"] + res.body["timings"]["prompt_n"]

    # This replaces the live long slot and saves A in cache-ram.
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_short,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    n_short = res.body["timings"]["cache_n"] + res.body["timings"]["prompt_n"]
    assert n_short < n_long // 4

    # B shares almost all of A, but the short live slot has a better f_keep
    # ratio. Cache selection must optimize reusable token count, not require
    # both ratios to improve.
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_b,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] > n_long // 2
