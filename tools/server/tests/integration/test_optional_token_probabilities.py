#!/usr/bin/env python3
"""Check optional token probabilities against an already running CPU test server."""
import argparse
import json
import math
import urllib.request


def complete(url, prompt, n_probs, stream, timeout):
    body = {
        "prompt": prompt,
        "n_predict": 6,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": False,
        "ignore_eos": True,
        "n_probs": n_probs,
        "stream": stream,
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if stream:
            events = []
            for line in response:
                line = line.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    events.append(json.loads(payload))
        else:
            events = [json.load(response)]
    assert events and events[-1].get("stop"), events
    assert not any("error" in event for event in events), events
    content = "".join(event.get("content", "") for event in events)
    rows = [row for event in events for row in event.get("completion_probabilities", [])]
    if n_probs == 0:
        assert not rows, rows
    else:
        assert rows, events
        for row in rows:
            top = row.get("top_logprobs", [])
            assert 0 < len(top) <= n_probs, row
            assert all(math.isfinite(entry["logprob"]) for entry in top), row
    return content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--prompt", default="Once upon a time, there was a little")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    baseline = None
    for n_probs, stream in ((0, False), (5, False), (0, True), (5, True)):
        content = complete(args.url, args.prompt, n_probs, stream, args.timeout)
        if baseline is None:
            baseline = content
        assert content == baseline, (n_probs, stream, content, baseline)
    print("PASS: same greedy text with optional probabilities, normal and streaming responses")


if __name__ == "__main__":
    main()

