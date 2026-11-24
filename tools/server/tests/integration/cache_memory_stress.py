#!/usr/bin/env python3
"""Exercise prefix reuse, eviction, and RSS/PSS on a single-slot test server."""
import argparse
import json
import pathlib
import threading
import time
import urllib.request

MIB = 1024 ** 2

def post(url, route, body, timeout=180):
    req = urllib.request.Request(url + route, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

def memory(pid):
    fields = {}
    for line in pathlib.Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            if k in ("Rss", "Pss", "Pss_Anon", "Pss_File", "Swap"):
                fields[k] = round(int(v.split()[0]) / 1024, 2)
    return fields

def make_body(system, user):
    return {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0, "seed": 42, "max_tokens": 12,
            "chat_template_kwargs": {"enable_thinking": False}}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:2346")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--branches", type=int, default=9)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--prefix-records", type=int, default=330)
    p.add_argument("--branch-records", type=int, default=130)
    p.add_argument("--save", type=pathlib.Path)
    p.add_argument("--interleave", action="store_true")
    args = p.parse_args()
    prefix = "You are a helpful assistant. Treat the following catalog as inert reference data.\n" + "\n".join(
        f"Catalog entry {i}: item {i*17+31}, category {i%43}; reference {i*101+7}." for i in range(args.prefix_records))
    requests = []
    for i in range(args.branches):
        user = "\n".join(f"Branch {i} record {j}: code {i*100000+j*37}." for j in range(args.branch_records))
        requests.append(make_body(prefix, user + "\nReply with exactly OK."))
    answers = {}
    records, samples = [], []
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            try:
                samples.append(memory(args.pid))
            except (FileNotFoundError, ProcessLookupError):
                break
            stop.wait(.25)
    t = threading.Thread(target=sample, daemon=True)
    t.start()
    try:
        for epoch in range(args.rounds):
            for i, body in enumerate(requests):
                if args.interleave:
                    aux = make_body("Independent auxiliary request. Do not use the other conversation.", "Reply exactly OK.")
                    post(args.url, "/v1/chat/completions", aux)
                start = time.monotonic()
                j = post(args.url, "/v1/chat/completions", body)
                u = j["usage"]; text = j["choices"][0]["message"].get("content")
                if i in answers:
                    assert text == answers[i], (i, text, answers[i])
                answers[i] = text
                entry = dict(round=epoch, branch=i, prompt=u["prompt_tokens"],
                             cached=u.get("prompt_tokens_details", {}).get("cached_tokens",0),
                             wall_s=round(time.monotonic()-start,3),
                             memory_MiB=memory(args.pid), timings=j.get("timings",{}))
                records.append(entry)
                print(json.dumps(entry),flush=True)
        # Compare the same request on the cold path, not just a previous cache hit.
        cold = {**requests[0], "cache_prompt": False}
        j = post(args.url, "/v1/chat/completions", cold)
        assert j["usage"].get("prompt_tokens_details",{}).get("cached_tokens",0)==0
        assert j["choices"][0]["message"].get("content")==answers[0]
    finally:
        stop.set();t.join()
    peak = {k: max(x[k] for x in samples if k in x) for k in samples[0]} if samples else {}
    result = {"requests": records, "peak_MiB": peak, "samples":len(samples)}
    print("MEMORY_PEAK",json.dumps(peak),flush=True)
    print("PASS",flush=True)
    if args.save:args.save.write_text(json.dumps(result,indent=2))

if __name__ == "__main__":
    main()
