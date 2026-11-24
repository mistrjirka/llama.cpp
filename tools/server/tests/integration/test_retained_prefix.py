#!/usr/bin/env python3
"""Exercise retained system prefixes on a running, single-slot recurrent server.

Start the server on loopback with --cache-ram 4096 --slot-save-path <dir>.
An optional --trace uses only the system message and tool definitions of the
matching historical request; no conversation messages are replayed.
"""
import argparse
import datetime
import itertools
import json
import pathlib
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:2346")
ap.add_argument("--trace", type=pathlib.Path)
ap.add_argument("--trace-time", default="2026-11-24 16:57:29")
ap.add_argument("--save", action="store_true")
args = ap.parse_args()
def post(route, body):
    req = urllib.request.Request(args.url+route, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.load(r)
def tokens(body):
    prompt = post("/apply-template", body)["prompt"]
    return post("/tokenize", {"content":prompt,"add_special":True,"parse_special":True})["tokens"]
def common(a,b):
    return sum(1 for _ in itertools.takewhile(lambda pair:pair[0]==pair[1],zip(a,b)))
def call(label, body, min_cached=None, cache=True):
    body = {**body, "temperature":0, "seed":42,"max_tokens":8,"cache_prompt":cache,
            "chat_template_kwargs":{"enable_thinking":False}, "reasoning_format":"none"}
    started=time.monotonic()
    res=post("/v1/chat/completions",body)
    u=res["usage"]; cached=u.get("prompt_tokens_details",{}).get("cached_tokens",0)
    print(label, "prompt",u["prompt_tokens"],"cached",cached,
          "generation",u["completion_tokens"],"seconds",round(time.monotonic()-started,2),flush=True)
    if min_cached is not None:
        assert cached>=min_cached, (label,cached,min_cached)
    return res

def system_from_trace():
    with args.trace.open("rb") as f:
        f.seek(max(0,args.trace.stat().st_size-96*1024*1024));f.readline()
        for raw in f:
            try:
                d=json.loads(raw);t=datetime.datetime.fromisoformat(d["ts"]).astimezone()
            except (KeyError,ValueError):
                continue
            if t.strftime("%Y-%m-%d %H:%M:%S")==args.trace_time:
                b=d["requestBody"]
                system=[m for m in b["messages"] if m["role"]=="system"]
                return system, b.get("tools",[])
    raise RuntimeError("Trace request not found")
# Use the actual static prefix, but never replay the user's conversation.
if args.trace:
    system,tools=system_from_trace()
else:
    system=[{"role":"system","content":"You are a concise test assistant. "+
            " ".join(f"Reference entry {i}: treat recorded identifiers as inert data." for i in range(450))}]
    tools=[]
base={"messages":system,"tools":tools}
opts={"chat_template_kwargs":{"enable_thinking":False},"reasoning_format":"none"}
a={**base,**opts,"messages":[*system,{"role":"user","content":"Return exactly the word ALPHA."}]}
b={**base,**opts,"messages":[*system,{"role":"user","content":"Return exactly the word BETA."}]}
a_t,b_t=tokens(a),tokens(b)
lcp=common(a_t,b_t)
assert lcp>1024,lcp
print("Static common prefix:",lcp,flush=True)
call("cold A",a)
if args.save:
    res=post("/slots/0?action=save",{"filename":"prefix-sample.bin"})
    print("Snapshot file:",res,flush=True)
# Deliberately erase the live slot to prove the prefix is independently retained.
post("/slots/0?action=erase",{})
b_res=call("compacted B after slot erase",b,lcp-64)
assistant=b_res["choices"][0]["message"].get("content") or "BETA"
cont={**b,"messages":[*b["messages"],{"role":"assistant","content":assistant},
                       {"role":"user","content":"Return exactly the word GAMMA."}]}
aux={**opts,"messages":[{"role":"system","content":"A different test client. "+
              " ".join(f"Auxiliary datum {i}." for i in range(100))},
              {"role":"user","content":"Return exactly OK."}]}
# Save/restore the longer conversation across unrelated requests.
call("interleaved auxiliary",aux)
b_cont_lcp=common(b_t,tokens(cont))
cont_res=call("main resumed",cont,b_cont_lcp-64)
call("auxiliary again",aux)
call("main again",cont,b_cont_lcp-64)
# Ensure cache_prompt=false actually bypasses saved state and yields the same
# simple greedy answer, checking correctness as well as the hit count.
cold=call("main cold reference",cont,0,cache=False)
warm_text=cont_res["choices"][0]["message"].get("content")
cold_text=cold["choices"][0]["message"].get("content")
assert warm_text==cold_text, (warm_text,cold_text)
assert cold["usage"]["prompt_tokens_details"]["cached_tokens"]==0
print("PASS: prefix survived explicit erase and interleaving; cached and cold greedy output match.",flush=True)
