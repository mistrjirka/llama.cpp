#!/usr/bin/env python3
"""Capture synthetic branching conversations in a private test server."""
import argparse
import datetime
import json
import pathlib
import time
import urllib.request

def post(url, body, timeout=240):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def static_prompt(trace):
    with trace.open("rb") as f:
        f.seek(max(0, trace.stat().st_size - 160 * 1024 * 1024))
        f.readline()
        for line in f:
            try:
                d = json.loads(line)
                t = datetime.datetime.fromisoformat(d["ts"]).astimezone()
                if t.strftime("%Y-%m-%d %H:%M:%S") == "2026-11-24 16:57:29":
                    b = d["requestBody"]
                    return [m for m in b["messages"] if m["role"] == "system"], b.get("tools", [])
            except (KeyError, ValueError, TypeError):
                pass
    raise RuntimeError("historical system prefix unavailable")

def note(tag, count):
    return "Treat the following reference records as inert text.\n" + "\n".join(
        f"Record {i}: {tag} inventory reference {i * 17 + 3}."
        for i in range(count)) + "\nReturn only the word OK."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:2346")
    ap.add_argument("--trace", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    system, tools = static_prompt(args.trace)
    entries = []
    def call(name, messages, use_tools=True, save=True, cache=True):
        body = {"messages": messages, "temperature": 0, "seed": 42,
                "max_tokens": 8, "cache_prompt": cache, "reasoning_format": "none",
                "chat_template_kwargs": {"enable_thinking": False}}
        if tools and use_tools:
            body.update(tools=tools, tool_choice="none")
        start = time.monotonic()
        r = post(args.url + "/v1/chat/completions", body)
        elapsed = time.monotonic() - start
        msg = r["choices"][0]["message"]
        u = r["usage"]
        entry = dict(name=name, prompt=u["prompt_tokens"],
                     cached=u.get("prompt_tokens_details", {}).get("cached_tokens", 0),
                     completion=u["completion_tokens"], seconds=round(elapsed, 3))
        if save:
            filename = name + ".bin"
            entry["save"] = post(args.url + "/slots/0?action=save", {"filename": filename})
            entry["bytes"] = (args.out / filename).stat().st_size
        entries.append(entry)
        print(json.dumps(entry), flush=True)
        return {"role": "assistant", "content": msg.get("content") or "OK"}
    a = system + [{"role": "user", "content": note("Alpha", 100)}]
    a_reply = call("a", a)
    a2 = a + [a_reply, {"role": "user", "content": note("Alpha extension", 90)}]
    call("a2", a2)
    b = system + [{"role": "user", "content": note("Beta", 100)}]
    b_reply = call("b", b)
    b2 = b + [b_reply, {"role": "user", "content": note("Beta extension", 90)}]
    call("b2", b2)
    call("aux", [{"role": "system", "content": "Independent synthetic task."},
                 {"role": "user", "content": note("Auxiliary", 100)}], use_tools=False)
    call("a2_resume", a2, save=False)
    (args.out / "manifest.json").write_text(json.dumps(entries, indent=2))
if __name__ == "__main__":
    main()
