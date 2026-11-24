#!/usr/bin/env python3
"""Live regression: a retained system prefix survives active-slot erase.

Run against a freshly built llama-server using a small text GGUF model.
The optional chat-template override is useful for synthetic fixtures.
No personal conversation data is required.
"""
import argparse
import contextlib
import itertools
import json
import pathlib
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def server(args, ram, enabled):
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="retained-prefix-") as d:
        log_path = pathlib.Path(d) / "server.log"
        cmd = [str(args.server.resolve()), "-m", str(args.model.resolve()),
               "--host", "127.0.0.1", "--port", str(port), "-c", "4096",
               "-b", "512", "-ub", "128", "-np", "1", "-t", "4",
               "-ngl", str(args.gpu_layers), "--cache-ram", str(ram),
               "--ctx-checkpoints", "3", "--slot-save-path", d]
        if args.template:
            cmd += ["--chat-template", args.template]
        if not enabled:
            cmd += ["--no-cache-system-prefix"]
        with log_path.open("wb") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            try:
                for _ in range(240):
                    if proc.poll() is not None:
                        raise RuntimeError("server exited during startup")
                    try:
                        with urllib.request.urlopen(url + "/health", timeout=1) as response:
                            if json.load(response).get("status") == "ok":
                                break
                    except (OSError, urllib.error.HTTPError):
                        pass
                    time.sleep(.25)
                else:
                    raise RuntimeError("server startup timeout")
                yield url, log_path
            except Exception:
                print(log_path.read_text(errors="replace")[-7000:], flush=True)
                raise
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def post(url, route, body):
    req = urllib.request.Request(url + route, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{route}: {error.code} {error.read().decode()}") from error


def make_request(system, user, cache=True):
    return dict(messages=[{"role":"system", "content":system},
                          {"role":"user", "content":user}],
                max_tokens=4, temperature=0, seed=42, cache_prompt=cache)


def tokenize(url, request):
    prompt = post(url, "/apply-template", request)["prompt"]
    return post(url, "/tokenize", {"content":prompt, "add_special":True,
                                   "parse_special":True})["tokens"]


def common_prefix(a, b):
    return sum(1 for _ in itertools.takewhile(lambda pair:pair[0]==pair[1], zip(a,b)))


def completion(url, label, request):
    start = time.monotonic()
    if "prompt" in request:
        result = post(url, "/completion", request)
        cached = result["tokens_evaluated"] - result["timings"]["prompt_n"]
        count = result["tokens_evaluated"]
        answer = result["content"]
    else:
        result = post(url, "/v1/chat/completions", request)
        usage = result["usage"]
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        count = usage["prompt_tokens"]
        answer = result["choices"][0]["message"].get("content")
    print(json.dumps({"test":label, "cached":cached, "prompt":count,
                      "elapsed_s":round(time.monotonic()-start,3)}), flush=True)
    return cached, answer


def erase(url):
    post(url, "/slots/0?action=erase", {})


def run(args, ram, enabled):
    with server(args, ram, enabled) as (url, log):
        prefix = "In a small garden, a little bird sang beside a tree. " * 50
        a = make_request(prefix, "Answer the first question.")
        b = make_request(prefix, "Tell a different short story.")
        # Fit a prefix > 1024 tokens without exceeding the 4096-token context.
        for _ in range(4):
            ta,tb=tokenize(url,a),tokenize(url,b)
            lcp = common_prefix(ta,tb)
            if 1100 <= lcp <= 1800:
                break
            prefix = prefix * 2 if lcp < 1100 else prefix[:len(prefix)*2//3]
            a,b=make_request(prefix,"Answer the first question."),make_request(prefix,"Tell a different short story.")
        assert 1024 <= lcp <= 1850, lcp
        if args.native:
            def token_ids(text, special=False):
                return post(url, "/tokenize", {"content":text,"add_special":special,
                                              "parse_special":True})["tokens"]
            prefix_ids = token_ids(prefix,True)
            # An explicit token-array fixture avoids assuming a tiny model's
            # tokenizer implements the production ChatML delimiters.
            delim = "\nQuestion:"
            user_ids = token_ids(delim)
            def native(system, user):
                return {"prompt":system+user_ids+token_ids(user),
                        "message_delimiters":[{"role":"user","delimiter":delim}],
                        "n_predict":4,"temperature":0,"seed":42,"cache_prompt":True}
            a,b=native(prefix_ids,"Answer the first question."),native(prefix_ids,"Tell a different short story.")
            lcp=common_prefix(a["prompt"],b["prompt"])
            aux=native(token_ids("Independent task.",True),"A story about a dog.")
            changed=native(token_ids("A new system instruction. "+prefix,True),"Tell a different short story.")
        else:
            aux=make_request("Independent task.","A story about a dog.")
            changed=make_request("A new system instruction. "+prefix,"Tell a different short story.")
        print("Static token prefix:",lcp,flush=True)
        completion(url,"cold A",a)
        erase(url)
        hit, answer = completion(url,"B after erase",b)
        if ram and enabled:
            assert hit >= lcp-64, (hit,lcp)
            # Restoring a prefix must not consume it.
            erase(url)
            hit2, answer2 = completion(url,"B after second erase",b)
            assert hit2 >= lcp-64, (hit2,lcp)
            assert answer == answer2
            completion(url,"unrelated",aux)
            hit3,answer3 = completion(url,"B after unrelated",b)
            assert hit3 >= lcp-64, (hit3,lcp)
            assert answer3 == answer
        else:
            assert hit == 0, hit
        erase(url)
        # Opting out for a request must bypass retained state.
        cold_hit,cold = completion(url,"cache_prompt=false",{**b,"cache_prompt":False})
        assert cold_hit == 0, cold_hit
        assert cold == answer, (cold,answer)
        if ram and enabled:
            erase(url)
            changed_hit, _ = completion(url,"changed system",changed)
            assert changed_hit < 64, changed_hit
        content=log.read_text(errors="replace")
        if ram and enabled:
            assert "retained system prefix:" in content
            assert "restored retained system prefix:" in content
        else:
            assert "retained system prefix:" not in content
        print("PASS",args.model.name,"RAM",ram,"prefix enabled",enabled,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--server",type=pathlib.Path,required=True)
    p.add_argument("--model",type=pathlib.Path,required=True)
    p.add_argument("--template",default="chatml")
    p.add_argument("--gpu-layers",type=int,default=0)
    p.add_argument("--ram",type=int,default=64)
    p.add_argument("--skip-opt-outs",action="store_true")
    p.add_argument("--native",action="store_true")
    args=p.parse_args()
    run(args,args.ram,True)
    if not args.skip_opt_outs:
        run(args,args.ram,False)
        run(args,0,True)


if __name__=="__main__":
    main()
