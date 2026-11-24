#!/usr/bin/env python3
"""Small ordinary-attention model fixture; no production service changes."""
import hashlib, json, pathlib, subprocess, sys, time, urllib.request

repo = "ggml-org/test-model-stories260K"
revision = "479896ec924af6d40fd419ab8f4d1eb2101de00d"
filename = "stories260K-f32.gguf"
sha = "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d"
out = pathlib.Path(sys.argv[1])
out.mkdir(mode=0o700, parents=True, exist_ok=True)
model = out / filename
if not model.exists():
    with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/{filename}", timeout=30) as r:
        model.write_bytes(r.read())
assert hashlib.sha256(model.read_bytes()).hexdigest() == sha
base = "http://127.0.0.1:2346"
def post(route, body):
    req = urllib.request.Request(base+route, data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

server = pathlib.Path(__file__).resolve().parents[4] / "build-cuda/bin/llama-server"
with (out/"server.log").open("w") as log:
    p = subprocess.Popen([str(server), "-m",str(model), "--host","127.0.0.1",
         "--port","2346","--ctx-size","512","--parallel","1",
         "--gpu-layers","0","--threads","2","--flash-attn","on",
         "--cache-ram","64","--slot-save-path",str(out)],stdout=log,stderr=log)
    try:
        for _ in range(100):
            if p.poll() is not None:
                raise RuntimeError("Test server exited")
            try:
                with urllib.request.urlopen(base+"/health",timeout=1) as r:
                    if json.load(r).get("status")=="ok":break
            except Exception:pass
            time.sleep(.2)
        else:raise RuntimeError("Test server not ready")
        prefix = "Once upon a time, a little girl found a blue bird. " * 10
        a = prefix + "The bird flew to a small tree."
        b = prefix + "The bird sat on a warm stone."
        entries=[]
        for name,prompt in [("a",a),("a2",a+" The girl walked into the garden. "*12),
            ("b",b),("b2",b+" She was very happy to see the bird. "*12),
            ("aux","A different story about a dog. "*25)]:
            r=post("/completion",{"prompt":prompt,"n_predict":2,"temperature":0,
                 "seed":42,"cache_prompt":True})
            result=post("/slots/0?action=save",{"filename":name+".bin"})
            entries.append({"name":name,"bytes":(out/(name+".bin")).stat().st_size,
                "prompt_n":r.get("timings",{}).get("prompt_n"),"save":result})
            print(json.dumps(entries[-1]),flush=True)
        (out/"manifest.json").write_text(json.dumps(entries,indent=2))
    finally:
        p.terminate()
        try:p.wait(timeout=10)
        except subprocess.TimeoutExpired:p.kill();p.wait()
