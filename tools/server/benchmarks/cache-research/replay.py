#!/usr/bin/env python3
"""Cache-policy simulator, using real rendered token prefixes and synthetic
auxiliary requests. No inference is performed. Byte-level equality and actual
I/O timings are measured separately by snapshot-store.cpp.
"""
import argparse
import array
import collections
import dataclasses
import datetime
import hashlib
import json
import pathlib
import urllib.request

MIB = 1024**2
GIB = 1024**3
PAGE = 256
KV = 30464                 # controlled SenseNova Q8 attention bytes/token
RS = 88 * MIB              # measured recurrent checkpoint (rounded)
BASE = 20570

def post(base, route, body):
    req = urllib.request.Request(base+route, data=json.dumps(body).encode(),
                                 headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)

def render(trace, url):
    result = []
    with trace.open("rb") as f:
        f.seek(max(0, trace.stat().st_size-180*MIB))
        f.readline()
        for line in f:
            try:
                d = json.loads(line)
                ts = datetime.datetime.fromisoformat(d["ts"]).astimezone().strftime("%Y-%m-%d %H:%M:%S")
                b = d["requestBody"]
            except (KeyError, ValueError, TypeError):
                continue
            if not "2026-11-24 16:48:58" <= ts <= "2026-11-24 17:38:28":
                continue
            if d.get("status") != 200 or b.get("reasoning_format") != "none":
                continue
            p = post(url, "/apply-template", b)["prompt"]
            t = post(url, "/tokenize", {"content":p,"add_special":True,"parse_special":True})["tokens"]
            result.append((ts, array.array("i",t).tobytes(), True))
    assert len(result) >= 20, "historical trace not found"
    return result

def lcp(a,b):
    lo,hi = 0,min(len(a),len(b))//4
    while lo < hi:
        mid = (lo+hi+1)//2
        if a[:mid*4] == b[:mid*4]:lo=mid
        else:hi=mid-1
    return lo

def pages(t):
    prev=b""
    result=[]
    for off in range(0,len(t),4*PAGE):
        part=t[off:off+4*PAGE]
        prev=hashlib.sha256(prev+part).digest()
        result.append((prev,len(part)//4))
    return tuple(result)

@dataclasses.dataclass
class Snapshot:
    tokens: bytes
    keys: tuple
    ckpts: tuple
    last: int
    hits: int = 0
    pinned: bool = False

    @property
    def n(self):return len(self.tokens)//4

def simulate(requests, budget, recurrent, shared, policy, adaptive=False, disk_budget=0, pin=True):
    ram=[]
    disk=[]
    active=None
    peak=0
    reused=total=disk_reads=disk_bytes=evicted=0
    seen=[]

    def size(entries, dedup=shared):
        kv = sum(n for _,n in ({k for s in entries for k in s.keys} if dedup else
                    [k for s in entries for k in s.keys]))*KV
        return kv+recurrent*sum(len(s.ckpts) for s in entries)

    def move_disk(s):
        if not disk_budget or s.pinned:return
        disk.append(s)
        while disk and size(disk)>disk_budget:disk.pop(0)

    def enforce(now):
        nonlocal evicted,peak
        while ram and size(ram)>budget:
            c=[s for s in ram if not s.pinned]
            if not c:c=list(ram)
            if policy=="lru":
                victim=min(c,key=lambda s:s.last)
            else:
                all_size=size(ram)
                def value(s):
                    marginal=all_size-size([q for q in ram if q is not s])
                    return (1+s.hits)*s.n/max(marginal,1)/(1+(now-s.last)/8)
                victim=min(c,key=value)
            ram.remove(victim)
            move_disk(victim)
            evicted+=1
        peak=max(peak,size(ram))

    def reusable(s,t):
        matched=min(lcp(s.tokens,t),len(t)//4-1)
        if not recurrent:return matched
        return max((p for p in s.ckpts if p<=matched),default=0)

    for now,(_,t,main) in enumerate(requests):
        n=len(t)//4
        options=[(s,"ram") for s in ram]+[(s,"disk") for s in disk]
        if active is not None:options.append((active,"active"))
        # Prefer already-loaded state on ties; disk has lower priority than RAM.
        best,source,k=None,"none",0
        for s,src in sorted(options,key=lambda x:{"disk":0,"ram":1,"active":2}[x[1]]):
            v=reusable(s,t)
            if v>0 and v>=k:best,source,k=s,src,v
        if best:
            best.hits+=1
            best.last=now
            if source=="disk":
                disk_reads+=1
                disk_bytes+=size([best],False)
                disk.remove(best)
            elif source=="ram" and not best.pinned:
                ram.remove(best)
        # Save a displaced branch, not every append-only request.
        if active is not None and budget and lcp(active.tokens,t)<active.n-16:
            ram.append(active)
        if main:total+=n;reused+=k

        # Checkpoints can only be made at a boundary encountered during
        # this prefill, or inherited from a restored state.
        ck=set(p for p in best.ckpts if p<=k) if best else set()
        if recurrent:
            candidates={n,max(1,n-4)}
            if main and BASE<=n:candidates.add(BASE)
            candidates.update(range(8192,n,8192))
            if adaptive:
                # Past branching boundaries, never look ahead.
                candidates.update(lcp(old,t) for old in seen[-8:])
            ck.update(p for p in candidates if k<=p<=n)
            # Limit to one final state and three rewind checkpoints.
            keep={n}
            if BASE in ck:keep.add(BASE)
            keep.add(max((p for p in ck if p<n),default=n))
            rest=ck-keep
            if adaptive:
                hits=collections.Counter(lcp(old,t) for old in seen[-8:])
                order=sorted(rest,key=lambda p:(hits[p],p),reverse=True)
            else:order=sorted(rest,reverse=True)
            keep.update(order[:max(0,4-len(keep))])
            ck=keep
        else:ck=set()
        active=Snapshot(t,pages(t),tuple(sorted(ck)),now)
        if pin and budget and main and n>=BASE and not any(s.pinned for s in ram) and k<=BASE:
            pt=t[:BASE*4]
            pinned=Snapshot(pt,pages(pt),(BASE,) if recurrent else (),now,pinned=True)
            if size([pinned])<=budget:ram.append(pinned)
        enforce(now)
        seen.append(t)
    return {"hit_pct":round(100*reused/max(total,1),2),
            "recomputed_tokens":total-reused,"peak_ram_MiB":round(peak/MIB,1),
            "evicted":evicted,"disk_restores":disk_reads,
            "disk_read_MiB":round(disk_bytes/MIB,1)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trace",type=pathlib.Path,required=True)
    ap.add_argument("--url",default="http://localhost:2345")
    ap.add_argument("--out",type=pathlib.Path,required=True)
    args=ap.parse_args()
    main_requests=render(args.trace,args.url)
    # Approximate the observed unrelated 6-17k-token requests. Their actual
    # prompt contents are unavailable, so use a distinct synthetic prefix.
    mixed=[]
    aux_n=0
    for ts,t,is_main in main_requests:
        if ts>"2026-11-24 17:02:37":
            n=min(17000,6181+aux_n*553)
            aux=array.array("i", (500000+i for i in range(n))).tobytes()
            mixed.append((ts,aux,False))
            aux_n+=1
        mixed.append((ts,t,is_main))
    scenarios=[
        ("whole-lru",False,"lru",False,0),
        ("shared-lru",True,"lru",False,0),
        ("shared-value",True,"value",False,0),
        ("shared-adaptive",True,"value",True,0),
        ("shared-adaptive-ssd",True,"value",True,16*GIB),
    ]
    results={"main_requests":len(main_requests),"synthetic_aux":aux_n,
             "kv_bytes_per_token":KV,"recurrent_checkpoint_bytes":RS,
             "assumptions":"Token-prefix identity, fixed checkpoint size, no GPU transfer/SSD latency. Policies are experimental heuristics, not exact reproductions of papers.",
             "runs":[]}
    for arch,recurrent in [("attention",0),("hybrid",RS)]:
        for name,seq in [("trace-only",main_requests),("interleaved",mixed)]:
            result=simulate(seq,0,recurrent,False,"lru",pin=False)
            results["runs"].append(dict(arch=arch,workload=name,policy="single-slot",budget_GiB=0,**result))
            for gb in (1,2,4,8):
                for pol,shared,eviction,adaptive,disk in scenarios:
                    result=simulate(seq,gb*GIB,recurrent,shared,eviction,adaptive,disk)
                    results["runs"].append(dict(arch=arch,workload=name,policy=pol,budget_GiB=gb,**result))
    args.out.write_text(json.dumps(results,indent=2))
    print("Trace requests:",len(main_requests),"synthetic auxiliary:",aux_n,flush=True)
    for r in results["runs"]:
        if r["arch"]=="hybrid" and r["workload"]=="interleaved":
            print(json.dumps(r),flush=True)

if __name__=="__main__":main()
