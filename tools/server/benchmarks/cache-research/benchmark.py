#!/usr/bin/env python3
"""Compare exact snapshot encodings on synthetic branching conversations.

The server is isolated on localhost and the snapshot files are temporary.
No real conversation data is needed. Output is JSON-lines, including timings
and exact-byte verification for each encoding.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

TEST = Path(__file__).resolve().parents[2] / "tests/integration/test_retained_prefix.py"
spec = importlib.util.spec_from_file_location("prefix_test", TEST)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--store",type=Path,required=True)
    p.add_argument("--template",default="")
    p.add_argument("--gpu-layers",type=int,default=0)
    p.add_argument("--ram",type=int,default=512)
    p.add_argument("--native",action="store_true")
    args=p.parse_args()
    with fixture.server(args,args.ram,True) as (url,log):
        directory=log.parent
        system="A small library catalog records the titles of books and the names of their authors. "*70
        def native_tokens(s, special=False):
            return fixture.post(url,"/tokenize",{"content":s,"add_special":special,"parse_special":True})["tokens"]
        def request(tag, extension=""):
            user=f"Discuss the {tag} catalog in one sentence. {extension}"
            if args.native:
                delim="\nQuestion:"
                return {"prompt":native_tokens(system,True)+native_tokens(delim)+native_tokens(user),
                        "message_delimiters":[{"role":"user","delimiter":delim}],
                        "n_predict":4,"temperature":0,"seed":42,"cache_prompt":True}
            return fixture.make_request(system,user)
        for name,tag,ext in [("a","fiction",""),("a2","fiction"," Add a short summary. "*8),
                             ("b","history",""),("b2","history"," Add a short summary. "*8)]:
            fixture.completion(url,name,request(tag,ext))
            fixture.post(url,"/slots/0?action=save",{"filename":name+".bin"})
        aux = {"prompt":native_tokens("A different small task.",True),"n_predict":4,
               "temperature":0,"seed":42,"cache_prompt":True} if args.native else fixture.make_request("Independent task.","Give a short greeting.")
        fixture.completion(url,"aux",aux)
        fixture.post(url,"/slots/0?action=save",{"filename":"aux.bin"})
        paths=[str(directory/(n+".bin")) for n in ("a","a2","b","b2","aux")]
        for mode,size in [("fixed",65536),("gear",65536),("tensor",64),("tensor",256)]:
            res=subprocess.run([str(args.store.resolve()),mode,str(size),"own",*paths],
                               capture_output=True,text=True)
            if res.returncode:
                # Tensor parsing is an offline optimization for the supported
                # contiguous KV serializer, not for arbitrary recurrent layouts.
                if mode=="tensor" and ("unsupported" in res.stderr or "invalid" in res.stderr):
                    print(json.dumps({"mode":mode,"skipped":res.stderr.strip()}),flush=True)
                    continue
                raise RuntimeError(res.stderr)
            print(res.stdout.strip(),flush=True)


if __name__=="__main__":
    main()
