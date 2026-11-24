#!/usr/bin/env python3
"""Check legacy PRC1 compatibility, budgeted import, and exact restore."""
import argparse
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import test_retained_prefix as fixture

p=argparse.ArgumentParser()
p.add_argument("--server",type=Path,required=True)
p.add_argument("--model",type=Path,required=True)
args=p.parse_args()
args.gpu_layers=0
args.template=""
def post(url,route,body):
    return fixture.post(url,route,body)

with fixture.server(args,64,True) as (url,log):
    root=log.parent
    def tok(s,bos=False):
        return post(url,"/tokenize",{"content":s,"add_special":bos,"parse_special":True})["tokens"]
    a={"prompt":tok("A little bird in a garden. "*65,True)+tok(" Tell a short story."),
       "temperature":0,"seed":42,"n_predict":4,"cache_prompt":True}
    b={"prompt":tok("A different story about a dog. "*40,True),
       "temperature":0,"seed":42,"n_predict":4,"cache_prompt":True}
    h0,answer=fixture.completion(url,"cold A",a)
    fixture.completion(url,"displace A",b)
    saved=post(url,"/prompt-cache?action=save",{"filename":"roundtrip.bin"})
    assert saved["n_saved"]>=1,saved
    raw=(root/"roundtrip.bin").read_bytes()
    magic,version,count=struct.unpack_from("<IIQ",raw)
    assert magic==0x31435250 and version==1 and count>=1
    fixture.erase(url)
    restored=post(url,"/prompt-cache?action=restore",{"filename":"roundtrip.bin"})
    assert restored["n_restored"]==saved["n_saved"],restored
    hit,answer2=fixture.completion(url,"restored A",a)
    assert hit>100 and answer2==answer,(hit,answer,answer2)
    # A valid v1 file structure, but the large synthetic state is not a real
    # sequence state and must be rejected before it can be used for inference.
    # This tests the loader's streaming admission, not the model parser.
    data=bytearray(struct.pack("<IIQ",magic,version,1))
    data+=struct.pack("<BQ",0,4)+struct.pack("<i",1)
    data+=struct.pack("<Q",70*1024*1024)
    oversize=root/"oversize.bin"
    with oversize.open("wb") as f:
        f.write(data)
        # Sparse zero-filled blocks deduplicate, so populate distinct chunks
        # instead of relying on the logical size to enforce the limit.
        import random
        rng=random.Random(74)
        for _ in range(70*8):
            f.write(rng.randbytes(128*1024))
        f.write(struct.pack("<QQ",0,0))
    result=post(url,"/prompt-cache?action=restore",{"filename":"oversize.bin"})
    assert result["n_restored"]==0,result
    # Malformed input fails closed and leaves the server responsive.
    (root/"truncated.bin").write_bytes(raw[:-5])
    try:
        post(url,"/prompt-cache?action=restore",{"filename":"truncated.bin"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("truncated cache accepted")
    fixture.erase(url)
    hit,answer3=fixture.completion(url,"after malformed cache",a)
    assert hit==0 and answer3==answer
    print("PASS PRC1 roundtrip, budget rejection, malformed-file recovery",flush=True)
