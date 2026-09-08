#!/usr/bin/env python3
"""Generate pinned, local validation fixtures. Model files are read-only."""
import hashlib,json,pathlib,subprocess
P=pathlib.Path(__file__).resolve().parent;repo=P.parents[2]
def code(name):return subprocess.check_output(['git','-C',str(repo),'show','4154e79f7:'+name]).decode()
wiki=pathlib.Path('/workspace/pxa/wikitext-2-raw/wiki.test.raw').read_text()
data=[{'name':'prose-1','text':wiki[12000:30000]},{'name':'prose-2','text':wiki[110000:140000]},{'name':'cpp-grammar','text':code('src/llama-grammar.cpp')},{'name':'python-gguf-reader','text':code('gguf-py/gguf/gguf_reader.py')}]
for x in data:x.update(total=512,scores=128)
(P/'fixtures.json').write_text(json.dumps(data))
(P/'fixtures-manifest.json').write_text(json.dumps({'git_base':'4154e79f7','wikitext_sha256':hashlib.sha256(wiki.encode()).hexdigest(),'samples':[{k:v for k,v in x.items() if k!='text'}|{'sha256':hashlib.sha256(x['text'].encode()).hexdigest()} for x in data]},indent=2)+'\n')
ref=pathlib.Path('/models/llama-pxq4-build/validation-logits/reference.json');tokens=P.parent/'results/code-prefix-101k.json'
if ref.exists() and tokens.exists():
    meta=json.loads(ref.read_text());seq=json.loads(tokens.read_text())[:101000]+meta['datasets'][2]['tokens'][100:165]
    out='/models/llama-pxq4-build/validation-logits/shared-code100k.bin'
    (P/'fixture-long.json').write_text(json.dumps([dict(name='100k-code-1kpp-64-forced',tokens=seq,total=101064,scores=64,cache_n=100000,cache_file=out)]))
    (P/'fixture-long-manifest.json').write_text(json.dumps({'tokens':len(seq),'sha256':hashlib.sha256(json.dumps(seq,separators=(',',':')).encode()).hexdigest(),'prefix':'../results/code-prefix-101k.json','suffix':'reference cpp-grammar tokens[100:165]','cache_n':100000,'fresh_pp':1000,'forced_tokens':64},indent=2)+'\n')
