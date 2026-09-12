#!/usr/bin/env python3
"""Relink an isolated CUDA variant, preserving the fixed Stream-K objects."""
import hashlib,json,os,shlex,shutil,subprocess,sys
from pathlib import Path
MAIN=Path('/workspace/llama-v100-optimized'); AREA=Path('/models/.bench-ornith-mtp4')
name,tu,source=sys.argv[1:4]; variant=sys.argv[4] if len(sys.argv)>4 else 'force'
build=MAIN/('build-compare-sm70-75-force-mmq' if variant=='force' else 'build-compare-sm70-75')
paired=AREA/'int8-streamk-validation'/('matched-'+variant)
base=AREA/'int8-streamk-fixed-src'/('build-matched-'+variant)/'bin'
work=AREA/'nondeterminism-0912'/name;dest=work/'bin';dest.mkdir(parents=True,exist_ok=True)
entries=json.loads((build/'compile_commands.json').read_text());entry=next(x for x in entries if Path(x['file']).name==tu)
argv=shlex.split(entry['command']);argv[argv.index('-o')+1]=str(work/(tu+'.o'));argv[argv.index('-c')+1]=str(Path(source).resolve());argv+=['-I'+str(MAIN/'ggml/src/ggml-cuda')]
with (work/'compile.log').open('wb') as log: subprocess.run(['ccache']+argv,cwd=build,stdout=log,stderr=subprocess.STDOUT,check=True)
for p in base.iterdir():
 if p.is_dir() or p.name.startswith('libggml-cuda') or p.name.startswith('.'):continue
 q=dest/p.name
 if q.is_symlink() or q.exists():q.unlink()
 if p.is_symlink():q.symlink_to(os.readlink(p))
 else:shutil.copy2(p,q)
raw=subprocess.check_output(['ninja','-C',str(build),'-t','commands','bin/libggml-cuda.so.0.23.0'],text=True).splitlines()[-1]
link=shlex.split(raw.split('&&')[1]);new=[]
for x in link:
 if x.endswith('.o'):
  if Path(x).name==tu+'.o': x=str(work/(tu+'.o'))
  elif Path(x).name in ['fattn-mma-f16-instance-ncols1_16-ncols2_2.cu.o','fattn-mma-f16-instance-ncols1_4-ncols2_8.cu.o','fattn-mma-f16-instance-ncols1_8-ncols2_4.cu.o']:x=str(paired/'fixed'/Path(x).name)
  else:x=str(paired/'unchanged-objects'/x)
 elif x.startswith('bin/'):x=str(dest/Path(x).name)
 x=x.replace(str(build/'bin'),str(dest));new.append(x)
new[new.index('-o')+1]=str(dest/'libggml-cuda.so.0.23.0')
with (work/'link.log').open('wb') as log:subprocess.run(new,cwd=build,stdout=log,stderr=subprocess.STDOUT,check=True)
for a,b in [('libggml-cuda.so','libggml-cuda.so.0'),('libggml-cuda.so.0','libggml-cuda.so.0.23.0')]:
 q=dest/a
 if q.is_symlink():q.unlink()
 q.symlink_to(b)
manifest={'name':name,'tu':tu,'source':str(Path(source).resolve()),'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'compile':argv,'link':new,'library_sha256':hashlib.sha256((dest/'libggml-cuda.so.0.23.0').read_bytes()).hexdigest()}
(work/'manifest.json').write_text(json.dumps(manifest,indent=2));print('BUILT',dest,flush=True)
