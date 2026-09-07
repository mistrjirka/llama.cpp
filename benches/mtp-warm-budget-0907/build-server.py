#!/usr/bin/env python3
"""Server-only build against the identical, already-tested KV-refresh engine.
No task layout/header changed. Recompile all modified translation units and
replace just their members in a private copy of the server archive.
"""
import concurrent.futures as cf,hashlib,json,pathlib,shlex,subprocess
W=pathlib.Path('/workspace/llama-mtp-warm-budget-0907')
OLD=pathlib.Path('/workspace/llama-parallel-all-0907'); BUILD=OLD/'build-sm70-75'
R=pathlib.Path(__file__).resolve().parent; B=R/'candidate-bin'; B.mkdir(exist_ok=True)
subprocess.run(['cp','-a','--reflink=auto',str(R.parent/'mtp-kv-refresh-0907/final-bin')+'/.',str(B)],check=True)
flags=(BUILD/'tools/server/CMakeFiles/server-context.dir/flags.make').read_text()
opts=[]
for line in flags.splitlines():
 if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
  text=line.split('=',1)[1]
  for arg in shlex.split(text):
   # Generated headers remain from the baseline; source includes use this worktree.
   if str(BUILD) not in arg:arg=arg.replace(str(OLD),str(W))
   opts.append(arg)
units=['server-context.cpp','server-schema.cpp','server-task.cpp']
def compile(name):
 cmd=['/usr/bin/c++',*opts,'-c',str(W/'tools/server'/name),'-o',str(R/(name+'.o'))]
 subprocess.run(cmd,check=True)
 return cmd
with cf.ThreadPoolExecutor(max_workers=3) as ex:commands=list(ex.map(compile,units))
archive=R/'libserver-context.a'
subprocess.run(['cp','--reflink=auto',str(BUILD/'tools/server/libserver-context.a'),str(archive)],check=True)
subprocess.run(['ar','r',str(archive),*[str(R/(x+'.o')) for x in units]],check=True)
subprocess.run(['ranlib',str(archive)],check=True)
cmd=shlex.split((BUILD/'tools/server/CMakeFiles/llama-server-impl.dir/link.txt').read_text())
cmd[cmd.index('-o')+1]=str(B/'libllama-server-impl.so')
cmd=[str(archive) if a=='libserver-context.a' else a.replace('-Wl,-rpath,'+str(BUILD/'bin')+':','-Wl,-rpath,'+str(B)) for a in cmd]
subprocess.run(cmd,cwd=BUILD/'tools/server',check=True)
manifest={'base':subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip(),'compile_commands':commands,'link_command':cmd,'engine_source':'5e5ee0440 + server-only request-budget patch','binaries':{}}
for p in B.iterdir():
 if p.is_file() and not p.is_symlink() and ('llama' in p.name or 'ggml-cuda' in p.name):manifest['binaries'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
(R/'build-manifest.json').write_text(json.dumps(manifest,indent=2))
print('Built private server library; engine and draft kernels unchanged.',flush=True)
