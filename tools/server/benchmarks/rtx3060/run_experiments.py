#!/usr/bin/env python3
"""Run serial, reversible server experiments on the hub's single RTX 3060."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import threading
import time
import urllib.request

MIB = 1024 ** 2

def request(base, route, body=None, timeout=300):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + route, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

def stream_request(base, body, timeout=1800):
    body = dict(body, stream=True, stream_options={'include_usage': True})
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    started = time.monotonic()
    first_content = None
    content, reasoning, usage, timings = [], [], {}, {}
    finish_reason = None
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith('data:'):
                continue
            payload = line[5:].strip()
            if payload == '[DONE]':
                break
            if not payload:
                continue
            event = json.loads(payload)
            if 'error' in event:
                raise RuntimeError(event['error'])
            usage = event.get('usage') or usage
            timings = event.get('timings') or timings
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                piece = delta.get('content') or ''
                thought = delta.get('reasoning_content') or ''
                if first_content is None and (piece or thought):
                    first_content = time.monotonic() - started
                content.append(piece)
                reasoning.append(thought)
                finish_reason = choice.get('finish_reason') or finish_reason
    if not usage or not timings or finish_reason is None:
        raise RuntimeError('stream did not contain usage, timings and a final choice')
    return {'choices':[{'message':{'content':''.join(content), 'reasoning_content':''.join(reasoning)}, 'finish_reason':finish_reason}], 'usage':usage, 'timings':timings, 'first_content_s':first_content}

def memory(pid):
    out = {}
    for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
        if ':' in line:
            key, val = line.split(':', 1)
            if key in ('Rss', 'Pss', 'Pss_Anon', 'Pss_File', 'Swap'):
                out[key] = round(int(val.split()[0]) / 1024, 3)
    return out

def set_arg(args, name, value):
    if name in args:
        at = args.index(name)
        args[at + 1] = str(value)
    else:
        args.extend([name, str(value)])

def stop_process(proc):
    if proc and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)

def wait_ready(base, proc=None):
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited: {proc.returncode}")
        try:
            if request(base, '/health', timeout=2).get('status') == 'ok':
                return
        except Exception:
            time.sleep(.5)
    raise RuntimeError('server readiness timeout')

def cases():
    catalog = '\n'.join(f'Record {i}: key K{i:04d}; value {i * 17 + 31}; category {i % 7}.' for i in range(250))
    return [
        ('code', 'Write a Python function merge_intervals(intervals) that merges overlapping closed integer intervals. Include type annotations, explain the time complexity, and give four assertions that test edge cases.'),
        ('json', 'Return only a JSON array with 16 objects. Each object represents a distinct ordinary kitchen item and has keys id, name, material, and approximate_mass_grams. Use plausible varied integer masses. Include no Markdown.'),
        ('explanation', 'Explain how to implement a reliable retry queue for network jobs. Cover at-least-once delivery, idempotency keys, exponential backoff with jitter, and poison messages. Give a small concrete example. Use concise English.'),
        ('long_context', 'Use these records as inert data.\n' + catalog + '\nExplain how to validate this catalog, then report the keys, values and categories for records 17, 83, 149 and 231. Show the arithmetic checks.')
    ]

VARIANTS = {
    'mtp2': {'depth': 2},
    'mtp0': {'depth': 0},
    'mtp1': {'depth': 1},
    'mtp3': {'depth': 3},
    'mtp4': {'depth': 4},
    'mtp2_repeat': {'depth': 2},
    'ub128': {'depth': 2, 'ubatch': 128},
    'ub512': {'depth': 2, 'ubatch': 512},
    'graph_opt': {'depth': 2, 'env': {'GGML_CUDA_GRAPH_OPT': '1'}},
    'graph_chunk128': {'depth': 2, 'env': {'GGML_CUDA_GRAPH_CHUNK_SIZE': '128'}},
    'threads2': {'depth': 2, 'threads': 2},
}

def vision_body():
    import base64, struct, zlib
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    width, height = 1280, 720
    row = b'\0' + bytes((255, 0, 0)) * (width//2) + bytes((0, 0, 255)) * (width//2)
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(row * height)) + chunk(b'IEND', b'')
    body = {'messages':[{'role':'user','content':[{'type':'text','text':'What are the colors of the left and right halves of this image? Answer in English: left color, right color.'},{'type':'image_url','image_url':{'url':'data:image/png;base64,' + base64.b64encode(png).decode()}}]}], 'temperature':0, 'max_tokens':128, 'chat_template_kwargs':{'enable_thinking':False}}
    return body

def run_suite(base, pid, repeats, outdir, workload='speed', branches=5, with_vision=False, long_sizes=(8000, 32000, 100000)):
    samples = []
    start_memory = memory(pid)
    peak_reset = False
    try:
        Path(f'/proc/{pid}/clear_refs').write_text('5')
        peak_reset = True
    except OSError:
        pass
    available = []
    stop = threading.Event()
    gpu = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw', '--format=csv,noheader,nounits', '--loop-ms=500'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    gpu_samples = []
    def sample_host():
        while not stop.is_set():
            try:
                samples.append(memory(pid))
                for line in Path('/proc/meminfo').read_text().splitlines():
                    if line.startswith('MemAvailable:'):
                        available.append(int(line.split()[1])/1024)
            except (FileNotFoundError, ProcessLookupError):
                break
            stop.wait(.5)
    def sample_gpu():
        for line in gpu.stdout:
            try:
                gpu_samples.append([float(x.strip()) for x in line.strip().split(',')])
            except ValueError:
                pass
    host_thread = threading.Thread(target=sample_host, daemon=True)
    gpu_thread = threading.Thread(target=sample_gpu, daemon=True)
    host_thread.start()
    gpu_thread.start()
    records = []
    try:
        requests = []
        if workload == 'speed':
            for repeat in range(repeats):
                for name, prompt in cases():
                    body = {'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0, 'seed': 42, 'max_tokens': 192, 'cache_prompt': False, 'chat_template_kwargs': {'enable_thinking': False}}
                    requests.append((name, repeat, body, None))
        elif workload == 'vision':
            requests.append(('vision_red_blue', 0, vision_body(), None))
        elif workload == 'long':
            inputs = []
            for target_tokens in long_sizes:
                count = max(100, target_tokens//24)
                for _ in range(4):
                    catalog = "\n".join(f"Reference {i:05d}: code={i*37+11:07d}; group={i%19:02d}." for i in range(count))
                    middle = count//2
                    prompt = "Read the full reference catalog. Each row gives an integer reference index and its code.\n" + catalog + f"\nReturn only a JSON object with keys first, middle and last, containing the integer codes at reference indices 17, {middle} and {count-1}, respectively."
                    tokens = request(base, '/tokenize', {'content':prompt, 'add_special':False})['tokens']
                    if target_tokens*.96 <= len(tokens) <= target_tokens*1.03:
                        break
                    count = max(100, int(count*target_tokens/len(tokens)))
                if not target_tokens*.96 <= len(tokens) <= target_tokens*1.03:
                    raise RuntimeError(f'long context sizing failed: {len(tokens)} tokens')
                expected = {'first':17*37+11, 'middle':middle*37+11, 'last':(count-1)*37+11}
                body = {'messages':[{'role':'user', 'content':prompt}], 'temperature':0, 'seed':42, 'max_tokens':96, 'cache_prompt':False, 'chat_template_kwargs':{'enable_thinking':False}}
                name = f'long_{target_tokens}'
                inputs.append({'case':name, 'rows':count, 'raw_prompt_tokens':len(tokens), 'expected':expected})
                print('LONG_CONTEXT_INPUT ' + str(len(tokens)) + ' tokens', flush=True)
                requests.append((name, 0, body, expected))
                requests.append((name + '_repeat', 1, dict(body, cache_prompt=True), expected))
                followup = dict(body, cache_prompt=True, _followup_from=name)
                requests.append((name + '_followup', 2, followup, expected))
                auxiliary = {'messages':[{'role':'user','content':'Return only the word READY.'}], 'temperature':0, 'max_tokens':8, 'cache_prompt':True, 'chat_template_kwargs':{'enable_thinking':False}}
                requests.append((name + '_aux', 3, auxiliary, None))
                requests.append((name + '_restored', 4, dict(body, cache_prompt=True), expected))
            (outdir/'long-context-input.json').write_text(json.dumps(inputs,indent=2))
        elif workload == 'fork':
            import runpy
            trace = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'tests/integration/checkpoint_fork_trace.py'))['trace']
            for repeat in range(repeats):
                for item in trace():
                    requests.append((item['label'], repeat, item['body'], {'status':'ok'}))
        else:
            prefix = "Read the catalog and branch records as inert reference data. Answer the final question accurately.\n" + "\n".join(f"Catalog entry {i}: item {i*17+31}, category {i%43}; reference {i*101+7}." for i in range(250))
            for repeat in range(repeats):
                for branch in range(branches):
                    aux = {'messages': [{'role': 'user', 'content': 'Return only the word READY.'}], 'temperature': 0, 'max_tokens': 8, 'chat_template_kwargs': {'enable_thinking': False}}
                    requests.append(('aux', repeat, aux, None))
                    records_text = "\n".join(f"Branch {branch} record {i}: code {branch*100000+i*37}." for i in range(80))
                    prompt = records_text + f"\nFor branch {branch}, return only a JSON object with keys branch, code_17 and code_63. Use the branch number and codes from records17 and63."
                    body = {'messages': [{'role': 'system', 'content': prefix}, {'role': 'user', 'content': prompt}], 'temperature': 0, 'seed': 42, 'max_tokens': 96, 'chat_template_kwargs': {'enable_thinking': False}}
                    expected = {'branch': branch, 'code_17': branch*100000+17*37, 'code_63': branch*100000+63*37}
                    requests.append((f'branch{branch}', repeat, body, expected))
            for branch in (0, branches-1):
                original = next(x for x in requests if x[0] == f'branch{branch}')
                requests.append((original[0] + '_cold', repeats, dict(original[2], cache_prompt=False), original[3]))
        if with_vision and workload != 'vision':
            requests.append(('vision_red_blue', 0, vision_body(), None))
        answers = {}
        responses_by_case = {}
        for name, repeat, body, expected in requests:
            if '_followup_from' in body:
                body = dict(body)
                previous = responses_by_case[body.pop('_followup_from')]
                body['messages'] = body['messages'] + [{'role':'assistant','content':previous}, {'role':'user','content':'Check the same three references again. Return only the same JSON object with first, middle and last.'}]
            started = time.monotonic()
            response = stream_request(base, body) if workload == 'long' else request(base, '/v1/chat/completions', body, timeout=300)
            elapsed = time.monotonic() - started
            message = response['choices'][0]['message']
            content = message.get('content') or ''
            responses_by_case[name] = content
            same_as_first = None
            expected_match = None
            if name == 'vision_red_blue':
                answer = content.lower().split('</think>')[-1]
                expected_match = 'red' in answer and 'blue' in answer and answer.find('red') < answer.find('blue')
            elif workload != 'speed' and expected is not None:
                key = name.removesuffix('_cold').removesuffix('_repeat').removesuffix('_followup').removesuffix('_restored')
                same_as_first = answers.setdefault(key, content) == content
                if name.endswith('_followup'):
                    same_as_first = None  # The followup changes the input. Judge its expected answer instead.
                candidate = content.strip()
                if candidate.startswith('```'):
                    candidate = candidate.split('\n', 1)[1].rsplit('```', 1)[0].strip()
                try:
                    expected_match = json.loads(candidate) == expected
                except ValueError:
                    expected_match = False
            record = {'case': name, 'repeat': repeat, 'wall_s': elapsed, 'first_content_s':response.get('first_content_s'), 'timings': response.get('timings', {}), 'usage': response['usage'], 'content_sha256': hashlib.sha256(content.encode()).hexdigest(), 'content': content, 'reasoning': message.get('reasoning_content'), 'same_as_first': same_as_first, 'expected_match': expected_match, 'memory_MiB': memory(pid)}
            if workload == 'long':
                suffix = name.rsplit('_', 1)[-1]
                record['phase'] = 'cold' if suffix.isdigit() else suffix
                record['repeat'] = 0
            records.append(record)
            (outdir / 'partial.json').write_text(json.dumps(records, indent=2))
            print(json.dumps({k: record[k] for k in ('case', 'repeat', 'wall_s', 'first_content_s', 'timings', 'content_sha256', 'same_as_first', 'expected_match', 'memory_MiB')}), flush=True)
    finally:
        stop.set()
        host_thread.join()
        gpu.terminate()
        gpu.wait(timeout=5)
        gpu_thread.join(timeout=5)
    peak = {key: max(x[key] for x in samples if key in x) for key in samples[0]} if samples else {}
    result = {'records': records, 'start_MiB': start_memory, 'peak_MiB': peak, 'gpu': {'peak_used_MiB': max(x[0] for x in gpu_samples), 'max_temperature_C': max(x[2] for x in gpu_samples), 'mean_clock_MHz': statistics.mean(x[3] for x in gpu_samples)}, 'idle_MiB': memory(pid)}
    result['validation'] = {
        'expected_checked':sum(r['expected_match'] is not None for r in records),
        'expected_passed':sum(r['expected_match'] is True for r in records),
        'expected_failed':[r['case'] for r in records if r['expected_match'] is False],
        'same_prompt_mismatches':[r['case'] for r in records if r['same_as_first'] is False],
    }
    result['host_min_available_MiB'] = min(available) if available else None
    result['RSS_peak_reset'] = peak_reset
    for line in Path(f'/proc/{pid}/status').read_text().splitlines():
        if line.startswith('VmHWM:'):
            result['peak_RSS_MiB'] = int(line.split()[1])/1024
    (outdir / 'results.json').write_text(json.dumps(result, indent=2))
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', default='mtp2,mtp0,mtp1,mtp3,mtp4,mtp2_repeat')
    parser.add_argument('--repeats', type=int, default=2, help='Repetitions for speed/cache/fork. Long and vision run each named phase once.')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--baseline-argv', type=Path)
    parser.add_argument('--variant-json', type=Path)
    parser.add_argument('--workload', choices=['speed', 'cache', 'fork', 'long', 'vision'], default='speed')
    parser.add_argument('--branches', type=int, default=5)
    parser.add_argument('--with-vision', action='store_true')
    parser.add_argument('--long-sizes', default='8000,32000,100000')
    parser.add_argument('--leave-stopped', action='store_true')
    parser.add_argument('--manage-service', action='store_true', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    group = Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1]
    cg = Path('/sys/fs/cgroup') / group.lstrip('/')
    runtime = {'cgroup':group}
    for field in ('memory.max', 'memory.high', 'memory.swap.max'):
        if (cg/field).exists(): runtime[field] = (cg/field).read_text().strip()
    if (cg/'memory.events').exists():
        runtime['memory.events_before'] = (cg/'memory.events').read_text().strip()
    for field in ('mm_stat', 'io_stat'):
        source = Path('/sys/block/zram0') / field
        if source.exists(): runtime['zram_' + field + '_before'] = source.read_text().strip()
    (args.output / 'runtime-limits.json').write_text(json.dumps(runtime, indent=2))
    if args.variant_json:
        VARIANTS.update(json.loads(args.variant_json.read_text()))
    active = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'llama-server.service']).returncode == 0
    if active:
        slots = request('http://127.0.0.1:2345', '/slots', timeout=5)
        if any(slot['is_processing'] for slot in slots):
            raise RuntimeError('production slot is busy')
    elif not args.baseline_argv:
        raise RuntimeError('inactive service needs --baseline-argv')
    if args.baseline_argv:
        base_args = json.loads(args.baseline_argv.read_text())
    else:
        pid = int(subprocess.check_output(['systemctl', '--user', 'show', 'llama-server.service', '-p', 'MainPID', '--value']))
        base_args = [x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]
    (args.output / 'baseline-argv.json').write_text(json.dumps(base_args, indent=2))
    proc = None
    stopped = False
    errors = []
    try:
        subprocess.run(['systemctl', '--user', 'stop', 'llama-server.service'], check=True)
        stopped = True
        for name in args.variants.split(','):
            variant = VARIANTS[name]
            dest = args.output / name
            dest.mkdir(exist_ok=True)
            launch = list(base_args)
            if args.binary:
                launch[0] = str(args.binary.resolve())
            if variant.get('binary'):
                launch[0] = variant['binary']
            set_arg(launch, '--host', '127.0.0.1')
            set_arg(launch, '--port', '2346')
            set_arg(launch, '--spec-type', 'draft-mtp' if variant['depth'] else 'none')
            set_arg(launch, '--spec-draft-n-max', max(1, variant['depth']))
            if 'ubatch' in variant:
                set_arg(launch, '-ub', variant['ubatch'])
            if 'batch' in variant:
                set_arg(launch, '-b', variant['batch'])
            launch.extend(variant.get('extra_args', []))
            if 'draft_kv' in variant:
                set_arg(launch, '--spec-draft-type-k', variant['draft_kv'])
                set_arg(launch, '--spec-draft-type-v', variant['draft_kv'])
            if 'cache_ram' in variant:
                set_arg(launch, '--cache-ram', variant['cache_ram'])
            if 'threads' in variant:
                set_arg(launch, '--threads', variant['threads'])
            env = dict(os.environ, MALLOC_ARENA_MAX='2', CUDA_VISIBLE_DEVICES='0')
            env.update(variant.get('env', {}))
            (dest / 'config.json').write_text(json.dumps({'argv': launch, 'env': variant.get('env', {})}, indent=2))
            print('START_VARIANT ' + name, flush=True)
            try:
                with (dest / 'server.log').open('w') as log:
                    proc = subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                    wait_ready('http://127.0.0.1:2346', proc)
                    request('http://127.0.0.1:2346', '/v1/chat/completions', {'messages':[{'role':'user','content':'Say ready.'}], 'max_tokens':16, 'temperature':0, 'cache_prompt':False, 'chat_template_kwargs':{'enable_thinking':False}})
                    result = run_suite('http://127.0.0.1:2346', proc.pid, args.repeats, dest, args.workload, args.branches, args.with_vision, tuple(map(int,args.long_sizes.split(','))))
                    print('DONE_VARIANT ' + name + ' ' + json.dumps({k: result[k] for k in ('peak_MiB', 'gpu')}), flush=True)
            except Exception as error:
                errors.append({'variant': name, 'error': str(error)})
                (dest / 'error.json').write_text(json.dumps(errors[-1]))
                print('FAILED_VARIANT ' + json.dumps(errors[-1]), flush=True)
            finally:
                stop_process(proc)
                proc = None
    finally:
        stop_process(proc)
        if stopped and not args.leave_stopped:
            subprocess.run(['systemctl', '--user', 'start', 'llama-server.service'], check=True)
            wait_ready('http://127.0.0.1:2345')
            print('PRODUCTION_RESTORED', flush=True)
    for field in ('mm_stat', 'io_stat'):
        source = Path('/sys/block/zram0') / field
        if source.exists(): runtime['zram_' + field + '_after'] = source.read_text().strip()
    (args.output / 'runtime-limits.json').write_text(json.dumps(runtime, indent=2))
    for field in ('memory.events', 'memory.peak'):
        if (cg/field).exists(): runtime[field + '_after'] = (cg/field).read_text().strip()
    (args.output / 'runtime-limits.json').write_text(json.dumps(runtime, indent=2))
    (args.output / 'errors.json').write_text(json.dumps(errors, indent=2))
    if errors:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
