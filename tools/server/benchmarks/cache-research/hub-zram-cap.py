#!/usr/bin/env python3
"""Inspect the hub's zram resident cap, or explicitly apply/restore the cap fix.

Default mode is read-only. Requires root for --apply or --restore.
Does not stop swap, reset zram, resize devices, or restart any service.
"""
import argparse
import base64
import datetime
import hashlib
import json
import os
import pathlib
import re
import stat
import tempfile
import time

CONFIG = pathlib.Path('/etc/systemd/zram-generator.conf')
DEVICE = pathlib.Path('/sys/block/zram0')
BACKUPS = pathlib.Path('/var/backups/llama-zram-cap')

def digest(data): return hashlib.sha256(data).hexdigest()

def stats():
    memory = [int(x) for x in (DEVICE / 'mm_stat').read_text().split()]
    io = [int(x) for x in (DEVICE / 'io_stat').read_text().split()]
    return {'logical_bytes':int((DEVICE / 'disksize').read_text()), 'original_bytes':memory[0],
            'compressed_bytes':memory[1], 'resident_bytes':memory[2], 'resident_limit_bytes':memory[3],
            'resident_peak_bytes':memory[4], 'failed_reads':io[0], 'failed_writes':io[1]}

def replacement(original):
    text = original.decode('utf-8')
    section = re.search(r'(?m)^\s*\[zram0\]\s*$', text)
    if not section: raise RuntimeError('No [zram0] section. No changes made.')
    following = re.search(r'(?m)^\s*\[.*\]\s*$', text[section.end():])
    end = section.end() + following.start() if following else len(text)
    body = text[section.end():end]
    keys = re.findall(r'(?m)^[ \t]*zram-resident-limit[ \t]*=', body)
    if len(keys) != 1: raise RuntimeError('Expected one zram-resident-limit in [zram0]. No changes made.')
    changed, count = re.subn(r'(?m)^([ \t]*zram-resident-limit[ \t]*=[ \t]*)3000([ \t]*(?:#.*)?)(\r?$)',
                            r'\g<1>0\2\3', body)
    if count != 1: raise RuntimeError('Expected zram-resident-limit=3000. No changes made.')
    return (text[:section.end()] + changed + text[end:]).encode('utf-8')

def atomic_config(data, info):
    descriptor, temporary = tempfile.mkstemp(prefix='.zram-cap-', dir=CONFIG.parent)
    try:
        os.fchmod(descriptor, info['mode'])
        os.fchown(descriptor, info['uid'], info['gid'])
        with os.fdopen(descriptor, 'wb') as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, CONFIG)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

def set_limit(value):
    (DEVICE / 'mem_limit').write_text(str(value) + '\n')
    current = stats()['resident_limit_bytes']
    if current != value: raise RuntimeError(f'Kernel reports resident limit {current}, expected {value}')

def observe(before, seconds):
    time.sleep(1)
    settled = stats()
    time.sleep(seconds)
    after = stats()
    result = {'before':before, 'after':after, 'observe_seconds':seconds,
              'failed_write_delta_since_change':after['failed_writes']-before['failed_writes'],
              'failed_write_delta_after_settle':after['failed_writes']-settled['failed_writes']}
    print(json.dumps(result, indent=2))
    if result['failed_write_delta_after_settle']:
        print('Resident cap changed, but swap writes still fail. Keep this evidence and investigate. The script did not reset counters.')
        return 2
    print('PASS: requested resident limit is active and the failed-write counter stayed stable during observation.')
    return 0

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--restore', type=pathlib.Path, metavar='BACKUP_JSON')
    parser.add_argument('--observe-seconds', type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.observe_seconds <= 30: parser.error('observe-seconds must be 1..30')
    before = stats()
    if not args.apply and not args.restore:
        print(json.dumps({'config':str(CONFIG), 'config_text':CONFIG.read_text(), 'runtime':before,
                         'mode':'read-only', 'planned_change':'[zram0] zram-resident-limit 3000 -> 0, live mem_limit -> 0'}, indent=2))
        return 0
    if os.geteuid() != 0: raise SystemExit('Root is required. No changes made.')
    if CONFIG.is_symlink() or not CONFIG.is_file(): raise RuntimeError('Expected a regular configuration file. No changes made.')
    current = CONFIG.read_bytes()
    if args.apply:
        updated = replacement(current)
        file_info = CONFIG.stat()
        info = {'mode':stat.S_IMODE(file_info.st_mode), 'uid':file_info.st_uid, 'gid':file_info.st_gid}
        record = {'schema':1, 'config':str(CONFIG), 'original_base64':base64.b64encode(current).decode('ascii'),
                  'original_sha256':digest(current), 'patched_sha256':digest(updated),
                  'original_metadata':info, 'runtime_before':before}
        BACKUPS.mkdir(mode=0o700, parents=True, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        backup = BACKUPS / (stamp + '.json')
        descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as output:
            json.dump(record, output, indent=2)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        atomic_config(updated, info)
        try: set_limit(0)
        except Exception:
            atomic_config(current, info)
            raise
        print('Backup:', backup)
        print('Changed only the zram resident limit. The device size, swap priority, and services are unchanged.')
    else:
        record = json.loads(args.restore.read_text())
        if record.get('schema') != 1 or record.get('config') != str(CONFIG): raise RuntimeError('Unexpected backup format')
        original = base64.b64decode(record['original_base64'], validate=True)
        if digest(original) != record['original_sha256'] or digest(current) != record['patched_sha256']:
            raise RuntimeError('Configuration changed since the backup. Refusing to overwrite it.')
        old_limit = record['runtime_before']['resident_limit_bytes']
        if old_limit and before['resident_bytes'] >= old_limit:
            raise RuntimeError('Restoring the old resident cap would reject writes immediately. No changes made.')
        if before['resident_limit_bytes'] != 0: raise RuntimeError('Live resident limit changed after the fix. No changes made.')
        set_limit(old_limit)
        try: atomic_config(original, record['original_metadata'])
        except Exception:
            set_limit(0)
            raise
        print('Restored the exact configuration and previous runtime cap from', args.restore)
        print('The old resident cap can reproduce the original failure under pressure.')
    return observe(before, args.observe_seconds)

if __name__ == '__main__': raise SystemExit(main())
