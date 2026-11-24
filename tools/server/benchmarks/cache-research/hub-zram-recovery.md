# Hub zram resident-cap failure

## Observed on 2026-11-24

The hub runs Linux `6.18.54-2-lts`. Its configuration advertises a 16000 MiB zram swap device, with Zstd and priority 100, but caps the compressed resident allocation at 3000 MiB. A separate 16 GiB disk swap file has priority 10 and was unused.

During the cache tests, `/sys/block/zram0/mm_stat` reported a resident peak of 3,145,850,880 bytes against a limit of 3,145,728,000 bytes. `/sys/block/zram0/io_stat` recorded 47,732,577 failed writes. Kernel logs reported `Write-error on swap-device (253:0:...)`.

The Linux driver returns `-ENOMEM` when a new compressed allocation exceeds the resident cap. The swap allocator still sees free logical slots in the higher-priority zram device. This explains the current write failures despite free disk swap. It does not establish the cause of every historical application crash.

## Immediate administrator fix

`hub-zram-cap.py` is read-only without arguments. The prepared script has not been executed with `--apply` or `--restore`.

```sh
python3 tools/server/benchmarks/cache-research/hub-zram-cap.py
sudo python3 tools/server/benchmarks/cache-research/hub-zram-cap.py --apply
```

Apply mode:

1. Require root and the expected single `zram-resident-limit=3000` setting in `[zram0]`.
2. Save the exact original configuration, metadata, hashes, and runtime cap under `/var/backups/llama-zram-cap/`.
3. Atomically change only that setting to `0`.
4. Write `0` to `/sys/block/zram0/mem_limit` and verify the fourth `mm_stat` field.
5. Observe the failed-write counter for ten seconds after a one-second settling period.

Changing `mem_limit` is supported while zram is active. This operation does not drain swap, reset the device, resize it, change its priority, or restart any service. The saved configuration persists the fix for the next setup. No daemon restart is needed to apply the runtime value.

For rollback, pass the backup path printed by apply mode:

```sh
sudo python3 tools/server/benchmarks/cache-research/hub-zram-cap.py --restore /var/backups/llama-zram-cap/PRINTED_BACKUP.json
```

Rollback refuses unrelated configuration edits and refuses to restore a cap at or below the current resident allocation. The original cap can reproduce the failure under pressure. A stable counter while idle is an initial check, not a replacement for repeating the workload.

## Current workaround

The working account cannot use passwordless sudo. Root confirmed that a user service can set `MemorySwapMax=0` and obtains `memory.swap.max=0`. Applying that setting to the llama service prevents its own anonymous pages from entering the broken swap path. It does not repair global zram or prevent RAM exhaustion, so process and cache bounds still matter.

## Later capacity adjustment

A conservative later configuration is `zram-size=4096`, `zram-resident-limit=0`, Zstd, priority 100, with the existing disk swap at priority 10. Limiting logical capacity lets normal swap selection reach the disk device after zram fills. Resident bytes remain bounded by the logical payload size plus metadata, without a lower allocation cap that rejects writes early.

Changing zram disk size or compression algorithm requires draining and resetting the device. Do this separately with the model stopped, adequate reclaimable RAM, and disk swap left active. Do not use `swapoff -a`. `swapoff` brings pages through RAM. It does not directly migrate all zram data to the disk swap file. Active swap priority also cannot be changed in place. An empty disk swap file can be deactivated and reactivated at a higher priority as a separate temporary measure.

## Primary sources

- [Linux zram documentation](https://docs.kernel.org/admin-guide/blockdev/zram.html): runtime `mem_limit`, zero meaning unlimited, statistics, reset requirements.
- [Linux v6.18 zram driver](https://github.com/torvalds/linux/blob/v6.18/drivers/block/zram/zram_drv.c): `zram_can_store_page`, write failure with `-ENOMEM`, and initialized-device disk-size rejection.
- [Linux v6.18 swap I/O](https://github.com/torvalds/linux/blob/v6.18/mm/page_io.c): failed swap writes leave the page dirty and report the observed error.
- [zram-generator configuration](https://github.com/systemd/zram-generator/blob/main/man/zram-generator.conf.md): resident limit, disk size, and priority settings.
- [Linux swapon API](https://man7.org/linux/man-pages/man2/swapon.2.html): priority selection, `EBUSY` for an already-active area, and `CAP_SYS_ADMIN` requirements.
- [util-linux swapoff documentation](https://man7.org/linux/man-pages/man8/swapoff.8.html): insufficient-memory exit status and swap-file handling.
