# Hub deployment

Target: the RTX 3060 hub, not the V100 hosts. Build in Release mode for CUDA architecture 86.

The installed launcher keeps the target cache at Q8_0, changes only the MTP draft cache to Q4_0, and uses native MTP depth 2. Logical batch remains 2048 and microbatch remains 256. The context is configured at 110,000 tokens, with a 4,096 MiB parked-cache budget and three recurrent checkpoints under a separate 384 MiB cap. The full-precision image projector runs on the CPU.

`20-resident-cache.conf` sets `MemorySwapMax=0` for this service. It prevents the latency-sensitive cache from being swapped out. It does not impose a total RAM limit. The host-wide zram resident-cap repair is documented separately in `../../cache-research/hub-zram-recovery.md`.

Installed paths:

- Launcher: `/home/jirka/.config/llama.cpp/sensenova-normal-110k.sh`
- Drop-in: `/home/jirka/.config/systemd/user/llama-server.service.d/20-resident-cache.conf`
- Service: `llama-server.service`, under the user systemd manager
- API: port 2345
- Binary: `/home/jirka/programy/llama-sensenova-abliterated-build/build-cuda/bin/llama-server`

The original launcher and complete original binary/library directory are preserved under `/home/jirka/.cache/rtx3060-experiments-20261124/baseline-launcher.sh` and `baseline-bin`. The old binary needs that directory in `LD_LIBRARY_PATH` to load the matching old libraries. Copying the launcher alone changes launch flags but does not revert the binary.

The release branch omits the adaptive-MTP, pinned-copy, and rollback-ablation runtime prototypes. Those experiments remain on `rtx3060-experiments-20261124`. Use that branch to reproduce their flags. The checked-in benchmark configurations record the exact paths and environments used on this host.

Deployment verification is recorded in `../results/2026-11-24/deployment.json`: service state, effective flags, loaded binary identity, memory policy, health, and a real MTP completion.
