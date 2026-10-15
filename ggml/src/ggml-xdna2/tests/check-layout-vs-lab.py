#!/usr/bin/env python3
"""Check that the C++ weight layout (test-xdna2-layout --compare) is bit identical to the lab packer stage5/packu.py.
usage: check-layout-vs-lab.py <test-xdna2-layout binary> <lab stage5 dir> [workdir]"""
import os, subprocess, sys, tempfile
import numpy as np

exe, lab = sys.argv[1], sys.argv[2]
work = sys.argv[3] if len(sys.argv) > 3 else tempfile.mkdtemp()
BB = {4: 144, 5: 176, 6: 210, 8: 272}      # bytes per (row, 256 weights)
NAME = {4: "Q4_K", 5: "Q5_K", 6: "Q6_K", 8: "Q8_0"}
bad = 0
for qt in (4, 5, 6, 8):
    for N, K in ((512, 256), (1024, 4096), (2048, 768), (1536, 1024)):
        nbk = K // 256
        rng = np.random.default_rng(qt * 100000 + N + K)
        raw = rng.integers(0, 256, size=(N, nbk, BB[qt]), dtype=np.uint8)
        pre = os.path.join(work, f"{NAME[qt]}_{N}_{K}")
        raw.tofile(pre + ".blk.bin")
        # dummy activation file (content irrelevant for the weight layout)
        asz = K // 32 * 34 if qt == 8 else nbk * 292
        np.zeros(asz, np.uint8).tofile(pre + ".act.bin")
        subprocess.run([sys.executable, "-I", os.path.join(lab, "packu.py"), str(qt), pre + ".blk.bin", pre + ".act.bin",
                        str(N), str(K), "1", "2", pre], check=True, stdout=subprocess.DEVNULL)
        r = subprocess.run([exe, "--compare", NAME[qt], str(N), str(K), pre + ".blk.bin", pre + ".w.bin"])
        bad += r.returncode != 0
        os.remove(pre + ".blk.bin"); os.remove(pre + ".w.bin"); os.remove(pre + ".a.bin"); os.remove(pre + ".act.bin")
print("ALL IDENTICAL" if bad == 0 else f"{bad} MISMATCHES")
sys.exit(1 if bad else 0)
