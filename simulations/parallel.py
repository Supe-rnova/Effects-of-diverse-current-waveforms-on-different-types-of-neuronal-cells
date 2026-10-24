"""
Run a per-model job in separate Python processes (one model per process,
pipeline change #6), several at a time, and collect their JSON results.

A worker script must, when called as   python <script> --one <MODEL> [args...]
print exactly one line   RESULT <json>   to stdout.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_per_model(script: str | Path, models, extra_args=(), max_parallel: int | None = None,
                  verbose: bool = True) -> dict:
    max_parallel = max_parallel or max(1, (os.cpu_count() or 2) - 2)
    pending = list(models)
    running, results = {}, {}
    t_start = time.perf_counter()
    while pending or running:
        while pending and len(running) < max_parallel:
            m = pending.pop(0)
            p = subprocess.Popen([sys.executable, str(script), "--one", m, *map(str, extra_args)],
                                 cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            running[m] = (p, time.perf_counter())
            if verbose:
                print(f"  started {m}", flush=True)
        for m, (p, t0) in list(running.items()):
            if p.poll() is None:
                continue
            out, err = p.communicate()
            del running[m]
            line = [ln for ln in out.splitlines() if ln.startswith("RESULT ")]
            if p.returncode != 0 or not line:
                print(f"  [FAIL] {m} crashed:\n{err[-3000:]}", flush=True)
                results[m] = None
            else:
                results[m] = json.loads(line[-1][7:])
                if verbose:
                    print(f"  finished {m} in {time.perf_counter() - t0:.0f} s", flush=True)
        time.sleep(0.5)
    if verbose:
        print(f"  all done in {time.perf_counter() - t_start:.0f} s")
    return results
