"""
Run the whole project end to end (Step 11).

    python main.py                      # every step; finished work is reused/skipped
    python main.py --list               # show the steps
    python main.py --from analysis      # start at a step (e.g. after changing the analysis)
    python main.py --only figures       # run a single step
    python main.py --skip demo          # skip optional steps

Every step is an independent script, so the pipeline can also be run piece by
piece. The batch (Step 7) is resumable: rerunning skips results that exist.
Expected time on a 12-core PC under WSL2: ~1.5 h, almost all of it Step 7.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("env",            "Step 1  environment check",             ["setup_env/check_env.py"]),
    ("models",         "Step 2  download + compile models",     ["models/fetch_models.py"]),
    ("smoke",          "Step 2  smoke-test models",             ["models/test_models.py"]),
    ("validate",       "Step 3  validate models",               ["models/validate_models.py"]),
    ("characterise",   "Step 4  characterise cells",            ["simulations/characterise.py"]),
    ("tests",          "Step 5  waveform unit tests",           ["-m", "pytest", "tests", "-q"]),
    ("waveforms",      "Step 5  plot waveforms",                ["waveforms/plot_waveforms.py"]),
    ("demo",           "Step 6  injection framework demo",      ["simulations/demo_injection.py"]),
    ("batch",          "Step 7  experiment batch (~70 min)",    ["simulations/batch.py"]),
    ("analysis",       "Step 8  analysis",                      ["analysis/run_analysis.py"]),
    ("figures",        "Step 9  figures",                       ["analysis/figures.py"]),
    ("provenance",     "Step 11 provenance record",             ["analysis/provenance.py"]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()
    names = [s[0] for s in STEPS]
    if a.list:
        for n, d, _ in STEPS:
            print(f"  {n:13s} {d}")
        return
    sel = STEPS
    if a.only:
        sel = [s for s in STEPS if s[0] == a.only]
    elif a.start:
        if a.start not in names:
            sys.exit(f"unknown step '{a.start}'. Options: {names}")
        sel = STEPS[names.index(a.start):]
    sel = [s for s in sel if s[0] not in a.skip]

    t_all = time.perf_counter()
    for n, desc, args in sel:
        print(f"\n{'=' * 70}\n{desc}   ({n})\n{'=' * 70}", flush=True)
        t0 = time.perf_counter()
        rc = subprocess.call([sys.executable, *args], cwd=ROOT)
        print(f"--- {n} finished in {(time.perf_counter() - t0) / 60:.1f} min (exit code {rc})")
        if rc != 0:
            sys.exit(f"\nStopped: step '{n}' failed. Fix it, then resume with:  python main.py --from {n}")
    print(f"\nPIPELINE COMPLETE in {(time.perf_counter() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
