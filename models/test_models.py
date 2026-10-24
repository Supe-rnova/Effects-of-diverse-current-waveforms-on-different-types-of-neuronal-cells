"""
Step 2c - smoke test: build every model (each in its own process), run it with
no input and report size, resting behaviour and simulation speed.

Run from the project root:
    python models/test_models.py

Writes results/logs/model_summary.csv
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_one(name: str) -> dict:
    """Executed inside a fresh Python process."""
    import numpy as np
    from neuron import h
    from config.settings import DT
    from models.cells import build_cell

    cell = build_cell(name)
    soma = cell.sites["soma"]
    v = h.Vector().record(soma._ref_v)
    t = h.Vector().record(h._ref_t)
    h.dt = DT
    h.celsius = cell.celsius
    T = cell.settle_ms + 1000.0
    h.finitialize(cell.v_init)
    t0 = time.perf_counter()
    h.continuerun(T)
    wall = time.perf_counter() - t0
    v, t = np.array(v), np.array(t)
    after = t >= cell.settle_ms
    up = np.where((v[1:] > -20) & (v[:-1] <= -20))[0]
    spk = t[1:][up]
    n_spk = int(np.sum(spk >= cell.settle_ms))
    out = {
        "model": name,
        "modeldb": cell.modeldb,
        "celsius": cell.celsius,
        "sections": sum(1 for _ in h.allsec()),
        "segments": cell.n_segments(),
        "sites": {k: f"{s.sec.name()}({s.x:.3f})" for k, s in cell.sites.items()},
        "rest_or_mean_V_mV": round(float(v[after].mean()), 2),
        "spont_rate_Hz": round(n_spk / 1.0, 2),
        "expected_spontaneous": cell.spontaneous,
        "wall_s_per_sim_s": round(wall / (T / 1000.0), 2),
        "channels": list(cell.channels),
    }
    if "dend" in cell.sites:
        h.distance(0, soma)
        out["dend_distance_um"] = round(h.distance(cell.sites["dend"]), 1)
    return out


def main() -> None:
    import csv
    from config.settings import LOGS_DIR
    from models.cells import MODEL_NAMES

    rows = []
    for name in MODEL_NAMES:
        print(f"--- {name}: building and running (fresh process) ...", flush=True)
        proc = subprocess.run([sys.executable, __file__, "--one", name], cwd=ROOT,
                              capture_output=True, text=True)
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
        if proc.returncode != 0 or not line:
            print(proc.stdout[-2000:], proc.stderr[-3000:])
            print(f"[FAIL] {name}")
            continue
        r = json.loads(line[-1][7:])
        rows.append(r)
        state = (f"fires {r['spont_rate_Hz']} Hz" if r["spont_rate_Hz"] > 0
                 else f"silent at {r['rest_or_mean_V_mV']} mV")
        print(f"[ OK ] {name}: {r['sections']} sections / {r['segments']} segments, "
              f"{state} at {r['celsius']} degC, {r['wall_s_per_sim_s']} s wall per simulated s")
        print(f"       sites: {r['sites']}" + (f"  (dend at {r['dend_distance_um']} um)"
                                                if "dend_distance_um" in r else ""))
        print(f"       blockable channels: {r['channels']}")

    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(LOGS_DIR / "model_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                            for k, v in r.items()})
        print(f"\nsummary written to {LOGS_DIR / 'model_summary.csv'}")
    print("ALL MODELS OK" if len(rows) == len(MODEL_NAMES) else "SOME MODELS FAILED")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        print("RESULT " + json.dumps(run_one(sys.argv[2])))
    else:
        main()
