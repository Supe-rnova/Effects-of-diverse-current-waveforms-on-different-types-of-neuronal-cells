"""
Step 7a - single-pulse threshold per model and injection site.

A 0.5 ms pulse needs far more current than a 1 s step (strength-duration
relation), so pulse-train amplitudes are scaled to this threshold instead of
the step rheobase. Bisection on the amplitude of ONE monophasic pulse.

Run (normally called automatically by simulations/batch.py):
    python simulations/pulse_threshold.py
Output: results/data/pulse_thresholds.json
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import DATA_DIR, LOGS_DIR, MODEL_NAMES  # noqa: E402

OUT = DATA_DIR / "pulse_thresholds.json"
PULSE_WIDTH = 0.5      # ms - must match config/experiments.py


def threshold(sim, site, width=PULSE_WIDTH, rel_tol=0.01):
    from analysis.spikes import detect_spikes
    from waveforms import generator as g

    def fires(amp):
        w = g.pulse_train(amp, width, 10.0, 20.0, 50.0, 80.0)       # exactly one pulse at 20 ms
        res = sim.run(w, site=site)
        s = detect_spikes(res["t"], res["v"]["soma"])
        return len(s.in_window(20.0, 45.0)) > 0

    lo, hi = 0.0, 2.0 * sim.unit(site)
    while not fires(hi):
        lo, hi = hi, hi * 2
        if hi > 500 * sim.unit(site):
            return None
    while (hi - lo) / hi > rel_tol:
        mid = 0.5 * (lo + hi)
        if fires(mid):
            hi = mid
        else:
            lo = mid
    return hi


def run_one(model: str) -> dict:
    from simulations.injection import Simulator
    t0 = time.perf_counter()
    sim = Simulator(model)
    cond = "held" if "held" in sim.conditions() else "native"
    sim.prepare(cond)
    out = {"model": model, "condition": cond, "width_ms": PULSE_WIDTH, "threshold_nA": {}}
    for site in sim.sites:
        thr = threshold(sim, site)
        out["threshold_nA"][site] = thr
        out.setdefault("ratio_to_rheobase", {})[site] = (thr / sim.unit(site)) if thr else None
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    return out


def main(models=None):
    from simulations.parallel import run_per_model
    models = models or MODEL_NAMES
    res = run_per_model(__file__, models, verbose=False)
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    old.update({m: r for m, r in res.items() if r})
    OUT.write_text(json.dumps(old, indent=2))
    for m in models:
        r = res.get(m)
        if r:
            print(f"  {m:9s} " + ", ".join(
                f"{s}: {v:.3f} nA ({r['ratio_to_rheobase'][s]:.1f}x rheobase)" if v else f"{s}: none"
                for s, v in r["threshold_nA"].items()))
        else:
            print(f"  {m}: FAILED")
    return old


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        try:
            print("RESULT " + json.dumps(run_one(sys.argv[2])))
        except Exception:
            (LOGS_DIR / f"step7_pulse_threshold_{sys.argv[2]}_error.log").write_text(traceback.format_exc())
            raise
    else:
        main(sys.argv[1:] or None)
