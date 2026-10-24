"""
Step 6 - demonstrate and verify the injection framework.

Plays one example of every waveform family into every model (amplitudes in
units of each cell's rheobase), plus a dendritic injection for Purkinje/CA1,
and checks for every run that the injected charge equals the waveform's charge
exactly, and that saving/loading a result round-trips.

Run from the project root (WSL):
    python simulations/demo_injection.py
Outputs:
    results/figures/injection_demo/<model>.png   voltage with the waveform underneath
    results/logs/step6_injection.log            checks + any error tracebacks
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import DT, FIGURES_DIR, LOGS_DIR, MODEL_NAMES, RESULTS_DIR  # noqa: E402

FIG_DIR = FIGURES_DIR / "injection_demo"
FIG_DIR.mkdir(parents=True, exist_ok=True)
LOG = LOGS_DIR / "step6_injection.log"


def demo_waveforms(u: float, ud: float | None):
    """Example of each family; u = somatic rheobase (nA), ud = dendritic rheobase."""
    from waveforms import generator as g
    T = 1200.0
    out = [
        ("step 1.5x", "soma", g.step(1.5 * u, 100, 800, T)),
        ("hyperpolarising -2x + release", "soma", g.hyperpolarising_step(-2 * u, 100, 600, T)),
        ("ramp 0 -> 2x", "soma", g.ramp(0, 2 * u, 100, 1000, T)),
        ("sinusoid 10 Hz, 1x on 0.5x DC", "soma", g.sinusoid(1.0 * u, 10, 100, 1000, T, dc=0.5 * u)),
        ("pulse train 130 Hz biphasic, 0.5 ms, 4x", "soma",
         g.pulse_train(4 * u, 0.5, 130, 100, 1000, T, polarity="biphasic", interphase=0.1)),
        ("OU noise mean 0.8x, sd 0.4x, tau 5 ms", "soma", g.ou_noise(0.8 * u, 0.4 * u, 5.0, 100, 1000, T, seed=1)),
        ("synaptic double-exp 3x, Poisson 20 Hz", "soma",
         g.synaptic(3 * u, g.poisson_times(20, 100, 1000, seed=2), T, tau_rise=0.5, tau_decay=5.0)),
    ]
    if ud is not None:
        out.append(("DENDRITE step 1.5x dend-rheobase", "dend", g.step(1.5 * ud, 100, 800, T)))
    return out


def run_one(name: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.spikes import detect_spikes
    from simulations.injection import Simulator, load_result, save_result

    t0 = time.perf_counter()
    sim = Simulator(name)
    u = sim.unit("soma")
    ud = sim.unit("dend") if "dend" in sim.sites else None
    cond = "held" if "held" in sim.conditions() else "native"
    sim.prepare(cond)
    checks, rows = [], []
    wfs = demo_waveforms(u, ud)
    fig, axes = plt.subplots(2 * len(wfs), 1, figsize=(11, 2.1 * len(wfs) * 1.33),
                             gridspec_kw={"height_ratios": [3, 1] * len(wfs)}, sharex=True)
    for k, (label, site, w) in enumerate(wfs):
        res = sim.run(w, site=site)
        n = len(w.i)
        exact = float(np.abs(res["i"][:n] - w.i).max())
        dq = abs(res["meta"]["charge_pC"] - w.charge_pC())
        s = detect_spikes(res["t"], res["v"]["soma"])
        checks.append({"waveform": label, "site": site, "max_abs_current_error_nA": exact,
                       "charge_error_pC": dq, "n_spikes_soma": len(s)})
        av, ai = axes[2 * k], axes[2 * k + 1]
        av.plot(res["t"], res["v"]["soma"], lw=.6, label="soma")
        if "dend" in res["v"]:
            av.plot(res["t"], res["v"]["dend"], lw=.5, alpha=.7, label="dend")
        av.set_ylabel("mV")
        av.set_title(f"{label}  ({len(s)} somatic spikes)", fontsize=9, loc="left")
        ai.plot(res["t"], res["i"], lw=.6, color="k")
        ai.set_ylabel("nA")
        if k == 0:
            av.legend(fontsize=7, loc="upper right")
    # save/load round trip on the last result
    tmp = RESULTS_DIR / "data" / "_roundtrip_test" / f"{name}.npz"
    save_result(tmp, res)
    back = load_result(tmp)
    rt = float(np.abs(back["v"]["soma"] - res["v"]["soma"]).max())
    axes[-1].set_xlabel("ms")
    fig.suptitle(f"{name} ({cond} condition, rheobase unit {u:.3f} nA) - injection framework demo",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=110)
    ok = all(c["max_abs_current_error_nA"] < 1e-9 and c["charge_error_pC"] < 1e-6 for c in checks) and rt < 1e-3
    return {"model": name, "condition": cond, "unit_nA": u, "checks": checks,
            "roundtrip_max_err_mV": rt, "pass": ok, "wall_s": round(time.perf_counter() - t0, 1)}


def main():
    from simulations.parallel import run_per_model
    models = sys.argv[1:] or MODEL_NAMES
    print(f"injection demo for {models} ...")
    results = run_per_model(__file__, models)
    lines = []
    for m in models:
        r = results.get(m)
        if not r:
            lines.append(f"[FAIL] {m}: crashed (see traceback above / in this log)")
            continue
        lines.append(f"[{'OK' if r['pass'] else '!!'}] {m} ({r['condition']}, unit {r['unit_nA']:.3f} nA, "
                     f"{r['wall_s']} s): save/load error {r['roundtrip_max_err_mV']:.1e} mV")
        for c in r["checks"]:
            lines.append(f"      {c['waveform']:<42s} [{c['site']}] spikes={c['n_spikes_soma']:<4d} "
                         f"|i - waveform|max={c['max_abs_current_error_nA']:.1e} nA  "
                         f"charge err={c['charge_error_pC']:.1e} pC")
    ok = all(results.get(m) and results[m]["pass"] for m in models)
    lines.append("INJECTION FRAMEWORK OK" if ok else "SOME CHECKS FAILED")
    text = "\n".join(lines)
    print(text)
    LOG.write_text(text + "\n\n" + json.dumps(results, indent=1, default=str))
    print(f"\nfigures: {FIG_DIR}\nlog: {LOG}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        try:
            print("RESULT " + json.dumps(run_one(sys.argv[2]), default=float))
        except Exception:
            tb = traceback.format_exc()
            (LOGS_DIR / f"step6_{sys.argv[2]}_error.log").write_text(tb)
            print(tb, file=sys.stderr)
            sys.exit(1)
    else:
        main()
