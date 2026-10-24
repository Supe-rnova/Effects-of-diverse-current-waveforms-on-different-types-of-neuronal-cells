"""
Step 4 - characterise every cell.

For each model and condition it measures:
  * holding current   (STN 'held' condition: DC current that holds the cell silent at -70 mV)
  * settling time     (time from finitialize until the cell is at steady state)
  * baseline          (silent: resting V; spontaneous: rate and ISI CV)
  * input resistance and membrane time constant (small hyperpolarising step, ~3 mV)
  * rheobase          (bisection over 1 s step amplitude; soma, plus dendrite for Purkinje/CA1)

Conditions (pipeline change #1):
  native : the cell as published (no bias current)
  held   : only for spontaneously firing cells (STN) - holding current added so the
           cell is silent at -70 mV; rheobase is measured on top of it

The rheobase of the silent condition becomes the amplitude unit for Step 7
(amplitudes = 0.8x, 1x, 1.5x, 2x rheobase).

Run (from the project root, in WSL):
    python simulations/characterise.py
Outputs:
    results/data/characterisation.json
    results/figures/characterisation/<model>.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import DATA_DIR, FIGURES_DIR, MODEL_NAMES  # noqa: E402

FIG_DIR = FIGURES_DIR / "characterisation"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = DATA_DIR / "characterisation.json"

HOLD_TARGET_MV = -70.0
RHEO_STEP_MS = 1000.0     # step duration used for rheobase (plan: long rectangular step)
RHEO_REL_TOL = 0.01       # stop bisection when bracket is within 1 %
RIN_TARGET_MV = 3.0       # size of the hyperpolarising test deflection


# ===================================================================== measurements
def settling_time(t, v, spikes, spontaneous: bool) -> float:
    """Time after which the cell is at steady state (rounded up to 100 ms, +200 ms margin)."""
    if spontaneous and len(spikes) > 4:
        isi = np.diff(spikes)
        final = np.median(isi[-3:])
        bad = np.where(np.abs(isi - final) > 0.02 * final)[0]
        t_ss = spikes[bad[-1] + 1] if len(bad) else spikes[0]
    else:
        v_end = v[-int(len(v) * 0.05):].mean()
        off = np.where(np.abs(v - v_end) > 0.2)[0]
        t_ss = t[off[-1]] if len(off) else 0.0
    return float(np.ceil(t_ss / 100.0) * 100.0 + 200.0)


def find_holding_current(cell, eng, target=HOLD_TARGET_MV, settle=2500.0, n_iter=14):
    """Bisection for the DC current that holds the cell silent at `target` mV."""
    from analysis.spikes import detect_spikes
    lo, hi = -2.0, 0.0              # V(lo) < target < V(hi)
    log = []
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        eng.set_hold(mid)
        t, r = eng.settle(settle)
        v = r["soma"]
        last = t > settle - 1500
        n_spk = len(detect_spikes(t[last], v[last]))
        vm = v[t > settle - 500].mean()
        log.append((mid, vm, n_spk))
        if n_spk > 0 or vm > target:
            hi = mid                 # need more hyperpolarising current
        else:
            lo = mid
    i_hold = lo                      # most depolarised value that is still silent and <= target
    return float(i_hold), log


def baseline_stats(t, v):
    from analysis.spikes import detect_spikes
    s = detect_spikes(t, v)
    dur = (t[-1] - t[0]) / 1000.0
    out = {"mean_V_mV": float(v.mean()), "min_V_mV": float(v.min()), "max_V_mV": float(v.max()),
           "n_spikes": len(s), "rate_Hz": len(s) / dur}
    if len(s) > 2:
        isi = np.diff(s.times)
        out["isi_cv"] = float(isi.std() / isi.mean())
    return out


def input_resistance(eng, pre=100.0, dur=800.0):
    """Rin (MOhm) and tau_m (ms) from a hyperpolarising step scaled to ~3 mV deflection."""
    amp = -0.01
    for _ in range(3):
        t, r = eng.run_step("soma", amp, dur, pre=pre, post=200)
        v = r["soma"]
        base = v[(t > pre - 50) & (t < pre)].mean()
        ss = v[(t > pre + dur - 100) & (t < pre + dur)].mean()
        dv = base - ss
        if abs(dv) < 1e-6:
            amp *= 10
            continue
        if 0.6 * RIN_TARGET_MV < dv < 1.6 * RIN_TARGET_MV:
            break
        amp *= RIN_TARGET_MV / dv
    rin = dv / -amp
    # tau_m: time to reach 63.2 % of the peak deflection (robust to sag and to the
    # very fast HH response, where an exponential fit is ill-conditioned)
    win = (t >= pre) & (t <= pre + dur)
    tw, vw = t[win] - pre, v[win]
    peak = vw.min()
    target = base - 0.632 * (base - peak)
    hit = np.where(vw <= target)[0]
    tau = float(tw[hit[0]]) if len(hit) else float("nan")
    sag = (base - ss) / (base - peak) if base - peak > 0 else 1.0
    return {"Rin_MOhm": float(rin), "tau_m_ms": tau, "test_amp_nA": float(amp),
            "deflection_mV": float(dv), "sag_ratio_small_step": float(sag)}, (t, v)


def rheobase(eng, site, pre=100.0, dur=RHEO_STEP_MS):
    """Smallest step amplitude (nA) evoking >= 1 spike at the soma (bisection)."""
    from analysis.spikes import detect_spikes

    def spikes(amp):
        t, r = eng.run_step(site, amp, dur, pre=pre, post=100)
        s = detect_spikes(t, r["soma"])
        return s.in_window(pre, pre + dur + 20), (t, r["soma"])

    lo, hi = 0.0, 0.05
    sp, tr_hi = spikes(hi)
    while len(sp) == 0:
        lo, hi = hi, hi * 2
        if hi > 20:
            return None
        sp, tr_hi = spikes(hi)
    tr_lo = None
    n_runs = 0
    while (hi - lo) / hi > RHEO_REL_TOL:
        mid = 0.5 * (lo + hi)
        sp_mid, tr = spikes(mid)
        n_runs += 1
        if len(sp_mid):
            hi, sp, tr_hi = mid, sp_mid, tr
        else:
            lo, tr_lo = mid, tr
    if tr_lo is None:
        _, tr_lo = spikes(lo)
    return {"rheobase_nA": float(hi), "below_nA": float(lo),
            "n_spikes_at_rheobase": int(len(sp)),
            "first_spike_latency_ms": float(sp[0] - pre),
            "bisection_runs": n_runs}, tr_hi, tr_lo


# ===================================================================== per model
def run_one(name: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.spikes import detect_spikes
    from models.cells import build_cell
    from simulations.engine import Engine

    t_start = time.perf_counter()
    cell = build_cell(name)
    sites = tuple(s for s in ("soma", "dend") if s in cell.sites)
    eng = Engine(cell, sites=sites, record=("soma",))

    conditions = ["native", "held"] if cell.spontaneous else ["native"]
    res = {"model": name, "celsius": cell.celsius, "spontaneous": cell.spontaneous,
           "sites": {s: f"{cell.sites[s].sec.name()}({cell.sites[s].x:.3f})" for s in sites},
           "conditions": {}}
    figs = {}

    for cond in conditions:
        c = {}
        if cond == "held":
            i_hold, log = find_holding_current(cell, eng)
            c["i_hold_nA"] = i_hold
            c["hold_search"] = [list(map(float, x)) for x in log]
        else:
            c["i_hold_nA"] = 0.0
        eng.set_hold(c["i_hold_nA"])

        # settling time from a long run started at finitialize
        t_s, r_s = eng.settle(3000.0)
        sp = detect_spikes(t_s, r_s["soma"]).times
        spontaneous_here = len(detect_spikes(t_s[t_s > 1500], r_s["soma"][t_s > 1500])) > 0
        c["settle_ms"] = min(3000.0, settling_time(t_s, r_s["soma"], sp, spontaneous_here))
        c["silent"] = not spontaneous_here

        # re-settle with the measured settling time and save that state
        eng.settle(max(c["settle_ms"], 200.0))
        t_b, r_b = eng.run_baseline(2000.0)
        c["baseline"] = baseline_stats(t_b, r_b["soma"])
        traces = {"settle": (t_s, r_s["soma"]), "baseline": (t_b, r_b["soma"])}

        if c["silent"]:
            rin, tr = input_resistance(eng)
            c.update(rin)
            traces["rin"] = tr
            c["rheobase"] = {}
            for site in sites:
                rb = rheobase(eng, site)
                if rb is None:
                    c["rheobase"][site] = None
                    continue
                c["rheobase"][site], traces[f"rheo_{site}_hi"], traces[f"rheo_{site}_lo"] = rb
        res["conditions"][cond] = c
        figs[cond] = traces

    # amplitude unit for Step 7: somatic rheobase of the silent condition
    silent_cond = "held" if cell.spontaneous else "native"
    res["amplitude_condition"] = silent_cond
    res["amplitude_unit_nA"] = res["conditions"][silent_cond]["rheobase"]["soma"]["rheobase_nA"]
    if "dend" in sites and res["conditions"][silent_cond]["rheobase"].get("dend"):
        res["amplitude_unit_dend_nA"] = res["conditions"][silent_cond]["rheobase"]["dend"]["rheobase_nA"]
    res["wall_s"] = round(time.perf_counter() - t_start, 1)

    # ---------------------------------------------------------------- figure
    n_rows = len(conditions)
    fig, ax = plt.subplots(n_rows, 4, figsize=(18, 3.6 * n_rows), squeeze=False)
    for i, cond in enumerate(conditions):
        tr, c = figs[cond], res["conditions"][cond]
        lab = f"{cond} (I_hold = {c['i_hold_nA']:.3f} nA)"
        ax[i, 0].plot(*tr["settle"], lw=.6)
        ax[i, 0].axvline(c["settle_ms"], color="r", ls="--", lw=1)
        ax[i, 0].set(title=f"{lab}\nsettling: {c['settle_ms']:.0f} ms", xlabel="ms", ylabel="mV")
        b = c["baseline"]
        ax[i, 1].plot(*tr["baseline"], lw=.6)
        ax[i, 1].set(title=f"baseline: {b['rate_Hz']:.1f} Hz, mean {b['mean_V_mV']:.1f} mV", xlabel="ms")
        if c["silent"]:
            ax[i, 2].plot(*tr["rin"], lw=.8)
            ax[i, 2].set(title=f"Rin = {c['Rin_MOhm']:.1f} MOhm, tau = {c['tau_m_ms']:.1f} ms", xlabel="ms")
            for site, col in (("soma", "C0"), ("dend", "C1")):
                if f"rheo_{site}_hi" in tr:
                    rb = c["rheobase"][site]
                    ax[i, 3].plot(*tr[f"rheo_{site}_hi"], color=col, lw=.6,
                                  label=f"{site}: {rb['rheobase_nA']:.3f} nA")
                    ax[i, 3].plot(*tr[f"rheo_{site}_lo"], color=col, lw=.6, ls=":", alpha=.7)
            ax[i, 3].legend(fontsize=8)
            ax[i, 3].set(title="rheobase (solid) and 1% below (dotted)", xlabel="ms")
        else:
            for a in ax[i, 2:]:
                a.text(0.5, 0.5, "spontaneously active:\nsee 'held' condition",
                       ha="center", va="center", transform=a.transAxes)
                a.set_axis_off()
    fig.suptitle(f"{name} characterisation ({cell.celsius} degC)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=120)
    return res


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main():
    from simulations.parallel import run_per_model
    models = sys.argv[1:] or MODEL_NAMES
    print(f"characterising {models} ...")
    results = run_per_model(__file__, models)
    old = json.loads(OUT_FILE.read_text()) if OUT_FILE.exists() else {}
    old.update({m: r for m, r in results.items() if r})
    OUT_FILE.write_text(json.dumps(old, indent=2, default=_jsonable))

    print("\n%-9s %-7s %9s %9s %9s %8s %11s %11s" % ("model", "cond", "I_hold", "settle", "Rin", "tau", "rheo_soma", "rheo_dend"))
    for m in models:
        r = results.get(m)
        if not r:
            print(f"{m}: FAILED")
            continue
        for cond, c in r["conditions"].items():
            rb = c.get("rheobase", {})
            f = lambda s: f"{rb[s]['rheobase_nA']:.3f} nA" if rb.get(s) else "-"
            extra = (f"{c['Rin_MOhm']:7.1f}MO {c['tau_m_ms']:6.1f}ms" if c["silent"]
                     else f"   fires {c['baseline']['rate_Hz']:.1f} Hz (CV {c['baseline'].get('isi_cv', 0):.2f})")
            print(f"{m:9s} {cond:7s} {c['i_hold_nA']:7.3f}nA {c['settle_ms']:7.0f}ms {extra} {f('soma'):>11s} {f('dend'):>11s}")
    print(f"\nsaved {OUT_FILE}\nfigures in {FIG_DIR}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        print("RESULT " + json.dumps(run_one(sys.argv[2]), default=_jsonable))
    else:
        main()
