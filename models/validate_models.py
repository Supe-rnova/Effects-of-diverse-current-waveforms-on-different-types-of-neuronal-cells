"""
Step 3 - validate every model against published behaviour + dt-convergence check.

Run from the project root:
    python models/validate_models.py            # all four models, in parallel processes
    python models/validate_models.py --one CA1  # a single model (debugging)

Outputs
    results/figures/validation/<model>.png   figure per model
    results/logs/validation.json             all numbers + PASS/FAIL per criterion

Each check compares a model output with the value reported for that model or
cell type. "Causal" checks block the channel that the paper says is responsible
and confirm the behaviour disappears (pipeline change #5, used here as validation).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import DT, DT_CHECK, FIGURES_DIR, LOGS_DIR, MODEL_NAMES  # noqa: E402

VAL_FIG_DIR = FIGURES_DIR / "validation"
VAL_FIG_DIR.mkdir(parents=True, exist_ok=True)


# ===================================================================== helpers
def simulate(cell, stims, T, dt=DT, record=("soma",)):
    """stims: list of (site, delay, dur, amp_nA). Returns t, {site: v}."""
    from neuron import h
    clamps = []
    for site, delay, dur, amp in stims:
        ic = h.IClamp(cell.sites[site])
        ic.delay, ic.dur, ic.amp = delay, dur, amp
        clamps.append(ic)
    vecs = {s: h.Vector().record(cell.sites[s]._ref_v) for s in record}
    tv = h.Vector().record(h._ref_t)
    h.dt = dt
    h.celsius = cell.celsius
    h.finitialize(cell.v_init)
    h.continuerun(T)
    out = np.array(tv), {s: np.array(v) for s, v in vecs.items()}
    del clamps
    return out


def zap_impedance(cell, site="soma", hold=0.0, amp=0.02, f0=0.5, f1=25.0,
                  dur=12000.0, settle=1500.0, dt=0.1):
    """Time-domain chirp -> |Z(f)| (MOhm). dt=0.1 ms is adequate for subthreshold runs."""
    from neuron import h
    from scipy.ndimage import uniform_filter1d
    t = np.arange(0, settle + dur, dt)
    tt = np.clip(t - settle, 0, None)
    k = (f1 - f0) / dur
    i = hold + np.where(t > settle, amp * np.sin(2 * np.pi * (f0 * tt + 0.5 * k * tt ** 2) / 1000.0), 0.0)
    ic = h.IClamp(cell.sites[site]); ic.delay = 0; ic.dur = 1e9
    tvec, ivec = h.Vector(t), h.Vector(i)
    ivec.play(ic._ref_amp, tvec, True)
    v = h.Vector().record(cell.sites[site]._ref_v)
    h.dt = dt; h.celsius = cell.celsius
    h.finitialize(cell.v_init); h.continuerun(settle + dur)
    v = np.array(v)[:len(t)]
    ivec.play_remove()
    m = t > settle
    Z = np.abs(np.fft.rfft(v[m] - v[m].mean()) / np.fft.rfft(i[m] - i[m].mean()))
    fr = np.fft.rfftfreq(m.sum(), dt / 1000.0)
    band = (fr > f0 + 0.1) & (fr < f1 - 1)
    Zs = uniform_filter1d(Z[band], 15)
    fb = fr[band]
    j = int(np.argmax(Zs))
    return {"f": fb, "Z": Zs, "fR": float(fb[j]), "Q": float(Zs[j] / Zs[:3].mean()),
            "Vm": float(v[m].mean()), "Vmax": float(v[m].max())}


def check(results, name, value, lo, hi, ref):
    ok = bool(lo <= value <= hi)
    results["checks"].append({"name": name, "value": round(float(value), 3),
                              "accept": [lo, hi], "pass": ok, "reference": ref})
    return ok


def dt_convergence(cell, stims, T, results):
    """Same protocol at DT and DT/2; spike counts must match, timing shift small."""
    from analysis.spikes import detect_spikes
    t1, v1 = simulate(cell, stims, T, dt=DT)
    t2, v2 = simulate(cell, stims, T, dt=DT_CHECK)
    s1, s2 = detect_spikes(t1, v1["soma"]), detect_spikes(t2, v2["soma"])
    n = min(len(s1), len(s2))
    first = abs(s1.times[0] - s2.times[0]) if n else np.nan
    mean_shift = float(np.mean(np.abs(s1.times[:n] - s2.times[:n]))) if n else np.nan
    results["dt_check"] = {"dt": DT, "dt_half": DT_CHECK, "n_spikes": [len(s1), len(s2)],
                           "first_spike_shift_ms": first, "mean_shift_ms": mean_shift}
    check(results, "dt: spike-count difference (dt vs dt/2)", abs(len(s1) - len(s2)), 0, 1,
          "numerical convergence")
    check(results, "dt: first-spike shift (ms)", first, 0, 0.1, "numerical convergence")
    return (t1, v1["soma"]), (t2, v2["soma"])


# ===================================================================== per model
def validate_hh(cell, res, plt):
    from analysis.spikes import detect_spikes
    area_cm2 = np.pi * cell.sites["soma"].sec.diam * cell.sites["soma"].sec.L * 1e-8
    nA_per_uAcm2 = area_cm2 * 1e3
    # f-I curve, 500 ms steps
    J = np.arange(0, 40.01, 0.5)                # uA/cm2
    rates = []
    for j in J:
        t, v = simulate(cell, [("soma", 50, 500, j * nA_per_uAcm2)], 600)
        s = detect_spikes(t, v["soma"])
        rates.append(s.rate(300, 550))          # steady-state rate
    rates = np.array(rates)
    onset = J[np.argmax(rates > 0)] if np.any(rates > 0) else np.nan
    fmin = rates[rates > 0][0] if np.any(rates > 0) else 0
    t, v = simulate(cell, [("soma", 50, 500, 10 * nA_per_uAcm2)], 600)
    s = detect_spikes(t, v["soma"])
    res["values"] = {"rest_mV": float(v["soma"][t < 50].mean()), "fI_J_uAcm2": J.tolist(),
                     "fI_rate_Hz": rates.tolist()}
    check(res, "resting potential (mV)", v["soma"][t < 50].mean(), -66, -64, "HH 1952: -65 mV")
    check(res, "AP peak (mV)", s.peaks.max(), 30, 50, "HH 1952: overshoot ~ +40 mV")
    check(res, "repetitive-firing onset (uA/cm2)", onset, 5.5, 7.5,
          "Rinzel & Miller 1980: periodic firing from ~6.3 uA/cm2")
    check(res, "minimum steady firing rate at onset (Hz)", fmin, 40, 90,
          "HH is class-2: abrupt onset at ~50-70 Hz (6.3 degC)")
    (ta, va), (tb, vb) = dt_convergence(cell, [("soma", 50, 300, 10 * nA_per_uAcm2)], 400, res)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].plot(t, v["soma"], lw=.8); ax[0].set(xlabel="ms", ylabel="mV", title="10 uA/cm2 step")
    ax[1].plot(J, rates, "o-", ms=3); ax[1].set(xlabel="J (uA/cm2)", ylabel="rate (Hz)",
                                                  title="f-I: abrupt (class-2) onset")
    return fig


def validate_stn(cell, res, plt):
    from analysis.spikes import detect_spikes
    # spontaneous firing at 37 degC
    t, v = simulate(cell, [], 3000)
    s = detect_spikes(t, v["soma"])
    isi = np.diff(s.in_window(1000, 3000))
    rate, cv = 1000 / isi.mean(), isi.std() / isi.mean()
    check(res, "spontaneous rate at 37 degC (Hz)", rate, 5, 20,
          "Gillies & Willshaw 2006 / Hallworth 2003: regular pacemaking ~10 Hz")
    check(res, "spontaneous ISI CV", cv, 0, 0.2, "regular pacemaker")
    # rebound burst: -0.25 nA for 500 ms at 35 degC (sample.hoc protocol, Bevan 2002)
    cell.celsius = 35.0
    t2, v2 = simulate(cell, [("soma", 1000, 500, -0.25)], 2500)
    s2 = detect_spikes(t2, v2["soma"])
    burst = s2.in_window(1500, 1600)
    intra = 1000 / np.diff(burst).mean() if len(burst) > 1 else 0
    check(res, "rebound spikes in 100 ms after release", len(burst), 4, 40,
          "Gillies & Willshaw 2006 Fig. (Bevan 2002): rebound burst")
    check(res, "rebound intra-burst rate (Hz)", intra, 100, 400, "high-frequency rebound burst")
    sag_min = v2["soma"][(t2 > 1000) & (t2 < 1500)].min()
    # causal: block T-type Ca -> rebound burst should disappear
    cell.block("CaT", 0.0)
    t3, v3 = simulate(cell, [("soma", 1000, 500, -0.25)], 2500)
    cell.restore_channels()
    s3 = detect_spikes(t3, v3["soma"])
    n_blk = len(s3.in_window(1500, 1600))
    check(res, "rebound spikes with CaT blocked", n_blk, 0, 3,
          "Gillies & Willshaw 2006: rebound needs T-type Ca2+")
    cell.celsius = 37.0
    res["values"] = {"spont_rate_Hz": rate, "spont_cv": cv, "rebound_n": len(burst),
                     "rebound_rate_Hz": intra, "hyperpol_min_mV": float(sag_min),
                     "rebound_n_CaT_blocked": n_blk}
    dt_convergence(cell, [("soma", 500, 300, 0.05)], 1000, res)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    ax[0].plot(t, v["soma"], lw=.6); ax[0].set(xlim=(1000, 2000), title="spontaneous, 37 degC", ylabel="mV")
    ax[1].plot(t2, v2["soma"], lw=.6); ax[1].set(xlim=(800, 2000), title="rebound (-0.25 nA), 35 degC")
    ax[2].plot(t3, v3["soma"], lw=.6, color="C3"); ax[2].set(xlim=(800, 2000), title="rebound, CaT blocked")
    for a in ax: a.set_xlabel("ms")
    return fig


def validate_purkinje(cell, res, plt):
    from analysis.spikes import detect_dendritic_ca_spikes, detect_spikes
    rec = ("soma", "dend")
    t, v = simulate(cell, [], 500, record=rec)
    rest = v["soma"][t > 400].mean()
    check(res, "resting potential (mV)", rest, -75, -60, "Miyasho 2001: silent, ~ -68 mV")
    # Miyasho Fig. 2 protocol: +2 nA somatic step for 400 ms
    t, v = simulate(cell, [("soma", 520, 400, 2.0)], 1000, record=rec)
    s = detect_spikes(t, v["soma"])
    ss_rate = s.rate(540, 740)
    ca = detect_dendritic_ca_spikes(t, v["dend"])
    check(res, "somatic simple-spike rate, first 200 ms (Hz)", ss_rate, 60, 200,
          "Miyasho 2001 Fig. 2: fast Na+ spiking")
    check(res, "dendritic Ca2+ spikes during 2 nA step", len(ca), 1, 30,
          "Miyasho 2001 Fig. 2: dendritic Ca2+ spikes")
    # causal: P-type Ca block abolishes dendritic Ca spikes
    cell.block("CaP", 0.0)
    tb, vb = simulate(cell, [("soma", 520, 400, 2.0)], 1000, record=rec)
    cell.restore_channels()
    ca_b = detect_dendritic_ca_spikes(tb, vb["dend"])
    check(res, "dendritic Ca2+ spikes with CaP blocked", len(ca_b), 0, 0,
          "Ca2+ spikes are P-type mediated (Llinas & Sugimori 1980)")
    res["values"] = {"rest_mV": float(rest), "simple_spike_rate_Hz": ss_rate,
                     "n_dend_ca_spikes": len(ca), "n_dend_ca_spikes_CaP_blocked": len(ca_b)}
    dt_convergence(cell, [("soma", 100, 200, 1.0)], 300, res)

    fig, ax = plt.subplots(2, 2, figsize=(13, 5.5), sharex=True)
    ax[0, 0].plot(t, v["soma"], lw=.5); ax[0, 0].set(title="soma, +2 nA", ylabel="mV")
    ax[1, 0].plot(t, v["dend"], lw=.6, color="C1"); ax[1, 0].plot(ca, np.full(len(ca), 40), "kv", ms=4)
    ax[1, 0].set(title="dendrite (100 um): Ca2+ spikes marked", ylabel="mV", xlabel="ms")
    ax[0, 1].plot(tb, vb["soma"], lw=.5, color="C3"); ax[0, 1].set(title="soma, CaP blocked")
    ax[1, 1].plot(tb, vb["dend"], lw=.6, color="C3"); ax[1, 1].set(title="dendrite, CaP blocked", xlabel="ms")
    for a in ax.flat: a.set_xlim(500, 950)
    return fig


def validate_ca1(cell, res, plt):
    from analysis.spikes import detect_spikes
    # input resistance & sag from a -0.1 nA, 800 ms step
    t, v = simulate(cell, [("soma", 200, 800, -0.1)], 1200)
    vs = v["soma"]
    base = vs[(t > 150) & (t < 200)].mean()
    peak = vs[(t > 200) & (t < 400)].min()
    ss = vs[(t > 900) & (t < 1000)].mean()
    rin = (base - ss) / 0.1
    sag = (base - ss) / (base - peak)          # 1 = no sag
    check(res, "input resistance (MOhm)", rin, 20, 100,
          "CA1 pyramidal at 34 degC: ~25-80 MOhm (Migliore 2018 data set)")
    check(res, "sag ratio (steady/peak deflection)", sag, 0.6, 0.95, "Ih-mediated sag in CA1")
    # resonance via time-domain ZAP, with and without Ih
    z = zap_impedance(cell, "soma")
    zd = zap_impedance(cell, "dend")
    cell.block("HCN", 0.0)
    z0 = zap_impedance(cell, "soma")
    cell.restore_channels()
    check(res, "somatic resonance frequency (Hz)", z["fR"], 3, 12,
          "CA1 theta resonance (Hu et al. 2002; Narayanan & Johnston 2007)")
    check(res, "somatic resonance strength Q", z["Q"], 1.05, 2.0, "CA1: Q ~ 1.1-1.4")
    check(res, "dendritic fR > somatic fR", float(zd["fR"] > z["fR"]), 1, 1,
          "Narayanan & Johnston 2007: fR increases with distance")
    check(res, "Q drop when HCN blocked", z["Q"] - z0["Q"], 0.1, 2.0,
          "H-resonance abolished by ZD7288 (Hu et al. 2002); small M-resonance may remain")
    # firing: 1 s step of 0.6 nA -> regular firing with spike-frequency adaptation
    t2, v2 = simulate(cell, [("soma", 200, 1000, 0.6)], 1300)
    s = detect_spikes(t2, v2["soma"])
    isi = np.diff(s.in_window(200, 1200))
    adapt = isi[-1] / isi[0] if len(isi) > 2 else np.nan
    check(res, "firing rate, 0.6 nA / 1 s step (Hz)", s.rate(200, 1200), 5, 40,
          "CA1 pyramidal: low-rate regular firing")
    check(res, "adaptation index (last ISI / first ISI)", adapt, 1.5, 10,
          "CA1 pyramidal: strong spike-frequency adaptation (Migliore 2018: cAC type)")
    res["values"] = {"Rin_MOhm": rin, "sag_ratio": sag, "fR_soma": z["fR"], "Q_soma": z["Q"],
                     "fR_dend": zd["fR"], "Q_dend": zd["Q"], "fR_soma_HCN_blocked": z0["fR"],
                     "Q_soma_HCN_blocked": z0["Q"], "rate_0.6nA_Hz": s.rate(200, 1200),
                     "adaptation_index": adapt}
    dt_convergence(cell, [("soma", 100, 300, 0.8)], 450, res)

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ax[0].plot(t, vs, lw=.8); ax[0].set(title="-0.1 nA step: sag", xlabel="ms", ylabel="mV")
    ax[1].plot(z["f"], z["Z"], label=f"soma (fR={z['fR']:.1f} Hz)")
    ax[1].plot(zd["f"], zd["Z"], label=f"dend 300 um (fR={zd['fR']:.1f} Hz)")
    ax[1].plot(z0["f"], z0["Z"], "--", label=f"soma, HCN blocked (fR={z0['fR']:.1f} Hz)")
    ax[1].set(xscale="log", xlabel="Hz", ylabel="|Z| (MOhm)", title="ZAP impedance"); ax[1].legend(fontsize=8)
    ax[2].plot(t2, v2["soma"], lw=.6); ax[2].set(title="+0.6 nA step: adaptation", xlabel="ms")
    return fig


VALIDATORS = {"HH": validate_hh, "STN": validate_stn, "Purkinje": validate_purkinje, "CA1": validate_ca1}


def run_one(name: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import neuron
    from models.cells import build_cell

    cell = build_cell(name)
    res = {"model": name, "celsius": cell.celsius, "neuron_version": neuron.__version__,
           "reference": cell.reference, "modeldb": cell.modeldb, "checks": []}
    t0 = time.perf_counter()
    fig = VALIDATORS[name](cell, res, plt)
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    res["pass"] = all(c["pass"] for c in res["checks"])
    fig.suptitle(f"{name} validation - {'PASS' if res['pass'] else 'FAIL'}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(VAL_FIG_DIR / f"{name}.png", dpi=130)
    return res


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main():
    procs = {n: subprocess.Popen([sys.executable, __file__, "--one", n], cwd=ROOT,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for n in MODEL_NAMES}
    print(f"validating {len(procs)} models in parallel (Purkinje and CA1 take a few minutes) ...")
    allres = []
    for name, p in procs.items():
        out, err = p.communicate()
        line = [ln for ln in out.splitlines() if ln.startswith("RESULT ")]
        if p.returncode != 0 or not line:
            print(f"\n[FAIL] {name} crashed:\n{err[-3000:]}")
            continue
        r = json.loads(line[-1][7:])
        allres.append(r)
        print(f"\n=== {name} ({r['celsius']} degC, {r['wall_s']} s) : {'PASS' if r['pass'] else 'FAIL'}")
        for c in r["checks"]:
            print(f"  [{'OK' if c['pass'] else '!!'}] {c['name']}: {c['value']}  "
                  f"(accept {c['accept'][0]}..{c['accept'][1]}; {c['reference']})")
    (LOGS_DIR / "validation.json").write_text(json.dumps(allres, indent=2, default=_jsonable))
    print(f"\nfigures: {VAL_FIG_DIR}\nnumbers: {LOGS_DIR / 'validation.json'}")
    ok = len(allres) == len(MODEL_NAMES) and all(r["pass"] for r in allres)
    print("ALL MODELS VALIDATED" if ok else "SOME CHECKS FAILED - do not continue to Step 4 yet")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        print("RESULT " + json.dumps(run_one(sys.argv[2]), default=_jsonable))
    else:
        main()
