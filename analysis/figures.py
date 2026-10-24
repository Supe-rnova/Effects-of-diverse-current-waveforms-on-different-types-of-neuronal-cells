"""
Step 9 - publication figures.

    python analysis/figures.py

Reads results/data/analysis/* (Step 8) and the raw runs, writes PNGs (300 dpi) to
results/figures/final/:

  01_traces_<model>.png    voltage with the injected waveform underneath, all 8 waveforms
  02_fI.png                f-I curves + adaptation (soma) and soma vs dendrite
  03_impedance.png         |Z(f)| for every cell, soma vs dendrite, Ih block
  04_sag_rebound.png       sag ratio and rebound spikes vs hyperpolarising amplitude
  05_phase_locking.png     vector strength vs frequency + phase histograms
  06_pulse_fidelity.png    following fidelity vs pulse rate (mono / biphasic)
  07_noise.png             frozen-noise rasters (reliability) + ISI CV + spike-triggered averages
  08_synaptic.png          spike probability and jitter vs synaptic amplitude
  09_blockade.png          control vs in-silico channel block
  10_heatmap.png           model x waveform summary (z-scored per metric, raw values printed)

Colour: categorical slots in fixed order (HH blue, STN orange, Purkinje aqua,
CA1 yellow), always with a marker shape + legend/direct label (two slots are
below 3:1 contrast on white, so identity is never colour-alone). Heatmap uses a
diverging blue-grey-red scale on z-scores.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import DATA_DIR, FIGURES_DIR, LOGS_DIR  # noqa: E402

AN = DATA_DIR / "analysis"
OUT = FIGURES_DIR / "final"
OUT.mkdir(parents=True, exist_ok=True)
LOG = LOGS_DIR / "step9_figures.log"

MODELS = ["HH", "STN", "Purkinje", "CA1"]
COL = {"HH": "#2a78d6", "STN": "#eb6834", "Purkinje": "#1baf7a", "CA1": "#eda100"}
MARK = {"HH": "o", "STN": "s", "Purkinje": "^", "CA1": "D"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"

# which condition supplies each kind of result (STN: suprathreshold from native,
# subthreshold/rebound from held - see analysis/README.md)
SUPRA = {"HH": "native", "STN": "native", "Purkinje": "native", "CA1": "native"}
SUB = {"HH": "native", "STN": "held", "Purkinje": "native", "CA1": "native"}


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 2.0, "lines.markersize": 6, "figure.facecolor": "white",
        "axes.facecolor": "#fcfcfb", "axes.titlecolor": INK, "text.color": INK, "savefig.dpi": 300,
    })


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def key(m, cond, block="ctrl", site="soma"):
    return f"{m}|{cond}|{block}|{site}"


def load_run(task_id):
    from simulations.injection import load_result
    return load_result(DATA_DIR / "runs" / f"{task_id}.npz")


def run_id(m, cond, block, site, proto, label):
    return f"{m}/{cond}/{block}/{site}/{proto}/{label}"


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    log(f"  wrote {name}")


def chirp_valid(S, m, cond, block="ctrl", site="soma"):
    r = S.get(key(m, cond, block, site), {}).get("resonance")
    return bool(r) and bool(r.get("valid_subthreshold"))


def direct_label(ax, x, y, m, dx=0, dy=0):
    ax.annotate(m, (x, y), xytext=(4 + dx, dy), textcoords="offset points", fontsize=8,
                color=INK2, va="center")


# ===================================================================== 01 traces
TRACE_SET = [
    ("Step 1.5x", "step", "+1.50x", SUPRA),
    ("Hyperpolarising -2x", "hyper", "-2.00x", SUB),
    ("Slow ramp 0-3x", "ramp", "slow", SUPRA),
    ("Sinusoid 7 Hz, 1.5x", "sine", "7Hz_1.50x", SUPRA),
    ("Chirp 0.5-50 Hz (subthreshold)", "chirp", "sub", SUB),
    ("Pulses 130 Hz, 1.2x threshold", "pulse", "130Hz_mono_1.2xthr", SUPRA),
    ("OU noise (seed 1001)", "noise_var", "seed1001", SUPRA),
    ("Synaptic 2x (trial 0)", "syn", "2.0x_trial00", SUPRA),
]


def fig_traces(m):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(8, 2, figsize=(11, 13), gridspec_kw={"height_ratios": [3, 1] * 4,
                                                                 "hspace": 0.15, "wspace": 0.18})
    for k, (title, proto, label, which) in enumerate(TRACE_SET):
        r, c = 2 * (k // 2), k % 2
        av, ai = axes[r, c], axes[r + 1, c]
        try:
            res = load_run(run_id(m, which[m], "ctrl", "soma", proto, label))
        except FileNotFoundError:
            av.text(0.5, 0.5, "not run", transform=av.transAxes, ha="center"); continue
        t, v, i = res["t"], res["v"]["soma"], res["i"]
        if proto == "chirp":
            t = t / 1000.0
        av.plot(t, v, color=COL[m], lw=0.6)
        ai.plot(t, i, color=INK2, lw=0.6)
        cond = which[m]
        av.set_title(f"{title}{'  [held]' if cond == 'held' else ''}", loc="left", fontsize=9)
        av.set_ylabel("mV"); ai.set_ylabel("nA")
        av.tick_params(labelbottom=False)
        ai.set_xlabel("s" if proto == "chirp" else "ms")
        ai.grid(False)
    fig.suptitle(f"{m}: responses to the eight waveforms (soma)", fontweight="bold", y=0.995)
    save(fig, f"01_traces_{m}.png")


# ===================================================================== 02 f-I
def fig_fI(S):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for m in MODELS:
        d = S.get(key(m, SUPRA[m]), {}).get("fI")
        if not d:
            continue
        x, y = np.array(d["x_rheobase"]), np.array(d["rate_Hz"])
        ax[0].plot(x, y, color=COL[m], marker=MARK[m], label=f"{m}{' (native)' if m == 'STN' else ''}")
        direct_label(ax[0], x[-1], y[-1], m)
        ad = np.array([np.nan if a is None else a for a in d["adaptation_index"]], float)
        ax[1].plot(x, ad, color=COL[m], marker=MARK[m], label=m)
    ax[0].set(xlabel="step amplitude (x rheobase)", ylabel="mean rate over 1 s (Hz)", title="f-I curves (soma)")
    ax[1].axhline(1, color=AXIS, lw=1)
    ax[1].set(xlabel="step amplitude (x rheobase)", ylabel="last ISI / first ISI", title="Spike-frequency adaptation")
    ax[0].legend(loc="upper left", fontsize=8)
    for m in ("Purkinje", "CA1"):
        for site, ls in (("soma", "-"), ("dend", "--")):
            d = S.get(key(m, "native", "ctrl", site), {}).get("fI")
            if d:
                ax[2].plot(d["x_rheobase"], d["rate_Hz"], color=COL[m], marker=MARK[m], ls=ls,
                           label=f"{m} {site}", mfc="white" if site == "dend" else COL[m])
    ax[2].set(xlabel="amplitude (x rheobase at that site)", ylabel="rate (Hz)", title="Somatic vs dendritic injection")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "02_fI.png")


# ===================================================================== 03 impedance
def fig_impedance(S, IMP):
    import matplotlib.pyplot as plt

    def curve(tid):
        k = tid.replace("/", "__")
        return IMP[k] if k in IMP.files else None

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for m in MODELS:
        z = curve(run_id(m, SUB[m], "ctrl", "soma", "chirp", "sub"))
        if z is None:
            continue
        f, Z = z[0], z[1]
        r = S[key(m, SUB[m])].get("resonance", {})
        lab = (f"{m}: low-pass (no resonance)" if r.get("Q", 1) < 1.05
               else f"{m}: fR {r.get('fR_Hz', np.nan):.1f} Hz, Q {r.get('Q', np.nan):.2f}")
        ax[0].plot(f, Z / Z[0], color=COL[m], label=lab)
        k = int(np.argmax(Z))
        ax[0].plot(f[k], Z[k] / Z[0], marker=MARK[m], color=COL[m], mec="white", mew=1.5, ms=8)
    ax[0].set(xscale="log", xlabel="frequency (Hz)", ylabel="|Z| / |Z(0.5 Hz)|", title="Impedance profile (soma, normalised)")
    ax[0].legend(fontsize=7.5)
    for m in ("Purkinje", "CA1"):
        for site, ls in (("soma", "-"), ("dend", "--")):
            z = curve(run_id(m, "native", "ctrl", site, "chirp", "sub"))
            if z is not None:
                ax[1].plot(z[0], z[1], color=COL[m], ls=ls, label=f"{m} {site}")
    ax[1].set(xscale="log", xlabel="frequency (Hz)", ylabel="|Z| (MOhm)", title="Soma vs dendrite (absolute)")
    ax[1].legend(fontsize=8)
    notes = []
    for m, cond in (("STN", "held"), ("CA1", "native")):
        for blk, ls in (("ctrl", "-"), ("HCN", ":")):
            z = curve(run_id(m, cond, blk, "soma", "chirp", "sub"))
            if z is not None and not chirp_valid(S, m, cond, blk):
                n = S[key(m, cond, blk)]["resonance"].get("n_spikes")
                notes.append(f"{m} Ih-blocked: fired {n:.0f} spikes during the chirp\n(rebound bursts) - impedance not defined")
                continue
            if z is not None:
                ax[2].plot(z[0], z[1] / z[1][0], color=COL[m], ls=ls,
                           label=f"{m} {'control' if blk == 'ctrl' else 'Ih blocked'}")
    ax[2].set(xscale="log", xlabel="frequency (Hz)", ylabel="|Z| / |Z(0.5 Hz)|", title="Resonance depends on Ih")
    ax[2].legend(fontsize=8, loc="upper left")
    for j, n in enumerate(notes):
        ax[2].text(0.02, 0.04 + 0.12 * j, n, transform=ax[2].transAxes, fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, "03_impedance.png")


# ===================================================================== 04 sag / rebound
def fig_sag(S):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for m in MODELS:
        d = S.get(key(m, SUB[m]), {}).get("hyper")
        if not d:
            continue
        x = -np.array(d["level"], float)
        sag = np.array([np.nan if s is None else s for s in d["sag_ratio"]], float)
        lab = f"{m}{' (held)' if SUB[m] == 'held' else ''}"
        ax[0].plot(x, sag, color=COL[m], marker=MARK[m], label=lab)
        ax[1].plot(x, d["rebound_spikes"], color=COL[m], marker=MARK[m], label=lab)
    ax[0].axhline(1, color=AXIS, lw=1)
    ax[0].set(xlabel="hyperpolarising amplitude (x rheobase)", ylabel="sag ratio (1 = no sag)", title="Sag ratio during the step\n(Ih in STN/CA1/Purkinje; HH: relaxation of its own Na/K gates)")
    ax[1].set(xlabel="hyperpolarising amplitude (x rheobase)", ylabel="spikes in 300 ms after release",
              title="Rebound firing after release")
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8, loc="center left")
    fig.tight_layout()
    save(fig, "04_sag_rebound.png")


# ===================================================================== 05 phase locking
def fig_phase(S):
    import matplotlib.pyplot as plt
    from analysis.metrics import spikes_of
    fig = plt.figure(figsize=(13, 4))
    gs = fig.add_gridspec(1, 6, width_ratios=[2.4, 0.3, 1, 1, 1, 1], wspace=0.45)
    FPOL = 20
    ax = fig.add_subplot(gs[0, 0])
    for m in MODELS:
        d = S.get(key(m, SUPRA[m]), {}).get("sine")
        if not d:
            continue
        pts = sorted([(x["freq_Hz"], x["vector_strength"], x["rayleigh_p"]) for x in d if x["amp_x"] == 1.5])
        f = np.array([p[0] for p in pts]); vs = np.array([np.nan if p[1] is None else p[1] for p in pts], float)
        ax.plot(f, vs, color=COL[m], marker=MARK[m], label=m)
        sig = np.array([p[2] is not None and p[2] < 0.05 for p in pts])
        ax.plot(f[~sig & ~np.isnan(vs)], vs[~sig & ~np.isnan(vs)], ls="none", marker=MARK[m], mfc="white", mec=COL[m])
    ax.set(xscale="log", ylim=(0, 1.05), xlabel="sinusoid frequency (Hz)", ylabel="vector strength",
           title="Phase locking at 1.5x rheobase\n(open marker = not significant, Rayleigh p >= 0.05; gap = no spikes)")
    ax.legend(fontsize=8, loc="lower left")
    for j, m in enumerate(MODELS):
        pa = fig.add_subplot(gs[0, 2 + j], projection="polar")
        try:
            res = load_run(run_id(m, SUPRA[m], "ctrl", "soma", "sine", f"{FPOL}Hz_1.50x"))
        except FileNotFoundError:
            continue
        p = res["meta"]["params"]
        sp = spikes_of(res)
        sp = sp[(sp >= p["onset"] + max(1000.0 / FPOL, 100.0)) & (sp < p["onset"] + p["dur"])]
        ph = (2 * np.pi * FPOL * (sp - p["onset"]) / 1000.0) % (2 * np.pi)
        pa.hist(ph, bins=18, range=(0, 2 * np.pi), color=COL[m], edgecolor="white", linewidth=1)
        pa.set_title(f"{m}\n{FPOL} Hz, n={len(sp)}", fontsize=8.5, pad=12)
        pa.set_yticklabels([]); pa.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
        pa.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=6.5)
        pa.tick_params(pad=1)
    save(fig, "05_phase_locking.png")


# ===================================================================== 06 pulses
def fig_pulses(S):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.7), sharey=True)
    for j, (pol, amp) in enumerate((("mono", 1.2), ("biphasic", 1.2), ("mono", 2.0))):
        for m in MODELS:
            d = S.get(key(m, SUPRA[m]), {}).get("pulse")
            if not d:
                continue
            pts = sorted((p["freq_Hz"], p["fidelity"]) for p in d if p["polarity"] == pol and p["amp_x_thr"] == amp)
            ax[j].plot([p[0] for p in pts], [p[1] for p in pts], color=COL[m], marker=MARK[m], label=m)
        ax[j].axvspan(100, 200, color=GRID, alpha=0.5, lw=0)
        ax[j].text(140, 0.03, "DBS range", fontsize=7.5, color=MUTED, ha="center")
        ax[j].set(xscale="log", ylim=(0, 1.05), xlabel="pulse rate (Hz)",
                  title=f"{pol}, {amp}x single-pulse threshold")
    ax[0].set_ylabel("following fidelity (fraction of pulses -> spike)")
    ax[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Spike following of 0.5 ms pulse trains (soma; STN native)", fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "06_pulse_fidelity.png")


# ===================================================================== 07 noise
def fig_noise(S, STA):
    import matplotlib.pyplot as plt
    from analysis.metrics import spikes_of
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(3, 4, height_ratios=[2.2, 0.6, 1.6], hspace=0.35)
    for j, m in enumerate(MODELS):
        ax = fig.add_subplot(gs[0, j])
        cond = SUPRA[m]
        trains = []
        for k in range(20):
            try:
                res = load_run(run_id(m, cond, "ctrl", "soma", "noise_rel", f"trial{k:02d}"))
            except FileNotFoundError:
                continue
            trains.append(spikes_of(res))
            if k == 0:
                axi = fig.add_subplot(gs[1, j])
                axi.plot(res["t"], res["i"], color=INK2, lw=0.4)
                axi.set(xlim=(200, 2200), xlabel="ms", ylabel="nA"); axi.grid(False)
        for k, sp in enumerate(trains):
            ax.vlines(sp, k + 0.6, k + 1.4, color=COL[m], lw=0.8)
        rel = S.get(key(m, cond), {}).get("reliability", {})
        R = rel.get("schreiber_R")
        ax.set(xlim=(200, 2200), ylim=(0.4, len(trains) + 0.6), ylabel="trial",
               title=f"{m}: frozen noise, R = {R:.2f}" if R is not None else f"{m}: no spikes")
        ax.tick_params(labelbottom=False); ax.grid(False)
    # ISI CV with 95% CI
    ax = fig.add_subplot(gs[2, 0:2])
    xs = np.arange(len(MODELS))
    for j, m in enumerate(MODELS):
        v = S.get(key(m, SUPRA[m]), {}).get("variability", {}).get("isi_cv")
        if v and v[0] is not None:
            ax.bar(j, v[0], color=COL[m], width=0.6)
            if v[1] is not None:
                ax.errorbar(j, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], color=INK, capsize=3, lw=1)
            ax.text(j + 0.33, v[0], f"{v[0]:.2f}", ha="left", va="center", fontsize=8, color=INK2)
    ax.set(xticks=xs, xticklabels=MODELS, ylabel="ISI CV (mean, 95% CI)", title="Spike-timing variability (20 noise seeds)")
    ax.grid(axis="x", visible=False)
    # STA
    ax = fig.add_subplot(gs[2, 2:4])
    for m in MODELS:
        k = key(m, SUPRA[m]).replace("|", "__")
        if k in STA.files:
            lag, sta = STA[k]
            base = sta[: int(len(sta) * 0.2)].mean()
            ax.plot(lag, (sta - base) / (sta.max() - base), color=COL[m], label=m)
    ax.set(xlabel="time before spike (ms)", ylabel="STA (normalised)", title="Spike-triggered average of input current")
    ax.legend(fontsize=8)
    save(fig, "07_noise.png")


# ===================================================================== 08 synaptic
def fig_syn(S):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for m in MODELS:
        d = S.get(key(m, SUPRA[m]), {}).get("synaptic")
        if not d:
            continue
        lv = sorted(d, key=float)
        x = [float(l) for l in lv]
        p = [d[l]["spike_prob"] for l in lv]
        lo = [d[l]["ci95"][0] for l in lv]; hi = [d[l]["ci95"][1] for l in lv]
        ax[0].plot(x, p, color=COL[m], marker=MARK[m], label=m)
        ax[0].fill_between(x, lo, hi, color=COL[m], alpha=0.15, lw=0)
        j = [d[l]["jitter_ms"] if d[l]["jitter_ms"] is not None else np.nan for l in lv]
        ax[1].plot(x, j, color=COL[m], marker=MARK[m], label=m)
    ax[0].set(xlabel="synaptic peak (x rheobase)", ylabel="spike probability per event", ylim=(-0.02, 1.05),
              title="Synaptic-like input: reliability (Wilson 95% CI)")
    ax[1].set(xlabel="synaptic peak (x rheobase)", ylabel="latency jitter, SD (ms)", title="Synaptic-like input: precision")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "08_synaptic.png")


# ===================================================================== 09 blockade
def fig_block(S):
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(AN / "metrics_runs.csv")

    def pre_rate(m, cond, blk):
        d = df[(df.model == m) & (df.condition == cond) & (df.block == blk) & (df.site == "soma")
               & (df.protocol == "step")]
        return float(d["rate_pre_Hz"].mean()) if len(d) else np.nan

    def at(s, sec, lvl_key, lvl, val):
        return s[sec][val][s[sec][lvl_key].index(lvl)]

    panels = [
        ("STN (held): rebound spikes after -2x", "STN", "held", ["ctrl", "CaT", "HCN"],
         lambda s, m, c, b: at(s, "hyper", "level", -2.0, "rebound_spikes"), False),
        ("STN (held): resonance Q", "STN", "held", ["ctrl", "CaT", "HCN"],
         lambda s, m, c, b: s["resonance"]["Q"], True),
        ("Purkinje: rate at 2x step", "Purkinje", "native", ["ctrl", "CaP", "KD"],
         lambda s, m, c, b: at(s, "fI", "x_rheobase", 2.0, "rate_Hz"), False),
        ("Purkinje: spontaneous rate (no input)", "Purkinje", "native", ["ctrl", "CaP", "KD"],
         lambda s, m, c, b: pre_rate(m, c, b), False),
        ("CA1: resonance Q", "CA1", "native", ["ctrl", "HCN", "KM"],
         lambda s, m, c, b: s["resonance"]["Q"], True),
        ("CA1: rate at 2x step", "CA1", "native", ["ctrl", "HCN", "KM"],
         lambda s, m, c, b: at(s, "fI", "x_rheobase", 2.0, "rate_Hz"), False),
    ]
    fig, ax = plt.subplots(1, len(panels), figsize=(17, 3.5))
    for a, (title, m, cond, blocks, fn, is_chirp) in zip(ax, panels):
        vals, invalid = [], []
        for b in blocks:
            s = S.get(key(m, cond, b), {})
            bad = is_chirp and not chirp_valid(S, m, cond, b)
            invalid.append(bad)
            try:
                vals.append(np.nan if bad else float(fn(s, m, cond, b)))
            except (KeyError, ValueError, TypeError, IndexError):
                vals.append(np.nan)
        vals = np.array(vals, float)
        cols = [COL[m] if b == "ctrl" else "#ffffff" for b in blocks]
        a.bar(range(len(blocks)), np.nan_to_num(vals), color=cols, edgecolor=COL[m], linewidth=1.5, width=0.6)
        top = np.nanmax(vals) if np.any(~np.isnan(vals)) else 1
        for x, (v, bad) in enumerate(zip(vals, invalid)):
            if bad:
                a.text(x, 0.05 * top, "cell fired\nduring chirp:\nnot defined", ha="center", va="bottom",
                       fontsize=7, color=INK2)
            elif not np.isnan(v):
                a.text(x, v, f"{v:.2f}" if v < 10 else f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
        a.set_xticks(range(len(blocks)))
        a.set_xticklabels(["control" if b == "ctrl" else f"{b}\nblocked" for b in blocks], fontsize=8)
        a.set_title(title, fontsize=9)
        a.grid(axis="x", visible=False)
    fig.suptitle("In-silico channel blockade (filled = control, outlined = channel removed)", fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "09_blockade.png")


# ===================================================================== 10 heatmap
def heatmap_table(S, char):
    rows = {}
    for m in MODELS:
        sp, sb = S.get(key(m, SUPRA[m]), {}), S.get(key(m, SUB[m]), {})
        u = char[m]["amplitude_unit_nA"]

        def get(fn):
            try:
                v = fn()
                return np.nan if v is None else float(v)
            except (KeyError, ValueError, IndexError, TypeError):
                return np.nan
        rows[m] = {
            "Step: rate @2x (Hz)": get(lambda: sp["fI"]["rate_Hz"][sp["fI"]["x_rheobase"].index(2.0)]),
            "Step: adaptation @3x": get(lambda: sp["fI"]["adaptation_index"][sp["fI"]["x_rheobase"].index(3.0)]),
            "Hyper: sag ratio @-2x": get(lambda: sb["hyper"]["sag_ratio"][sb["hyper"]["level"].index(-2.0)]),
            "Hyper: rebound spikes @-2x": get(lambda: sb["hyper"]["rebound_spikes"][sb["hyper"]["level"].index(-2.0)]),
            "Ramp: threshold (x rheobase)": get(lambda: sb["ramp"]["slow"]["threshold_current_nA"] / u),
            "Sine: VS @7 Hz": get(lambda: [x for x in sp["sine"] if x["freq_Hz"] == 7 and x["amp_x"] == 1.5][0]["vector_strength"]),
            "Chirp: fR (Hz)": get(lambda: sb["resonance"]["fR_Hz"] if sb["resonance"]["Q"] >= 1.05 else None),
            "Chirp: Q": get(lambda: sb["resonance"]["Q"]),
            "Pulse: fidelity 130 Hz": get(lambda: [p for p in sp["pulse"] if p["freq_Hz"] == 130 and p["polarity"] == "mono" and p["amp_x_thr"] == 1.2][0]["fidelity"]),
            "Noise: ISI CV": get(lambda: sp["variability"]["isi_cv"][0]),
            "Noise: reliability R": get(lambda: sp["reliability"]["schreiber_R"]),
            "Synaptic: P(spike) @2x": get(lambda: sp["synaptic"]["2.0"]["spike_prob"]),
        }
    return rows


def fig_heatmap(S, char):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    rows = heatmap_table(S, char)
    metrics = list(next(iter(rows.values())))
    raw = np.array([[rows[m][k] for k in metrics] for m in MODELS], float)
    mu, sd = np.nanmean(raw, 0), np.nanstd(raw, 0)
    sd[sd == 0] = 1
    z = (raw - mu) / sd
    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
    fig, ax = plt.subplots(figsize=(13, 3.6))
    zz = np.ma.masked_invalid(z)
    im = ax.imshow(zz, cmap=cmap, vmin=-1.6, vmax=1.6, aspect="auto")
    ax.set_facecolor("#ffffff")
    for i in range(len(MODELS)):
        for j in range(len(metrics)):
            v = raw[i, j]
            txt = "n/a" if np.isnan(v) else (f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}")
            if np.isnan(v) and metrics[j].startswith("Chirp: fR"):
                txt = "low-pass"
            elif np.isnan(v) and metrics[j].startswith("Sine"):
                txt = "no spikes"
            elif np.isnan(v) and metrics[j].startswith("Ramp"):
                txt = "no spike"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color=INK if np.isnan(z[i, j]) or abs(z[i, j]) < 1.1 else "#ffffff")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([f"{m}" + (" (native / held*)" if m == "STN" else "") for m in MODELS])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(MODELS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("z-score across cells", color=INK2)
    ax.set_title("Model x waveform summary (cell value printed; colour = z-score within each metric)\n"
                 "*STN: hyper, ramp and chirp columns from the held condition (silent at -70 mV), all others native", fontsize=9, loc="left")
    save(fig, "10_heatmap.png")
    import csv
    with open(AN / "heatmap_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + metrics)
        for m in MODELS:
            w.writerow([m] + [rows[m][k] for k in metrics])
    log("  wrote analysis/heatmap_table.csv")


def main():
    LOG.write_text("")
    style()
    S = json.loads((AN / "summary.json").read_text())
    char = json.loads((DATA_DIR / "characterisation.json").read_text())
    IMP = np.load(AN / "impedance.npz")
    STA = np.load(AN / "sta.npz")
    jobs = [(f"traces {m}", lambda m=m: fig_traces(m)) for m in MODELS] + [
        ("f-I", lambda: fig_fI(S)), ("impedance", lambda: fig_impedance(S, IMP)),
        ("sag/rebound", lambda: fig_sag(S)), ("phase locking", lambda: fig_phase(S)),
        ("pulses", lambda: fig_pulses(S)), ("noise", lambda: fig_noise(S, STA)),
        ("synaptic", lambda: fig_syn(S)), ("blockade", lambda: fig_block(S)),
        ("heatmap", lambda: fig_heatmap(S, char))]
    errors = 0
    for name, fn in jobs:
        try:
            fn()
        except Exception:
            errors += 1
            log(f"  ERROR in {name}:\n{traceback.format_exc()}")
    log(f"figures in {OUT}")
    log("FIGURES COMPLETE" if errors == 0 else f"{errors} FIGURES FAILED - see {LOG}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()
