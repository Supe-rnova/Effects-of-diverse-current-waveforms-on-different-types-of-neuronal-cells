"""
Step 7 - experiment design.

build_tasks() turns the design below into a list of fully specified runs.
Every amplitude is converted to nA from each cell's own measurements:
    rheobase (Step 4)               -> steps, ramps, sinusoids, noise, synaptic
    single-pulse threshold (Step 7a) -> pulse trains
    input resistance (Step 4)       -> subthreshold chirp (~ +-5 mV)

A task is a dict:
    id          unique name, also the result file path  results/data/runs/<id>.npz
    model, condition ('native'|'held'), block (channel or None), site ('soma'|'dend')
    protocol    e.g. 'step', 'sine', 'noise_rel' ...
    components  list of (waveform kind, params) - summed to give the injected current
    lead_in     ms of unstimulated time before the waveform (random firing phase, STN native)
    group / level / trial   labels used by the analysis
"""
from __future__ import annotations

import hashlib

import numpy as np

# ===================================================================== design
DT_GRID = 0.025
PRE = 200.0                         # ms baseline before every stimulus

STEP_AMPS = [0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]       # x rheobase
HYPER_AMPS = [-0.5, -1.0, -2.0, -3.0]
RAMPS = {"slow": 2000.0, "fast": 500.0}                # 0 -> 3x over this many ms
RAMP_PEAK = 3.0
SINE_FREQS = [1, 2, 4, 7, 10, 20, 50, 100]             # Hz
SINE_AMPS = [1.0, 1.5, 2.0]
SINE_DUR = 2000.0
CHIRP = dict(f0=0.5, f1=50.0, dur=15000.0, method="log", target_mV=5.0, max_x_rheobase=0.5)
PULSE = dict(width=0.5, interphase=0.1, freqs=[5, 20, 60, 130, 200], amps=[1.2, 2.0], dur=1000.0)
NOISE = dict(mean=1.0, sigma=0.4, tau=5.0, dur=2000.0, n_var=20, n_rel=20,
             signal_seed=2000, sigma_bg=0.05, tau_bg=2.0)
SYN = dict(peaks=[1.0, 2.0, 4.0], n_trials=20, n_events=5, interval=200.0,
           tau_rise=0.5, tau_decay=5.0, sigma_bg=0.05)

# reduced set for dendritic injection (Purkinje, CA1)
DEND = dict(step_amps=[0.8, 1.0, 1.5, 2.0, 3.0], hyper_amps=[-1.0, -2.0], sine_amp=1.5,
            pulse_freqs=[130], n_var=10, syn_trials=10)

# in-silico channel blockade (pipeline change #5): channels per model
BLOCKS = {"STN": ["CaT", "HCN"], "Purkinje": ["CaP", "KD"], "CA1": ["HCN", "KM"]}
BLOCK_SET = dict(step_amps=[1.5, 2.0, 3.0], hyper_amps=[-1.0, -2.0], sine_freqs=[4, 10],
                 sine_amp=1.5, pulse_freqs=[20, 130], pulse_amp=2.0, dend_step_amps=[1.5, 2.0])


# ===================================================================== helpers
def _r(x):
    """Round a time to the dt grid (avoids 0.30000000000000004-type errors)."""
    return round(round(x / DT_GRID) * DT_GRID, 6)


def _seed(text: str) -> int:
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)


class _Builder:
    def __init__(self, model, condition, block, site, unit, pulse_thr, rin, spont_isi):
        self.model, self.condition, self.block, self.site = model, condition, block, site
        self.u, self.pthr, self.rin, self.isi = unit, pulse_thr, rin, spont_isi
        self.tasks = []

    def add(self, protocol, label, components, group, level=None, trial=None):
        tid = "/".join([self.model, self.condition, self.block or "ctrl", self.site, protocol, label])
        lead = 0.0
        if self.isi:     # spontaneously firing: random phase relative to the stimulus
            lead = _r(np.random.default_rng(_seed(tid)).uniform(0, self.isi))
        total = components[0][1]["total"]
        self.tasks.append(dict(id=tid, model=self.model, condition=self.condition, block=self.block,
                               site=self.site, protocol=protocol, components=components,
                               lead_in=lead, group=group, level=level, trial=trial,
                               sim_ms=total + lead))

    # ---- protocols ------------------------------------------------------------
    def steps(self, amps):
        for a in amps:
            self.add("step", f"{a:+.2f}x", [("step", dict(amp=a * self.u, onset=PRE, dur=1000.0,
                                                           total=PRE + 1300.0))], "step", a)

    def hyper(self, amps):
        for a in amps:
            self.add("hyper", f"{a:+.2f}x", [("hyperpolarising_step", dict(amp=a * self.u, onset=PRE,
                                                                            dur=800.0, total=PRE + 1500.0))],
                     "hyper", a)

    def ramps(self, which=("slow", "fast")):
        for name in which:
            d = RAMPS[name]
            self.add("ramp", name, [("ramp", dict(amp_start=0.0, amp_end=RAMP_PEAK * self.u, onset=PRE,
                                                  dur=d, total=PRE + d + 200.0))], "ramp", name)

    def sines(self, freqs, amps):
        for f in freqs:
            dur = 3000.0 if f < 2 else SINE_DUR
            for a in amps:
                self.add("sine", f"{f:g}Hz_{a:.2f}x",
                         [("sinusoid", dict(amp=a * self.u, freq=float(f), onset=PRE, dur=dur,
                                            total=PRE + dur + 200.0))], "sine", [f, a])

    def chirp(self, scale=1.0):
        if not self.rin:
            return
        # ~ +-5 mV (mV / MOhm = nA) but never more than 0.5x rheobase, so it stays subthreshold
        amp = scale * min(CHIRP["target_mV"] / self.rin, CHIRP["max_x_rheobase"] * self.u)
        self.add("chirp", "sub", [("chirp", dict(amp=amp, f0=CHIRP["f0"], f1=CHIRP["f1"], onset=500.0,
                                                 dur=CHIRP["dur"], total=500.0 + CHIRP["dur"] + 200.0,
                                                 method=CHIRP["method"]))], "chirp", amp)

    def pulses(self, freqs, amps, polarities=("mono", "biphasic")):
        if not self.pthr:
            return
        for f in freqs:
            for pol in polarities:
                for a in amps:
                    self.add("pulse", f"{f}Hz_{pol}_{a:.1f}xthr",
                             [("pulse_train", dict(amp=a * self.pthr, width=PULSE["width"], freq=float(f),
                                                   onset=PRE, dur=PULSE["dur"], total=PRE + PULSE["dur"] + 200.0,
                                                   polarity=pol, interphase=PULSE["interphase"]))],
                             "pulse", [f, pol, a])

    def noise_var(self, n):
        for k in range(n):
            self.add("noise_var", f"seed{1001 + k}",
                     [("ou_noise", dict(mean=NOISE["mean"] * self.u, sigma=NOISE["sigma"] * self.u,
                                        tau=NOISE["tau"], onset=PRE, dur=NOISE["dur"],
                                        total=PRE + NOISE["dur"] + 200.0, seed=1001 + k))],
                     "noise_var", None, k)

    def noise_rel(self, n):
        for k in range(n):
            self.add("noise_rel", f"trial{k:02d}",
                     [("frozen_noise", dict(mean=NOISE["mean"] * self.u, sigma=NOISE["sigma"] * self.u,
                                            tau=NOISE["tau"], onset=PRE, dur=NOISE["dur"],
                                            total=PRE + NOISE["dur"] + 200.0,
                                            signal_seed=NOISE["signal_seed"], trial_seed=3001 + k,
                                            sigma_bg=NOISE["sigma_bg"] * self.u, tau_bg=NOISE["tau_bg"]))],
                     "noise_rel", None, k)

    def synaptic(self, peaks, n_trials):
        events = [PRE + 100.0 + j * SYN["interval"] for j in range(SYN["n_events"])]
        total = events[-1] + 200.0
        for p in peaks:
            for k in range(n_trials):
                self.add("syn", f"{p:.1f}x_trial{k:02d}",
                         [("synaptic", dict(peak=p * self.u, event_times=events, total=total,
                                            kind="double_exp", tau_rise=SYN["tau_rise"],
                                            tau_decay=SYN["tau_decay"])),
                          ("ou_noise", dict(mean=0.0, sigma=SYN["sigma_bg"] * self.u, tau=2.0, onset=0.0,
                                            dur=total, total=total, seed=4001 + k))],
                         "syn", p, k)


# ===================================================================== task list
def build_tasks(char: dict, pulse_thr: dict, models=None) -> list[dict]:
    tasks = []
    for model, c in char.items():
        if models and model not in models:
            continue
        silent = c["amplitude_condition"]
        u_soma = c["amplitude_unit_nA"]
        rin = c["conditions"][silent].get("Rin_MOhm")
        pthr = (pulse_thr.get(model) or {}).get("threshold_nA", {})

        # 1) full somatic set, silent condition
        b = _Builder(model, silent, None, "soma", u_soma, pthr.get("soma"), rin, None)
        b.steps(STEP_AMPS); b.hyper(HYPER_AMPS); b.ramps(); b.sines(SINE_FREQS, SINE_AMPS)
        b.chirp(); b.pulses(PULSE["freqs"], PULSE["amps"])
        b.noise_var(NOISE["n_var"]); b.noise_rel(NOISE["n_rel"]); b.synaptic(SYN["peaks"], SYN["n_trials"])
        tasks += b.tasks

        # 2) spontaneously active cells: native condition, random phase, no chirp
        if c["spontaneous"]:
            rate = c["conditions"]["native"]["baseline"]["rate_Hz"]
            isi = 1000.0 / rate if rate > 0 else None
            b = _Builder(model, "native", None, "soma", u_soma, pthr.get("soma"), None, isi)
            b.steps(STEP_AMPS); b.hyper(HYPER_AMPS); b.ramps(); b.sines(SINE_FREQS, SINE_AMPS)
            b.pulses(PULSE["freqs"], PULSE["amps"])
            b.noise_var(NOISE["n_var"]); b.noise_rel(NOISE["n_rel"]); b.synaptic(SYN["peaks"], SYN["n_trials"])
            tasks += b.tasks

        # 3) dendritic injection (reduced set)
        u_d = c.get("amplitude_unit_dend_nA")
        if u_d:
            b = _Builder(model, silent, None, "dend", u_d, pthr.get("dend"), rin, None)
            b.steps(DEND["step_amps"]); b.hyper(DEND["hyper_amps"]); b.ramps(("slow",))
            b.sines(SINE_FREQS, [DEND["sine_amp"]]); b.chirp(scale=0.5)
            b.pulses(DEND["pulse_freqs"], PULSE["amps"])
            b.noise_var(DEND["n_var"]); b.synaptic(SYN["peaks"], DEND["syn_trials"])
            tasks += b.tasks

        # 4) channel blockade (somatic key protocols, + dendritic steps)
        for ch in BLOCKS.get(model, []):
            b = _Builder(model, silent, ch, "soma", u_soma, pthr.get("soma"), rin, None)
            b.steps(BLOCK_SET["step_amps"]); b.hyper(BLOCK_SET["hyper_amps"])
            b.sines(BLOCK_SET["sine_freqs"], [BLOCK_SET["sine_amp"]]); b.chirp()
            b.pulses(BLOCK_SET["pulse_freqs"], [BLOCK_SET["pulse_amp"]], polarities=("mono",))
            tasks += b.tasks
            if u_d:
                bd = _Builder(model, silent, ch, "dend", u_d, pthr.get("dend"), rin, None)
                bd.steps(BLOCK_SET["dend_step_amps"])
                tasks += bd.tasks

    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    return tasks
