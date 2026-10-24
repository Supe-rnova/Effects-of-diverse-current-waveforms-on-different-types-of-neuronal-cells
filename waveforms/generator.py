"""
Step 5 - current-injection waveform generator.

Every function returns a `Waveform` (time vector, current vector, metadata),
sampled on the simulation grid t = 0, dt, 2dt, ... (dt = config.settings.DT).

Units: time in ms, current in nA, frequency in Hz.

Edge exactness (pipeline change #4): every onset, offset and pulse edge is
snapped to a whole number of dt steps (an error is raised if a requested time
is not a multiple of dt), and Step 6 plays the vectors WITHOUT interpolation.
So a 0.1 ms pulse is exactly 4 samples with vertical edges, and a biphasic
pulse carries exactly zero net charge.

Families (8 waveforms, 6 families):
  1 step               steps
  2 hyperpolarising    steps          (step with negative amplitude + release)
  3 ramp               ramps
  4 sinusoid           periodic
  5 chirp / ZAP        periodic
  6 pulse train        pulse trains   (monophasic / biphasic)
  7 noise              stochastic     (Gaussian white / Ornstein-Uhlenbeck, frozen-noise trials)
  8 synaptic-like      biological     (alpha / double exponential, given or Poisson event times)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import DT  # noqa: E402


@dataclass
class Waveform:
    t: np.ndarray            # ms
    i: np.ndarray            # nA
    kind: str
    params: dict = field(default_factory=dict)

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0]) if len(self.t) > 1 else DT

    @property
    def duration(self) -> float:
        return len(self.t) * self.dt

    def charge_pC(self, t0: float | None = None, t1: float | None = None) -> float:
        """Injected charge (pC = nA * ms) - exact for the zero-order-hold playback."""
        m = np.ones_like(self.t, bool)
        if t0 is not None:
            m &= self.t >= t0 - 1e-9
        if t1 is not None:
            m &= self.t < t1 - 1e-9
        return float(self.i[m].sum() * self.dt)

    def scaled(self, factor: float) -> "Waveform":
        p = dict(self.params, scale=self.params.get("scale", 1.0) * factor)
        return Waveform(self.t, self.i * factor, self.kind, p)

    def __add__(self, other: "Waveform") -> "Waveform":
        if len(self.t) != len(other.t):
            raise ValueError("waveforms must have the same length to be added")
        return Waveform(self.t, self.i + other.i, f"{self.kind}+{other.kind}",
                        {"a": self.params, "b": other.params})


# ===================================================================== helpers
def _n(time_ms: float, dt: float, what: str = "time") -> int:
    """Number of dt steps in `time_ms`; raises if not an exact multiple of dt."""
    n = time_ms / dt
    if abs(n - round(n)) > 1e-6:
        raise ValueError(f"{what} = {time_ms} ms is not a multiple of dt = {dt} ms")
    return int(round(n))


def _grid(total: float, dt: float):
    n = _n(total, dt, "total duration")
    return np.arange(n) * dt, np.zeros(n)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.PCG64(seed))


# ===================================================================== 1-2 steps
def step(amp: float, onset: float, dur: float, total: float, dt: float = DT) -> Waveform:
    """Rectangular step of `amp` nA from `onset` for `dur` ms."""
    t, i = _grid(total, dt)
    a, b = _n(onset, dt, "onset"), _n(onset + dur, dt, "onset+dur")
    i[a:b] = amp
    return Waveform(t, i, "step", dict(amp=amp, onset=onset, dur=dur, total=total))


def hyperpolarising_step(amp: float, onset: float, dur: float, total: float, dt: float = DT) -> Waveform:
    """Negative step followed by release (rebound window = after onset+dur)."""
    if amp > 0:
        raise ValueError("hyperpolarising step needs amp <= 0")
    w = step(amp, onset, dur, total, dt)
    w.kind = "hyperpolarising_step"
    w.params["release"] = onset + dur
    return w


# ===================================================================== 3 ramp
def ramp(amp_start: float, amp_end: float, onset: float, dur: float, total: float,
         dt: float = DT) -> Waveform:
    """Linear ramp from amp_start to amp_end over `dur` ms, zero elsewhere."""
    t, i = _grid(total, dt)
    a, b = _n(onset, dt, "onset"), _n(onset + dur, dt, "onset+dur")
    i[a:b] = np.linspace(amp_start, amp_end, b - a, endpoint=False)
    return Waveform(t, i, "ramp", dict(amp_start=amp_start, amp_end=amp_end, onset=onset,
                                       dur=dur, total=total, slope_nA_per_s=(amp_end - amp_start) / dur * 1e3))


# ===================================================================== 4-5 periodic
def sinusoid(amp: float, freq: float, onset: float, dur: float, total: float, dc: float = 0.0,
             phase: float = 0.0, dt: float = DT) -> Waveform:
    """dc + amp * sin(2 pi f (t - onset) + phase) during [onset, onset+dur)."""
    t, i = _grid(total, dt)
    a, b = _n(onset, dt, "onset"), _n(onset + dur, dt, "onset+dur")
    tt = t[a:b] - onset
    i[a:b] = dc + amp * np.sin(2 * np.pi * freq * tt / 1000.0 + phase)
    return Waveform(t, i, "sinusoid", dict(amp=amp, freq=freq, dc=dc, phase=phase,
                                           onset=onset, dur=dur, total=total))


def chirp(amp: float, f0: float, f1: float, onset: float, dur: float, total: float,
          dc: float = 0.0, method: str = "linear", dt: float = DT) -> Waveform:
    """ZAP: sine whose frequency sweeps from f0 to f1 Hz (linear or logarithmic)."""
    t, i = _grid(total, dt)
    a, b = _n(onset, dt, "onset"), _n(onset + dur, dt, "onset+dur")
    tt = (t[a:b] - onset) / 1000.0           # s
    T = dur / 1000.0
    if method == "linear":
        ph = 2 * np.pi * (f0 * tt + 0.5 * (f1 - f0) / T * tt ** 2)
    elif method == "log":
        k = (f1 / f0) ** (1 / T)
        ph = 2 * np.pi * f0 * (k ** tt - 1) / np.log(k)
    else:
        raise ValueError("method must be 'linear' or 'log'")
    i[a:b] = dc + amp * np.sin(ph)
    return Waveform(t, i, "chirp", dict(amp=amp, f0=f0, f1=f1, dc=dc, method=method,
                                        onset=onset, dur=dur, total=total))


def chirp_inst_freq(w: Waveform) -> np.ndarray:
    """Instantaneous frequency (Hz) of a chirp at each sample of its active window."""
    p = w.params
    tt = np.arange(_n(p["dur"], w.dt)) * w.dt / 1000.0
    T = p["dur"] / 1000.0
    if p["method"] == "linear":
        return p["f0"] + (p["f1"] - p["f0"]) * tt / T
    return p["f0"] * (p["f1"] / p["f0"]) ** (tt / T)


# ===================================================================== 6 pulse trains
def pulse_train(amp: float, width: float, freq: float, onset: float, dur: float, total: float,
                polarity: str = "mono", interphase: float = 0.0, dt: float = DT) -> Waveform:
    """Train of rectangular pulses at `freq` Hz.

    polarity = 'mono'     : +amp for `width` ms
             = 'biphasic' : +amp for `width`, gap `interphase`, then -amp for `width`
                            (charge-balanced: net charge exactly 0)
    Use a negative amp for cathodic-first / hyperpolarising pulses.
    """
    if polarity not in ("mono", "biphasic"):
        raise ValueError("polarity must be 'mono' or 'biphasic'")
    t, i = _grid(total, dt)
    nw, ng = _n(width, dt, "pulse width"), _n(interphase, dt, "interphase gap")
    if nw < 1:
        raise ValueError("pulse width must be at least one dt")
    period = 1000.0 / freq
    n_period = period / dt
    per_pulse = nw * (2 if polarity == "biphasic" else 1) + (ng if polarity == "biphasic" else 0)
    if per_pulse >= n_period:
        raise ValueError(f"pulse ({per_pulse * dt} ms) does not fit in the period ({period:.3f} ms)")
    a = _n(onset, dt, "onset")
    b = _n(onset + dur, dt, "onset+dur")
    starts = []
    k = 0
    while True:
        s = a + int(round(k * n_period))      # pulse starts rounded to the dt grid
        if s + per_pulse > b:
            break
        i[s:s + nw] = amp
        if polarity == "biphasic":
            i[s + nw + ng:s + 2 * nw + ng] = -amp
        starts.append(s * dt)
        k += 1
    return Waveform(t, i, "pulse_train", dict(amp=amp, width=width, freq=freq, polarity=polarity,
                                              interphase=interphase, onset=onset, dur=dur,
                                              total=total, pulse_times=starts, n_pulses=len(starts)))


# ===================================================================== 7 noise
def white_noise(mean: float, sigma: float, onset: float, dur: float, total: float, seed: int,
                dt: float = DT) -> Waveform:
    """Gaussian white noise, one independent sample per dt (std = sigma)."""
    t, i = _grid(total, dt)
    a, b = _n(onset, dt, "onset"), _n(onset + dur, dt, "onset+dur")
    i[a:b] = mean + sigma * _rng(seed).standard_normal(b - a)
    return Waveform(t, i, "white_noise", dict(mean=mean, sigma=sigma, onset=onset, dur=dur,
                                              total=total, seed=seed))


def ou_noise(mean: float, sigma: float, tau: float, onset: float, dur: float, total: float,
             seed: int, dt: float = DT) -> Waveform:
    """Ornstein-Uhlenbeck (coloured) noise: stationary std = sigma, correlation time = tau ms.
    Exact discretisation: x[n+1] = mean + (x[n]-mean) e^(-dt/tau) + sigma sqrt(1-e^(-2dt/tau)) N(0,1).
    Started from the stationary distribution, so there is no transient."""
    t, i = _grid(total, dt)
    a, b = _n(onset, dt, "onset"), _n(onset + dur, dt, "onset+dur")
    rng = _rng(seed)
    rho = np.exp(-dt / tau)
    q = sigma * np.sqrt(1 - rho ** 2)
    z = rng.standard_normal(b - a)
    x = np.empty(b - a)
    x[0] = sigma * z[0]
    for k in range(1, b - a):
        x[k] = rho * x[k - 1] + q * z[k]
    i[a:b] = mean + x
    return Waveform(t, i, "ou_noise", dict(mean=mean, sigma=sigma, tau=tau, onset=onset, dur=dur,
                                           total=total, seed=seed))


def frozen_noise_trial(mean: float, sigma: float, tau: float, onset: float, dur: float, total: float,
                       signal_seed: int, trial_seed: int, sigma_bg: float, tau_bg: float = 2.0,
                       dt: float = DT) -> Waveform:
    """Mainen & Sejnowski (1995) style trial (pipeline change #2):
    the SAME frozen OU signal (signal_seed) on every trial, plus a small INDEPENDENT
    background noise (trial_seed, std sigma_bg) so that deterministic models can show
    trial-to-trial variability. Reliability = how similar spike trains are across trials."""
    sig = ou_noise(mean, sigma, tau, onset, dur, total, signal_seed, dt)
    bg = ou_noise(0.0, sigma_bg, tau_bg, onset, dur, total, trial_seed, dt) if sigma_bg > 0 else None
    i = sig.i + (bg.i if bg is not None else 0.0)
    return Waveform(sig.t, i, "frozen_noise", dict(mean=mean, sigma=sigma, tau=tau, onset=onset,
                                                   dur=dur, total=total, signal_seed=signal_seed,
                                                   trial_seed=trial_seed, sigma_bg=sigma_bg,
                                                   tau_bg=tau_bg))


# ===================================================================== 8 synaptic-like
def _kernel(t, kind, tau_rise, tau_decay):
    if kind == "alpha":
        return (t / tau_decay) * np.exp(1 - t / tau_decay)          # peak 1 at t = tau
    # double exponential, normalised to peak 1
    if tau_rise >= tau_decay:
        raise ValueError("tau_rise must be < tau_decay")
    tp = tau_rise * tau_decay / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
    norm = np.exp(-tp / tau_decay) - np.exp(-tp / tau_rise)
    return (np.exp(-t / tau_decay) - np.exp(-t / tau_rise)) / norm


def synaptic(peak: float, event_times, total: float, kind: str = "double_exp",
             tau_rise: float = 0.5, tau_decay: float = 5.0, dt: float = DT) -> Waveform:
    """Synaptic-like current: one kernel (peak `peak` nA) per event time, summed linearly.
    kind='alpha' uses tau_decay as the alpha time constant (time to peak)."""
    if kind not in ("alpha", "double_exp"):
        raise ValueError("kind must be 'alpha' or 'double_exp'")
    t, i = _grid(total, dt)
    klen = int(np.ceil(10 * tau_decay / dt))
    k = _kernel(np.arange(klen) * dt, kind, tau_rise, tau_decay)
    ev = np.asarray(sorted(event_times), float)
    for te in ev:
        s = int(round(te / dt))                # events snapped to the dt grid
        if s >= len(t):
            continue
        e = min(len(t), s + klen)
        i[s:e] += peak * k[:e - s]
    return Waveform(t, i, f"synaptic_{kind}", dict(peak=peak, kind=kind, tau_rise=tau_rise,
                                                   tau_decay=tau_decay, total=total,
                                                   event_times=ev.tolist(), n_events=len(ev)))


def poisson_times(rate: float, onset: float, dur: float, seed: int, refractory: float = 0.0) -> np.ndarray:
    """Homogeneous Poisson event times (ms) at `rate` Hz in [onset, onset+dur)."""
    rng = _rng(seed)
    out, t = [], onset
    while True:
        t += rng.exponential(1000.0 / rate) + refractory
        if t >= onset + dur:
            break
        out.append(t)
    return np.array(out)


# ===================================================================== registry
WAVEFORMS = {
    "step": step,
    "hyperpolarising_step": hyperpolarising_step,
    "ramp": ramp,
    "sinusoid": sinusoid,
    "chirp": chirp,
    "pulse_train": pulse_train,
    "ou_noise": ou_noise,
    "white_noise": white_noise,
    "frozen_noise": frozen_noise_trial,
    "synaptic": synaptic,
}


def make(waveform_name: str, /, **params) -> Waveform:
    """Build a waveform by name, e.g. make('sinusoid', amp=0.1, freq=10, onset=100, dur=1000, total=1200).
    (positional-only name, so waveform parameters may themselves be called 'kind')"""
    if waveform_name not in WAVEFORMS:
        raise KeyError(f"unknown waveform '{waveform_name}'. Options: {list(WAVEFORMS)}")
    return WAVEFORMS[waveform_name](**params)
