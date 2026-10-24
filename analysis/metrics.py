"""
Step 8 - response metrics.

Per-run metrics (one function per protocol) take a loaded result
(simulations.injection.load_result) and return a flat dict.
Across-trial metrics (reliability, spike probability/jitter, spike-triggered
average) take lists of results.

Conventions: t = 0 is the waveform start; every stimulus starts at
meta['params']['onset'] (200 ms) except chirp (500 ms) and synaptic events.
Spike times come from analysis.spikes.detect_spikes (dV/dt + peak criterion),
already stored in each result's metadata by the batch runner.
"""
from __future__ import annotations

import numpy as np

from analysis.spikes import detect_dendritic_ca_spikes


# ===================================================================== helpers
def spikes_of(res) -> np.ndarray:
    return np.asarray(res["meta"].get("spike_times_soma", []), float)


def _window(sp, t0, t1):
    return sp[(sp >= t0) & (sp < t1)]


def _isi_stats(sp):
    out = {}
    if len(sp) >= 2:
        isi = np.diff(sp)
        out["isi_mean_ms"] = float(isi.mean())
        out["isi_cv"] = float(isi.std() / isi.mean()) if len(isi) >= 2 else np.nan
        out["first_isi_ms"] = float(isi[0])
        out["last_isi_ms"] = float(isi[-1])
        out["adaptation_index"] = float(isi[-1] / isi[0]) if len(isi) >= 2 else np.nan
        out["initial_rate_Hz"] = float(1000.0 / isi[0])
    return out


def _baseline(res, sp, onset):
    """Pre-stimulus window: mean V and rate (rate meaningful for spontaneous cells)."""
    v, t = res["v"]["soma"], res["t"]
    m = t < onset
    return {"v_base_mV": float(v[m].mean()) if m.any() else np.nan,
            "rate_pre_Hz": 1000.0 * len(_window(sp, 0, onset)) / onset if onset > 0 else np.nan}


def dend_ca_count(res, t0, t1):
    if "dend" not in res["v"]:
        return np.nan
    ca = detect_dendritic_ca_spikes(res["t"], res["v"]["dend"])
    return int(np.sum((ca >= t0) & (ca < t1)))


# ===================================================================== per protocol
def m_step(res):
    p, sp = res["meta"]["params"], spikes_of(res)
    on, dur = p["onset"], p["dur"]
    s = _window(sp, on, on + dur)
    out = {"n_spikes": len(s), "rate_Hz": 1000.0 * len(s) / dur,
           "rate_ss_Hz": 1000.0 * len(_window(sp, on + dur / 2, on + dur)) / (dur / 2),
           "latency_ms": float(s[0] - on) if len(s) else np.nan, **_isi_stats(s), **_baseline(res, sp, on)}
    # depolarisation block: firing stops early although current is still on
    if len(s) >= 2:
        out["depol_block"] = int(s[-1] < on + dur - max(200.0, 3 * out.get("isi_mean_ms", 0)))
    else:
        out["depol_block"] = 0
    thr = res["meta"].get("spike_thresholds_soma", [])
    st = np.asarray(res["meta"].get("spike_times_soma", []))
    if len(thr):
        sel = np.asarray(thr)[(st >= on) & (st < on + dur)]
        out["spike_threshold_mV"] = float(sel[0]) if len(sel) else np.nan
    v = res["v"]["soma"]
    out["v_plateau_mV"] = float(v[(res["t"] >= on + dur - 100) & (res["t"] < on + dur)].mean())
    out["n_dend_ca"] = dend_ca_count(res, on, on + dur)
    return out


def m_hyper(res):
    p, sp = res["meta"]["params"], spikes_of(res)
    on, dur = p["onset"], p["dur"]
    t, v = res["t"], res["v"]["soma"]
    base = v[(t >= on - 50) & (t < on)].mean()
    during = v[(t >= on) & (t < on + dur)]
    peak = during.min()
    ss = v[(t >= on + dur - 100) & (t < on + dur)].mean()
    rel = on + dur
    reb = _window(sp, rel, rel + 300)
    out = {"v_base_mV": float(base), "v_peak_mV": float(peak), "v_ss_mV": float(ss),
           "sag_ratio": float((base - ss) / (base - peak)) if base - peak > 0.1 else np.nan,
           "sag_mV": float(ss - peak),
           "rebound_spikes": len(reb),
           "rebound_latency_ms": float(reb[0] - rel) if len(reb) else np.nan,
           "rebound_max_rate_Hz": float(1000.0 / np.diff(reb).min()) if len(reb) >= 2 else np.nan,
           "rebound_peak_mV": float(v[(t >= rel) & (t < rel + 300)].max()),
           "spikes_during": len(_window(sp, on, rel)),
           "rate_pre_Hz": _baseline(res, sp, on)["rate_pre_Hz"],
           "n_dend_ca_rebound": dend_ca_count(res, rel, rel + 300)}
    return out


def m_ramp(res):
    p, sp = res["meta"]["params"], spikes_of(res)
    on, dur = p["onset"], p["dur"]
    s = _window(sp, on, on + dur)
    out = {"n_spikes": len(s), "slope_nA_per_s": p["slope_nA_per_s"], **_baseline(res, sp, on)}
    if len(s):
        k = int(round(s[0] / res["meta"]["dt"]))
        out["threshold_current_nA"] = float(res["i"][min(k, len(res["i"]) - 1)])
        out["latency_ms"] = float(s[0] - on)
        thr = np.asarray(res["meta"].get("spike_thresholds_soma", []))
        st = np.asarray(res["meta"].get("spike_times_soma", []))
        sel = thr[(st >= on) & (st < on + dur)] if len(thr) else []
        out["spike_threshold_mV"] = float(sel[0]) if len(sel) else np.nan
    else:
        out["threshold_current_nA"] = np.nan
        out["v_max_mV"] = float(res["v"]["soma"][(res["t"] >= on) & (res["t"] < on + dur)].max())
    out.update({k: v for k, v in _isi_stats(s).items() if k in ("isi_cv", "adaptation_index")})
    return out


def vector_strength(spike_times, freq, onset):
    """Phase locking to a sinusoid starting at `onset` (phase 0). Returns VS, Rayleigh p, mean phase."""
    n = len(spike_times)
    if n == 0:
        return np.nan, np.nan, np.nan
    ph = 2 * np.pi * freq * (np.asarray(spike_times) - onset) / 1000.0
    z = np.exp(1j * ph).mean()
    vs = float(np.abs(z))
    R = n * vs
    # Rayleigh test p-value (Zar 1999 approximation)
    p = float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))) if n > 1 else 1.0
    return vs, min(max(p, 0.0), 1.0), float(np.angle(z) % (2 * np.pi))


def m_sine(res):
    p, sp = res["meta"]["params"], spikes_of(res)
    on, dur, f = p["onset"], p["dur"], p["freq"]
    skip = max(1000.0 / f, 100.0)       # ignore the first cycle (onset transient)
    s = _window(sp, on + skip, on + dur)
    cycles = (dur - skip) * f / 1000.0
    vs, pval, ph = vector_strength(s, f, on)
    t, v = res["t"], res["v"]["soma"]
    m = (t >= on + skip) & (t < on + dur)
    out = {"freq_Hz": f, "n_spikes": len(_window(sp, on, on + dur)), "n_spikes_locked_window": len(s),
           "rate_Hz": 1000.0 * len(s) / (dur - skip), "spikes_per_cycle": len(s) / cycles,
           "vector_strength": vs, "rayleigh_p": pval, "mean_phase_rad": ph,
           "v_p2p_mV": float(v[m].max() - v[m].min()), **_baseline(res, sp, on)}
    # fraction of cycles with >= 1 spike (entrainment)
    if len(s):
        cyc = np.floor((s - on) * f / 1000.0).astype(int)
        first = int(np.ceil(skip * f / 1000.0))
        last = int(np.floor(dur * f / 1000.0))
        out["cycles_with_spike"] = float(len(np.unique(cyc[(cyc >= first) & (cyc < last)])) / max(1, last - first))
    else:
        out["cycles_with_spike"] = 0.0
    out["n_dend_ca"] = dend_ca_count(res, on, on + dur)
    return out


def impedance(res, n_bins=60):
    """|Z(f)| (MOhm) from the chirp: FFT(V)/FFT(I), averaged in log-spaced frequency bins."""
    p = res["meta"]["params"]
    on, dur, f0, f1 = p["onset"], p["dur"], p["f0"], p["f1"]
    dt = res["meta"]["dt"]
    t = res["t"]
    site = res["meta"]["site"]
    v = res["v"][site]
    m = (t >= on) & (t < on + dur)
    vv, ii = v[m] - v[m].mean(), res["i"][m] - res["i"][m].mean()
    win = np.hanning(len(vv))
    V, I = np.fft.rfft(vv * win), np.fft.rfft(ii * win)
    fr = np.fft.rfftfreq(len(vv), dt / 1000.0)
    edges = np.logspace(np.log10(f0 * 1.1), np.log10(f1 * 0.9), n_bins + 1)
    fc, Z, ph = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        k = (fr >= a) & (fr < b)
        if k.sum() == 0:
            continue
        # cross-spectral estimate: Z = <V I*> / <|I|^2>  (robust to noise in V)
        z = (V[k] * np.conj(I[k])).sum() / (np.abs(I[k]) ** 2).sum()
        fc.append(np.sqrt(a * b)); Z.append(np.abs(z)); ph.append(np.angle(z))
    return np.array(fc), np.array(Z), np.array(ph)


def m_chirp(res):
    sp = spikes_of(res)
    p = res["meta"]["params"]
    f, Z, ph = impedance(res)
    k = int(np.argmax(Z))
    z0 = Z[0]
    out = {"fR_Hz": float(f[k]), "Q": float(Z[k] / z0), "Z_low_MOhm": float(z0), "Z_max_MOhm": float(Z[k]),
           "n_spikes": len(_window(sp, p["onset"], p["onset"] + p["dur"])),
           "valid_subthreshold": int(len(_window(sp, p["onset"], p["onset"] + p["dur"])) == 0)}
    below = np.where(Z < Z[k] / np.sqrt(2))[0]
    above = below[below > k]
    out["f3dB_Hz"] = float(f[above[0]]) if len(above) else np.nan
    # frequency at which phase crosses zero (inductive -> capacitive), if any
    zc = np.where((ph[:-1] > 0) & (ph[1:] <= 0))[0]
    out["phase_zero_crossing_Hz"] = float(f[zc[0]]) if len(zc) else np.nan
    return out, (f, Z, ph)


def m_pulse(res):
    p, sp = res["meta"]["params"], spikes_of(res)
    pt = np.asarray(p["pulse_times"], float)
    period = 1000.0 / p["freq"]
    w = min(period, 5.0)                      # spike must follow a pulse within this window
    followed = np.array([np.any((sp >= t0) & (sp < t0 + w)) for t0 in pt])
    on, dur = p["onset"], p["dur"]
    half = pt >= on + dur / 2
    s = _window(sp, on, on + dur)
    out = {"freq_Hz": p["freq"], "polarity": p["polarity"], "n_pulses": len(pt),
           "fidelity": float(followed.mean()) if len(pt) else np.nan,
           "fidelity_first10": float(followed[:10].mean()) if len(pt) else np.nan,
           "fidelity_2nd_half": float(followed[half].mean()) if half.any() else np.nan,
           "n_spikes": len(s), "rate_Hz": 1000.0 * len(s) / dur,
           "extra_spikes": int(len(s) - followed.sum()), **_baseline(res, sp, on)}
    post = _window(sp, on + dur, on + dur + 200)
    out["rate_post_Hz"] = 1000.0 * len(post) / 200.0
    return out


def m_noise(res):
    p, sp = res["meta"]["params"], spikes_of(res)
    on, dur = p["onset"], p["dur"]
    s = _window(sp, on + 100, on + dur)        # skip 100 ms onset transient
    out = {"n_spikes": len(s), "rate_Hz": 1000.0 * len(s) / (dur - 100), **_baseline(res, sp, on)}
    out.update({k: v for k, v in _isi_stats(s).items() if k in ("isi_cv", "isi_mean_ms")})
    return out


def m_syn(res):
    """Per-event response for one trial (aggregated across trials by syn_summary)."""
    p = res["meta"]["params"]["components"][0][1]
    ev = np.asarray(p["event_times"], float)
    sp = spikes_of(res)
    lat = []
    for e in ev:
        s = sp[(sp >= e) & (sp < e + 20.0)]
        lat.append(float(s[0] - e) if len(s) else np.nan)
    lat = np.array(lat)
    return {"n_events": len(ev), "n_responded": int(np.sum(~np.isnan(lat))),
            "spike_prob_trial": float(np.mean(~np.isnan(lat))),
            "latencies_ms": lat.tolist(), "n_spikes_total": len(sp)}


PER_RUN = {"step": m_step, "hyper": m_hyper, "ramp": m_ramp, "sine": m_sine, "pulse": m_pulse,
           "noise_var": m_noise, "noise_rel": m_noise, "syn": m_syn}


# ===================================================================== across trials
def schreiber_reliability(spike_trains, t0, t1, sigma=3.0, dt=0.5):
    """Schreiber et al. (2003) correlation-based reliability: mean pairwise cosine
    similarity of Gaussian-filtered spike trains (0 = unrelated, 1 = identical)."""
    t = np.arange(t0, t1, dt)
    vecs = []
    for sp in spike_trains:
        s = np.asarray(sp)
        s = s[(s >= t0) & (s < t1)]
        x = np.zeros_like(t)
        for ts in s:
            x += np.exp(-0.5 * ((t - ts) / sigma) ** 2)
        vecs.append(x)
    norms = [np.linalg.norm(x) for x in vecs]
    vals = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            if norms[i] > 0 and norms[j] > 0:
                vals.append(np.dot(vecs[i], vecs[j]) / (norms[i] * norms[j]))
    return float(np.mean(vals)) if vals else np.nan


def event_precision(spike_trains, t0, t1, bin_ms=1.0, min_frac=0.5, win=5.0):
    """Spike 'events': PSTH peaks reached by >= min_frac of trials. Returns mean jitter
    (SD of spike times within an event, ms) and mean participation."""
    trains = [np.asarray(s)[(np.asarray(s) >= t0) & (np.asarray(s) < t1)] for s in spike_trains]
    n = len(trains)
    if n == 0:
        return np.nan, np.nan, 0
    allsp = np.concatenate(trains) if any(len(x) for x in trains) else np.array([])
    if len(allsp) == 0:
        return np.nan, np.nan, 0
    bins = np.arange(t0, t1 + bin_ms, bin_ms)
    h, _ = np.histogram(allsp, bins)
    hs = np.convolve(h, np.ones(int(win / bin_ms)), "same")
    peaks = [k for k in range(1, len(hs) - 1) if hs[k] >= hs[k - 1] and hs[k] > hs[k + 1] and hs[k] >= min_frac * n]
    jit, part = [], []
    for k in peaks:
        c = bins[k] + bin_ms / 2
        firsts = [tr[np.argmin(np.abs(tr - c))] for tr in trains
                  if len(tr) and np.min(np.abs(tr - c)) <= win]
        if len(firsts) >= 3:
            jit.append(np.std(firsts)); part.append(len(firsts) / n)
    if not jit:
        return np.nan, np.nan, 0
    return float(np.mean(jit)), float(np.mean(part)), len(jit)


def spike_triggered_average(results, window=50.0, skip=100.0):
    """STA of the injected current (nA) over `window` ms before each spike (noise runs)."""
    segs = []
    for res in results:
        p = res["meta"]["params"]
        dt = res["meta"]["dt"]
        n = int(round(window / dt))
        i = res["i"]
        for ts in spikes_of(res):
            if ts < p["onset"] + skip or ts >= p["onset"] + p["dur"]:
                continue
            k = int(round(ts / dt))
            if k - n >= 0:
                segs.append(i[k - n:k + 1])
    if not segs:
        return None, None, 0
    sta = np.mean(segs, axis=0)
    lag = np.arange(-len(sta) + 1, 1) * results[0]["meta"]["dt"]
    return lag, sta, len(segs)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    d = 1 + z ** 2 / n
    c = ph + z ** 2 / (2 * n)
    r = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2))
    return float((c - r) / d), float((c + r) / d)


def mean_ci(x):
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)], float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    m = x.mean()
    if len(x) < 2:
        return float(m), np.nan, np.nan, 1
    from scipy import stats
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(m), float(m - h), float(m + h), len(x)
