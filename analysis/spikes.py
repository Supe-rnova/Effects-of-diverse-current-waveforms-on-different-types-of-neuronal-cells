"""
Spike detection (pipeline change #9).

detect_spikes:            somatic Na+ spikes, using a dV/dt criterion plus a peak
                          criterion, so small spikes near depolarisation block are
                          still caught and slow depolarisations are not.
detect_dendritic_ca_spikes: broad dendritic Ca2+ spikes (Purkinje), identified by
                          amplitude AND width, so back-propagating Na+ spikes are
                          not counted.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import SPIKE_DVDT_THRESH, SPIKE_MIN_PEAK, SPIKE_REFRACTORY  # noqa: E402


@dataclass
class Spikes:
    times: np.ndarray        # ms, time of the voltage peak
    onsets: np.ndarray       # ms, time dV/dt first crossed threshold
    thresholds: np.ndarray   # mV, voltage at onset (spike threshold)
    peaks: np.ndarray        # mV

    def __len__(self):
        return len(self.times)

    def rate(self, t0: float, t1: float) -> float:
        """Firing rate (Hz) of spikes whose peak lies in [t0, t1)."""
        n = np.sum((self.times >= t0) & (self.times < t1))
        return 1000.0 * n / (t1 - t0) if t1 > t0 else np.nan

    def in_window(self, t0: float, t1: float) -> np.ndarray:
        return self.times[(self.times >= t0) & (self.times < t1)]


def detect_spikes(t, v, dvdt_thresh: float = SPIKE_DVDT_THRESH,
                  min_peak: float = SPIKE_MIN_PEAK,
                  refractory: float = SPIKE_REFRACTORY) -> Spikes:
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    dvdt = np.gradient(v, t)
    above = dvdt >= dvdt_thresh
    starts = np.where(above[1:] & ~above[:-1])[0] + 1
    times, onsets, thr, peaks = [], [], [], []
    last = -np.inf          # onset of previous accepted spike
    last_peak = -np.inf     # peak time of previous accepted spike
    n = len(v)
    for s in starts:
        if t[s] - last < refractory or t[s] <= last_peak:
            continue        # still inside the previous spike (e.g. a kink on the upstroke)
        # peak = maximum within 5 ms after onset, but stop at the first point where
        # dV/dt turns negative after crossing threshold (= the spike's own peak)
        end = min(n, np.searchsorted(t, t[s] + 5.0))
        if end <= s + 1:
            continue
        fall = np.where(dvdt[s:end] < 0)[0]
        stop = s + (fall[0] + 1 if len(fall) else end - s)
        k = s + int(np.argmax(v[s:stop]))
        if v[k] < min_peak:
            continue
        if t[k] - last_peak < refractory:
            continue        # same peak reached from a second dV/dt crossing
        times.append(t[k]); onsets.append(t[s]); thr.append(v[s]); peaks.append(v[k])
        last, last_peak = t[s], t[k]
    return Spikes(np.array(times), np.array(onsets), np.array(thr), np.array(peaks))


def detect_dendritic_ca_spikes(t, v, v_thresh: float = -10.0, min_width: float = 3.0,
                               refractory: float = 5.0) -> np.ndarray:
    """Return peak times of broad events that exceed v_thresh for >= min_width ms."""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    above = v >= v_thresh
    up = np.where(above[1:] & ~above[:-1])[0] + 1
    down = np.where(~above[1:] & above[:-1])[0] + 1
    out, last = [], -np.inf
    for u in up:
        d = down[down > u]
        if len(d) == 0:
            break
        d = d[0]
        if t[d] - t[u] >= min_width and t[u] - last >= refractory:
            out.append(t[u + int(np.argmax(v[u:d]))])
            last = t[u]
    return np.array(out)
