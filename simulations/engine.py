"""
Simulation engine (Steps 4 and 6).

Settle a cell ONCE per configuration, save the state with h.SaveState, then
restore it before every run (pipeline change #7) - identical initial conditions
and no settling time in each run. A "configuration" is (holding current,
channel block); several configurations can be cached at the same time.

    eng = Engine(cell, sites=("soma", "dend"), record=("soma", "dend"))
    eng.settle(cell.settle_ms)                       # default configuration
    t, rec = eng.run_step("soma", amp=0.2, dur=1000)
    t, rec, i = eng.run_waveform("soma", waveform)   # Step 6: any Waveform

Times returned are relative to the restore point (t = 0 = start of the waveform).
Waveforms are played WITHOUT interpolation (zero-order hold on the dt grid), so
rectangular pulses keep vertical edges and exact charge (pipeline change #4).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import DT  # noqa: E402


class Engine:
    def __init__(self, cell, sites=("soma",), record=("soma",), i_hold: float = 0.0, dt: float = DT):
        from neuron import h
        self.h = h
        self.cell = cell
        self.dt = dt
        # Every point process must exist BEFORE SaveState.save(), otherwise restore()
        # fails. So the holding clamp and one (silent) test clamp per site are made now.
        self.hold = h.IClamp(cell.sites["soma"])
        self.hold.delay, self.hold.dur, self.hold.amp = 0.0, 1e9, i_hold
        self.stim = {}
        for s in sites:
            ic = h.IClamp(cell.sites[s])
            ic.delay, ic.dur, ic.amp = 0.0, 0.0, 0.0
            self.stim[s] = ic
        self.rec = {s: h.Vector() for s in record}
        for s, v in self.rec.items():
            v.record(cell.sites[s]._ref_v)
        self.t_vec = h.Vector().record(h._ref_t)
        self.i_vec = {s: h.Vector().record(ic._ref_i) for s, ic in self.stim.items()}
        self.state = None
        self.t0 = 0.0
        self._cache = {}          # key -> (SaveState, t0, i_hold)
        self._play = None         # keeps played vectors alive during a run

    # ------------------------------------------------------------------ settling
    def settle(self, settle_ms: float, key=None):
        """Run from finitialize to steady state and save it (optionally under `key`)."""
        h = self.h
        h.dt = self.dt
        h.celsius = self.cell.celsius
        self._silence()
        h.finitialize(self.cell.v_init)
        h.continuerun(settle_ms)
        self.state = h.SaveState()
        self.state.save()
        self.t0 = h.t
        if key is not None:
            self._cache[key] = (self.state, self.t0, self.hold.amp)
        return np.array(self.t_vec), {s: np.array(v) for s, v in self.rec.items()}

    def use(self, key) -> bool:
        """Switch to a cached configuration. Returns False if it was never settled.
        NOTE: channel-block parameters are NOT part of SaveState - the caller must
        apply the matching block before calling use()."""
        if key not in self._cache:
            return False
        self.state, self.t0, amp = self._cache[key]
        self.hold.amp = amp
        return True

    def set_hold(self, amp: float):
        """Change the holding current. Requires settle() (or use()) afterwards."""
        self.hold.amp = amp
        self.state = None

    def _silence(self):
        for ic in self.stim.values():
            ic.amp, ic.dur, ic.delay = 0.0, 0.0, 0.0
        if self._play is not None:
            self._play[1].play_remove()
            self._play = None

    def _restore(self, keep_play: bool = False):
        """Return to the saved steady state.

        NEURON arms Vector.play events only inside finitialize(), so we call
        finitialize() first and then overwrite every state variable (and t) with
        the saved ones via restore(1) - the '1' keeps the freshly queued play events.
        Verified: identical voltage (0.0 mV difference) and exact injected charge
        compared with an uninterrupted run.
        """
        if self.state is None:
            raise RuntimeError("call settle() (or use()) first")
        h = self.h
        h.dt = self.dt
        h.celsius = self.cell.celsius
        if not keep_play:
            self._silence()
        h.finitialize(self.cell.v_init)
        self.state.restore(1)
        h.frecord_init()             # restart the recording vectors at the restore point

    def _collect(self, t_start):
        t = np.array(self.t_vec) - t_start
        return t, {s: np.array(v) for s, v in self.rec.items()}

    # ------------------------------------------------------------------ runs
    def run_baseline(self, dur: float):
        """No stimulus, continue from the settled state."""
        self._restore()
        self.h.continuerun(self.t0 + dur)
        return self._collect(self.t0)

    def run_step(self, site: str, amp: float, dur: float, pre: float = 100.0, post: float = 100.0):
        """Rectangular current step at `site` (nA), starting `pre` ms after the restore point."""
        self._restore()
        ic = self.stim[site]
        ic.delay, ic.dur, ic.amp = self.t0 + pre, dur, amp
        self.h.continuerun(self.t0 + pre + dur + post)
        out = self._collect(self.t0)
        self._silence()
        return out

    def run_waveform(self, site: str, waveform, lead_in: float = 0.0):
        """Play `waveform` (generator.Waveform, sampled at this dt) into `site`.

        lead_in : extra unstimulated time (ms) inserted after the restore point and
                  before the waveform starts. Used to randomise the firing phase of
                  spontaneously active cells across trials (pipeline change #7).
        Returns t (ms, 0 = waveform start), {site: v}, injected current at `site`.
        """
        if abs(waveform.dt - self.dt) > 1e-9:
            raise ValueError(f"waveform dt {waveform.dt} != simulation dt {self.dt}")
        lead_in = round(lead_in / self.dt) * self.dt     # keep the waveform on the dt grid
        h = self.h
        self._silence()
        t_start = self.t0 + lead_in
        ic = self.stim[site]
        ic.delay, ic.dur, ic.amp = 0.0, 1e9, 0.0
        tv = h.Vector(np.asarray(waveform.t) + t_start)
        iv = h.Vector(np.asarray(waveform.i))
        iv.play(ic._ref_amp, tv, False)                   # False = no interpolation
        self._play = (tv, iv)
        self._restore(keep_play=True)                     # play must be set up BEFORE restore
        h.continuerun(t_start + waveform.duration)
        t, rec = self._collect(t_start)
        # NEURON records at the END of each step the current applied DURING it, so shift
        # by one sample: i[k] = current applied over [t_k, t_k + dt)
        i_rec = np.array(self.i_vec[site])
        i_inj = np.r_[i_rec[1:], 0.0]
        keep = t >= -1e-9
        self._silence()
        return t[keep], {s: v[keep] for s, v in rec.items()}, i_inj[keep]
