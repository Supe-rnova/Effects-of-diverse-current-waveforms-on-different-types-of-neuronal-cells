"""
Step 6 - injection framework.

    from simulations.injection import run_simulation
    res = run_simulation("CA1", waveform, site="soma")        # plan's single entry point

or, for many runs in one process (faster - the cell is built and settled once):

    sim = Simulator("STN")
    sim.prepare(condition="held")                  # 'native' | 'held' (spontaneous cells)
    sim.prepare(condition="native", block="CaT")   # in-silico channel blockade
    res = sim.run(waveform, site="dend", lead_in=37.5)

A result is a dict:
    t        ms, 0 = waveform start (dt = config.settings.DT)
    v        {site: membrane potential (mV)} for soma (+ dend if the model has one)
    i        injected current (nA) at the stimulated site, i[k] applied over [t_k, t_k+dt)
    meta     model, site, condition, block, holding current, waveform kind + parameters, ...

Holding currents, settling times and rheobase units come from Step 4
(results/data/characterisation.json). One model per process (change #6).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import DATA_DIR, DT  # noqa: E402

CHAR_FILE = DATA_DIR / "characterisation.json"


def load_characterisation() -> dict:
    if not CHAR_FILE.exists():
        raise FileNotFoundError(f"{CHAR_FILE} missing - run simulations/characterise.py (Step 4) first")
    return json.loads(CHAR_FILE.read_text())


class Simulator:
    def __init__(self, model: str):
        from models.cells import build_cell
        from simulations.engine import Engine
        self.model = model
        self.cell = build_cell(model)
        self.char = load_characterisation()[model]
        self.sites = tuple(s for s in ("soma", "dend") if s in self.cell.sites)
        self.eng = Engine(self.cell, sites=self.sites, record=self.sites)
        self.condition, self.block, self.block_factor = None, None, 1.0

    # ------------------------------------------------------------------ set-up
    def conditions(self):
        return list(self.char["conditions"])

    def unit(self, site: str = "soma") -> float:
        """Amplitude unit (nA) = rheobase of the silent condition at that site."""
        if site == "soma":
            return self.char["amplitude_unit_nA"]
        u = self.char.get("amplitude_unit_dend_nA")
        if u is None:
            raise KeyError(f"{self.model} has no dendritic rheobase")
        return u

    def prepare(self, condition: str = "native", block: str | None = None, block_factor: float = 0.0):
        """Select condition and optional channel block; settles once per combination."""
        if condition not in self.char["conditions"]:
            raise KeyError(f"{self.model}: condition '{condition}' not characterised "
                           f"(available: {self.conditions()})")
        self.cell.restore_channels()
        if block is not None:
            self.cell.block(block, block_factor)
        key = (condition, block, block_factor if block else None)
        if not self.eng.use(key):
            c = self.char["conditions"][condition]
            self.eng.set_hold(c["i_hold_nA"])
            settle = c["settle_ms"] if block is None else max(c["settle_ms"], 1500.0)
            self.eng.settle(settle, key=key)
        self.condition, self.block = condition, block
        self.block_factor = block_factor if block else 1.0

    # ------------------------------------------------------------------ running
    def run(self, waveform, site: str = "soma", lead_in: float = 0.0, extra_meta: dict | None = None) -> dict:
        if self.condition is None:
            self.prepare()
        if site not in self.sites:
            raise KeyError(f"{self.model} has no site '{site}' (available: {self.sites})")
        t, v, i = self.eng.run_waveform(site, waveform, lead_in=lead_in)
        meta = {
            "model": self.model, "site": site, "condition": self.condition,
            "block": self.block, "block_factor": self.block_factor,
            "i_hold_nA": self.char["conditions"][self.condition]["i_hold_nA"],
            "celsius": self.cell.celsius, "dt": DT, "lead_in_ms": lead_in,
            "waveform": waveform.kind, "params": waveform.params,
            "charge_pC": float(i[:-1].sum() * DT),
        }
        if extra_meta:
            meta.update(extra_meta)
        return {"t": t, "v": v, "i": i, "meta": meta}


_SIM: Simulator | None = None


def run_simulation(model: str, waveform, params: dict | None = None, site: str = "soma") -> dict:
    """Plan's interface: run_simulation(model, waveform, params, site) -> t, v, i (+meta).
    `params` may contain condition, block, block_factor, lead_in."""
    global _SIM
    params = params or {}
    if _SIM is None:
        _SIM = Simulator(model)
    elif _SIM.model != model:
        raise RuntimeError("one model per process - start a new process for another model")
    _SIM.prepare(params.get("condition", "native"), params.get("block"), params.get("block_factor", 0.0))
    return _SIM.run(waveform, site=site, lead_in=params.get("lead_in", 0.0))


# ===================================================================== storage
def save_result(path: str | Path, res: dict) -> None:
    """Compressed .npz: voltages/current as float32, metadata as JSON (time is implicit: k*dt)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"v_{s}": v.astype(np.float32) for s, v in res["v"].items()}
    arrays["i"] = res["i"].astype(np.float32)
    np.savez_compressed(path, meta=json.dumps(res["meta"], default=_jsonable), **arrays)


def load_result(path: str | Path) -> dict:
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    v = {k[2:]: d[k].astype(float) for k in d.files if k.startswith("v_")}
    n = len(next(iter(v.values())))
    return {"t": np.arange(n) * meta["dt"], "v": v, "i": d["i"].astype(float), "meta": meta}


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
