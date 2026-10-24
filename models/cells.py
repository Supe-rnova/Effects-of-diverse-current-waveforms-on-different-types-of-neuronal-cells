"""
Step 2b - one common Python interface for the four neuron models.

    from models.cells import build_cell
    cell = build_cell("STN")          # "HH", "STN", "Purkinje", "CA1"
    cell.sites["soma"]                # NEURON segment for injection/recording
    cell.sites.get("dend")            # dendritic site (Purkinje, CA1 only)
    cell.block("CaT", 0.0)            # in-silico channel blockade (change #5)
    cell.restore_channels()

RULE: build only ONE model per Python process (change #6). Mechanism names
clash between models (e.g. `CaT` exists in both STN and Purkinje).
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.mechanisms import load_mechanisms  # noqa: E402

MODELS_DIR = ROOT / "models"
_BUILT: str | None = None          # which model this process has already built


@contextmanager
def _cwd(path: Path):
    """hoc files open data files with relative paths, so load them from their folder."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


@dataclass
class Cell:
    name: str
    celsius: float              # temperature the model was published/validated at
    v_init: float               # initial voltage passed to finitialize
    settle_ms: float            # time to reach steady state before any stimulus
    sites: dict                 # {"soma": seg, "dend": seg}
    reference: str
    modeldb: int | None
    channels: dict = field(default_factory=dict)   # blockable channel -> list of RANGE params
    spontaneous: bool = False   # fires without input at rest?
    _orig: dict = field(default_factory=dict, repr=False)

    # ----------------------------------------------------------- blockade
    def _segments(self):
        from neuron import h
        for sec in h.allsec():
            for seg in sec:
                yield seg

    def _snapshot(self):
        """Store original conductance values so blockade is exactly reversible."""
        for ch, params in self.channels.items():
            vals = []
            for seg in self._segments():
                for p in params:
                    if hasattr(seg, p):
                        vals.append((seg, p, getattr(seg, p)))
            self._orig[ch] = vals

    def block(self, channel: str, factor: float = 0.0) -> None:
        """Scale every conductance of `channel` to factor x its original value."""
        if channel not in self.channels:
            raise KeyError(f"{self.name}: unknown channel '{channel}'. Options: {list(self.channels)}")
        for seg, p, v in self._orig[channel]:
            setattr(seg, p, v * factor)

    def restore_channels(self) -> None:
        for vals in self._orig.values():
            for seg, p, v in vals:
                setattr(seg, p, v)

    def n_segments(self) -> int:
        return sum(1 for _ in self._segments())


# =================================================================== builders
def _build_hh():
    from neuron import h
    soma = h.Section(name="soma")
    soma.L = soma.diam = 18.8          # um  (NEURON tutorial geometry, ~1100 um^2)
    soma.Ra, soma.cm = 123.0, 1.0
    soma.insert("hh")                  # classic squid-axon parameters
    return Cell(
        name="HH", celsius=6.3, v_init=-65.0, settle_ms=100.0,
        sites={"soma": soma(0.5)},
        reference="Hodgkin AL, Huxley AF (1952) J Physiol 117:500-544",
        modeldb=None,
        channels={"Na": ["gnabar_hh"], "K": ["gkbar_hh"]},
    )


def _build_stn():
    from neuron import h
    folder = MODELS_DIR / "stn"
    load_mechanisms(folder)
    with _cwd(folder):
        h.load_file("stn_build.hoc")
    c = h.SThcells[0]
    return Cell(
        name="STN", celsius=37.0, v_init=-60.0, settle_ms=1000.0,
        sites={"soma": c.soma(0.5)},
        reference="Gillies A, Willshaw D (2006) J Neurophysiol 95:2352-2365",
        modeldb=74298,
        channels={
            "CaT": ["gcaT_CaT"],          # T-type Ca2+  (rebound bursts)
            "HCN": ["gk_Ih"],             # Ih           (sag)
            "NaL": ["gna_NaL"],           # persistent Na+ (pacemaking)
            "sKCa": ["gk_sKCa"],          # SK channels  (AHP)
            "CaL": ["gcaL_HVA"],          # L-type Ca2+  (plateaus)
        },
        spontaneous=True,
    )


def _build_purkinje():
    from neuron import h
    folder = MODELS_DIR / "purkinje"
    load_mechanisms(folder)
    with _cwd(folder):
        h.load_file("purkinje_nogui.hoc")
    soma = h.soma
    h.distance(0, soma(0.5))
    # dendritic site: point on the smooth (main) dendrite closest to 100 um path distance
    dend = min((seg for i in range(int(h.NumSmoothDendrite)) for seg in h.SmoothDendrite[i]),
               key=lambda s: abs(h.distance(s) - 100.0))
    return Cell(
        name="Purkinje", celsius=37.0, v_init=-65.0, settle_ms=500.0,
        sites={"soma": soma(0.5), "dend": dend},
        reference="Miyasho T et al. (2001) Brain Res 891:106-115",
        modeldb=17664,
        channels={
            "CaP": ["gcabar_CaP2"],       # P-type Ca2+  (dendritic Ca spikes)
            "CaT": ["gcabar_CaT"],        # T-type Ca2+
            "CaE": ["gcabar_CaE"],        # class-E (R-type) Ca2+
            "HCN": ["gkbar_Kh"],          # Ih
            "KD":  ["gkbar_KD"],          # D-type K+
        },
        spontaneous=False,                # verified: silent at rest (~ -69 mV)
    )


_KEEP = []   # keeps hoc template objects alive (otherwise the cell is garbage-collected)


def _build_ca1():
    from neuron import h
    folder = MODELS_DIR / "ca1"
    load_mechanisms(folder)
    h.celsius, h.v_init = 34.0, -70.0         # values the model was optimised with
    with _cwd(folder):                         # template reads morphologies/ relatively
        h.load_file("cell_seed3_0-pyr-08.hoc")
        c = h.CA1_PC_cAC_sig6()                # continuous-accommodating pyramidal cell
    _KEEP.append(c)
    soma = c.soma[0]
    h.distance(0, soma(0.5))
    # dendritic site: apical point ~300 um from the soma; among the candidates within
    # +-25 um take the thickest one, i.e. the main trunk rather than a thin oblique
    cands = [seg for sec in c.apical for seg in sec if abs(h.distance(seg) - 300.0) < 25.0]
    dend = max(cands, key=lambda s: s.diam)
    return Cell(
        name="CA1", celsius=34.0, v_init=-70.0, settle_ms=500.0,
        sites={"soma": soma(0.5), "dend": dend},
        reference=("Migliore R, Lupascu CA, Bologna LL, et al. (2018) PLoS Comput Biol "
                   "14:e1006423 (cell CA1_PC_cAC_sig6, cell_seed3_0-pyr-08)"),
        modeldb=244688,
        channels={
            "HCN": ["ghdbar_hd"],                     # Ih gradient (theta resonance, sag)
            "KM":  ["gbar_kmb"],                      # M-current (adaptation, M-resonance)
            "KA":  ["gkabar_kap", "gkabar_kad"],      # A-type K+ (prox./dist.)
            "KCa": ["gbar_kca", "gbar_cagk"],         # Ca2+-activated K+ (AHP, adaptation)
            "CaT": ["gcatbar_cat"],
            "Na":  ["gbar_nax"],
        },
    )


_BUILDERS = {"HH": _build_hh, "STN": _build_stn, "Purkinje": _build_purkinje, "CA1": _build_ca1}
MODEL_NAMES = list(_BUILDERS)


def build_cell(name: str) -> Cell:
    global _BUILT
    if name not in _BUILDERS:
        raise KeyError(f"unknown model '{name}', choose from {MODEL_NAMES}")
    if _BUILT is not None:
        raise RuntimeError(f"This process already built '{_BUILT}'. "
                           "Build each model in its own Python process.")
    from neuron import h
    h.load_file("stdrun.hoc")
    cell = _BUILDERS[name]()
    h.celsius = cell.celsius
    cell._snapshot()
    _BUILT = name
    return cell
