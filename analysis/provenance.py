"""
Step 11 - reproducibility record.

    python analysis/provenance.py

Writes results/logs/provenance.json and results/logs/provenance.md with:
software versions (Python, NEURON, numpy, scipy, matplotlib, pandas), platform,
model sources (ModelDB accession, reference, MD5 of every .mod/.hoc file used),
all random seeds, numerical settings and the full experiment design.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import experiments as E  # noqa: E402
from config import settings as S  # noqa: E402


def _ver(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "?")
    except Exception:  # noqa: BLE001
        return "not installed"


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def model_sources():
    from models.fetch_models import SOURCES
    out = {"HH": {"modeldb": None, "reference": "Hodgkin AL, Huxley AF (1952) J Physiol 117:500-544",
                  "files": {"(built-in NEURON 'hh' mechanism)": None}}}
    for folder, (acc, ref, _) in SOURCES.items():
        d = ROOT / "models" / folder
        files = {}
        for p in sorted(list(d.glob("*.mod")) + list(d.glob("*.hoc"))):
            files[p.name] = _md5(p)
        out[folder] = {"modeldb": acc, "url": f"https://modeldb.science/{acc}",
                       "mirror": f"https://github.com/ModelDBRepository/{acc}",
                       "reference": ref, "files": files}
    return out


def design():
    keys = ["PRE", "STEP_AMPS", "HYPER_AMPS", "RAMPS", "RAMP_PEAK", "SINE_FREQS", "SINE_AMPS",
            "SINE_DUR", "CHIRP", "PULSE", "NOISE", "SYN", "DEND", "BLOCKS", "BLOCK_SET"]
    return {k: getattr(E, k) for k in keys}


def seeds():
    return {
        "GLOBAL_SEED": S.GLOBAL_SEED,
        "noise_variability_seeds": [1001 + k for k in range(E.NOISE["n_var"])],
        "frozen_noise_signal_seed": E.NOISE["signal_seed"],
        "frozen_noise_trial_seeds": [3001 + k for k in range(E.NOISE["n_rel"])],
        "synaptic_background_seeds": [4001 + k for k in range(E.SYN["n_trials"])],
        "random_phase_lead_in": "per task: numpy PCG64 seeded with sha1(task id)[:8]",
        "rng": "numpy.random.Generator(PCG64)",
    }


def main():
    rec = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": sys.version.split()[0],
        "packages": {m: _ver(m) for m in ("neuron", "numpy", "scipy", "matplotlib", "pandas")},
        "numerics": {"dt_ms": S.DT, "dt_check_ms": S.DT_CHECK, "integration": "fixed step (backward Euler, NEURON default)",
                     "spike_detection": {"dvdt_mV_per_ms": S.SPIKE_DVDT_THRESH, "min_peak_mV": S.SPIKE_MIN_PEAK,
                                         "refractory_ms": S.SPIKE_REFRACTORY}},
        "models": model_sources(),
        "seeds": seeds(),
        "experiment_design": design(),
    }
    for f in ("characterisation.json", "pulse_thresholds.json"):
        p = S.DATA_DIR / f
        if p.exists():
            rec.setdefault("derived_inputs_md5", {})[f] = _md5(p)
    idx = S.DATA_DIR / "runs_index.csv"
    if idx.exists():
        lines = idx.read_text().strip().splitlines()[1:]
        rec["batch"] = {"n_tasks": len(lines), "n_complete": sum(l.endswith(",1") for l in lines)}
    (S.LOGS_DIR / "provenance.json").write_text(json.dumps(rec, indent=2, default=str))

    md = [f"# Provenance ({rec['generated']})", "",
          f"* Platform: {rec['platform']}", f"* Python {rec['python']}",
          "* " + ", ".join(f"{k} {v}" for k, v in rec["packages"].items()),
          f"* dt = {S.DT} ms (fixed step), convergence checked at {S.DT_CHECK} ms", "",
          "## Models", "| cell | ModelDB | reference | files (MD5) |", "|---|---|---|---|"]
    for name, m in rec["models"].items():
        files = "; ".join(f"{k} `{(v or '')[:8]}`" for k, v in m["files"].items())
        md.append(f"| {name} | {m['modeldb'] or '-'} | {m['reference']} | {files} |")
    md += ["", "## Seeds"] + [f"* {k}: {v}" for k, v in rec["seeds"].items()]
    md += ["", "## Experiment design", "```json", json.dumps(rec["experiment_design"], indent=1), "```"]
    if "batch" in rec:
        md += ["", f"Batch: {rec['batch']['n_complete']}/{rec['batch']['n_tasks']} runs complete."]
    (S.LOGS_DIR / "provenance.md").write_text("\n".join(md) + "\n")
    print(f"wrote {S.LOGS_DIR / 'provenance.json'} and provenance.md")
    print("  " + ", ".join(f"{k} {v}" for k, v in rec["packages"].items()))


if __name__ == "__main__":
    main()
