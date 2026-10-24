"""
Global settings shared by every script in the project.

Edit values here rather than hard-coding them in individual scripts, so the
whole pipeline stays consistent and reproducible (Step 11).
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = RESULTS_DIR / "data"
STATES_DIR = RESULTS_DIR / "states"      # saved steady states (h.SaveState)
LOGS_DIR = RESULTS_DIR / "logs"

for _d in (RESULTS_DIR, FIGURES_DIR, DATA_DIR, STATES_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- numerics
DT = 0.025            # ms, fixed integration step (Step 6)
DT_CHECK = 0.0125     # ms, used for the dt-convergence check (change #8)
GLOBAL_SEED = 20261023

# ---------------------------------------------------------------- spike detection (change #9)
SPIKE_DVDT_THRESH = 20.0   # mV/ms, upward rate-of-rise criterion
SPIKE_MIN_PEAK = -20.0     # mV, peak must exceed this to count as a somatic spike
SPIKE_REFRACTORY = 1.0     # ms, minimum separation between detected spikes

# ---------------------------------------------------------------- models
# The authoritative per-model settings (temperature, sites, channels) live in
# models/cells.py. Order used everywhere in the pipeline:
MODEL_NAMES = ["HH", "STN", "Purkinje", "CA1"]
