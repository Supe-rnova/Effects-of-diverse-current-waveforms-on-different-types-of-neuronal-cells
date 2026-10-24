"""
Step 1 - environment check.

Run from the project root:
    python setup_env/check_env.py

It verifies Python packages, NEURON, .mod compilation and a basic HH
simulation, and writes results/logs/environment.txt for the report (Step 11).
"""
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OK, FAIL = "[ OK ]", "[FAIL]"
lines = []


def report(status, msg):
    line = f"{status} {msg}"
    print(line)
    lines.append(line)


def main():
    all_ok = True
    report(OK, f"Python {sys.version.split()[0]} on {platform.system()} {platform.release()}")

    # 1. scientific stack ------------------------------------------------
    for pkg in ("numpy", "scipy", "matplotlib", "pandas", "pytest"):
        try:
            mod = __import__(pkg)
            report(OK, f"{pkg} {mod.__version__}")
        except ImportError:
            report(FAIL, f"{pkg} missing -> pip install -r requirements.txt")
            all_ok = False

    # 2. NEURON ----------------------------------------------------------
    try:
        import neuron
        from neuron import h
        report(OK, f"NEURON {neuron.__version__}")
    except ImportError as e:
        report(FAIL, f"NEURON not importable ({e}). See setup_env/SETUP.md")
        _write_log()
        return False

    # 3. nrnivmodl + loading a compiled mechanism -----------------------
    exe = shutil.which("nrnivmodl")
    report(OK if exe else FAIL, f"nrnivmodl on PATH: {exe}")
    if exe:
        try:
            from models.mechanisms import compile_mechanisms, load_mechanisms
            test_dir = ROOT / "setup_env" / "test_mech"
            compile_mechanisms(test_dir, force=True, verbose=False)
            load_mechanisms(test_dir)
            s = h.Section(name="mechtest")
            s.insert("testpas")
            report(OK, "compiled and loaded a test .mod file")
        except Exception as e:  # noqa: BLE001
            report(FAIL, f".mod compilation/loading failed: {e}")
            all_ok = False
    else:
        all_ok = False

    # 4. HH sanity simulation -------------------------------------------
    try:
        import numpy as np
        h.load_file("stdrun.hoc")
        soma = h.Section(name="soma")
        soma.L = soma.diam = 20.0
        soma.insert("hh")
        stim = h.IClamp(soma(0.5))
        stim.delay, stim.dur, stim.amp = 5, 50, 0.1   # nA
        v = h.Vector().record(soma(0.5)._ref_v)
        h.celsius = 6.3
        h.dt = 0.025
        h.finitialize(-65)
        h.continuerun(60)
        v = np.array(v)
        n_spk = int(np.sum((v[1:] >= 0) & (v[:-1] < 0)))
        good = n_spk >= 2 and v.max() > 30
        report(OK if good else FAIL,
               f"HH test run: {n_spk} spikes, peak {v.max():.1f} mV (expect >=2 spikes, peak > 30 mV)")
        all_ok &= good
    except Exception as e:  # noqa: BLE001
        report(FAIL, f"HH simulation failed: {e}")
        all_ok = False

    # 5. multiprocessing (needed for batch runs) ------------------------
    try:
        from concurrent.futures import ProcessPoolExecutor
        import os
        with ProcessPoolExecutor(max_workers=2) as ex:
            r = list(ex.map(abs, [-1, -2]))
        report(OK, f"multiprocessing works ({os.cpu_count()} CPU cores available)")
    except Exception as e:  # noqa: BLE001
        report(FAIL, f"multiprocessing failed: {e}")
        all_ok = False

    report(OK if all_ok else FAIL, "ENVIRONMENT READY" if all_ok else "FIX THE ITEMS MARKED [FAIL]")
    _write_log()
    return all_ok


def _write_log():
    from config.settings import LOGS_DIR
    (LOGS_DIR / "environment.txt").write_text("\n".join(lines) + "\n")
    print(f"\nlog written to {LOGS_DIR / 'environment.txt'}")


if __name__ == "__main__":       # guard is required for multiprocessing on Windows
    sys.exit(0 if main() else 1)
