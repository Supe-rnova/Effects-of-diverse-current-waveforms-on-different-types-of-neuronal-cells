"""
Cross-platform helpers to compile and load NEURON .mod files.

* Linux / macOS / WSL : `nrnivmodl` -> <dir>/x86_64 (or arm64)/...libnrnmech.so
* Windows (installer) : `nrnivmodl` (a .bat shipped with NEURON) -> <dir>/nrnmech.dll

IMPORTANT (pipeline change #6): load the mechanisms of only ONE model per
Python process. Several ModelDB models define mechanisms with the same name
(e.g. both STN and Purkinje define `CaT`), and NEURON refuses to load a
second library that redefines an existing mechanism.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

IS_WINDOWS = platform.system() == "Windows"


def _compiled_marker(mod_dir: Path) -> Path | None:
    """Return the compiled library FOR THIS OPERATING SYSTEM if it exists, else None.

    Windows and Linux/WSL builds can sit side by side in the same folder
    (nrnmech.dll vs x86_64/), so only look for the one this platform can load.
    """
    if IS_WINDOWS:
        dll = mod_dir / "nrnmech.dll"
        return dll if dll.exists() else None
    for arch in ("x86_64", "arm64", "aarch64", "i686"):
        if (mod_dir / arch).exists():
            for lib in (mod_dir / arch).glob("**/libnrnmech.*"):
                return lib
    return None


def is_compiled(mod_dir: str | Path) -> bool:
    return _compiled_marker(Path(mod_dir)) is not None


def compile_mechanisms(mod_dir: str | Path, force: bool = False, verbose: bool = True) -> Path:
    """Run nrnivmodl in `mod_dir`. Returns the path of the compiled library."""
    mod_dir = Path(mod_dir).resolve()
    if not list(mod_dir.glob("*.mod")):
        raise FileNotFoundError(f"No .mod files found in {mod_dir}")
    if is_compiled(mod_dir) and not force:
        if verbose:
            print(f"[mechanisms] already compiled: {mod_dir}")
        return _compiled_marker(mod_dir)

    exe = shutil.which("nrnivmodl")
    if exe is None:
        hint = ("On Windows, make sure the NEURON installer added its 'bin' folder to PATH, "
                "or compile manually: Start menu > NEURON > mknrndll, then pick this folder.")
        raise RuntimeError(f"`nrnivmodl` not found on PATH. {hint}")

    if verbose:
        print(f"[mechanisms] compiling {mod_dir} ...")
    proc = subprocess.run([exe], cwd=mod_dir, capture_output=True, text=True,
                          shell=IS_WINDOWS)
    if proc.returncode != 0 or not is_compiled(mod_dir):
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise RuntimeError(f"nrnivmodl failed in {mod_dir}")
    if verbose:
        print(f"[mechanisms] OK -> {_compiled_marker(mod_dir)}")
    return _compiled_marker(mod_dir)


def load_mechanisms(mod_dir: str | Path, compile_if_needed: bool = True) -> None:
    """Compile (if needed) and load the mechanisms found in `mod_dir`."""
    import neuron  # imported here so this module can be imported without NEURON

    mod_dir = Path(mod_dir).resolve()
    if compile_if_needed and not is_compiled(mod_dir):
        compile_mechanisms(mod_dir)
    ok = neuron.load_mechanisms(str(mod_dir), warn_if_already_loaded=False)
    if not ok:
        raise RuntimeError(f"Could not load mechanisms from {mod_dir}")
