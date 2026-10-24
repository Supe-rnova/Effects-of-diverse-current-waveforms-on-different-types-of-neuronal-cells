# Step 1 — Environment setup (Windows + VS Code)

NEURON is **not** available through `pip` on Windows (there are no Windows
wheels on PyPI). Use the official installer.

## 1. Python
1. Install **Python 3.11** from python.org (64-bit). Tick **"Add python.exe to PATH"**.
   *Why 3.11:* the NEURON installer only supports certain Python versions — they are
   listed in the installer file name (e.g. `...-py-39-310-311-312-setup.exe`).
   Pick a Python version that appears there.
2. Check: open a new terminal → `python --version`

## 2. NEURON
1. Download the latest Windows installer (`nrn-9.0.x-...-setup.exe`) from
   https://www.neuron.yale.edu/neuron/download (or the GitHub releases page of
   `neuronsimulator/nrn`).
2. Run it. When asked, **keep the option that sets PATH and PYTHONPATH** enabled.
   Default install location is `C:\nrn`.
3. **Restart VS Code** (and any terminals) so the new PATH is picked up.

## 3. VS Code
1. File → Open Folder → `Project_CompN`
2. `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the Python 3.11 you installed.
3. Open a terminal (`Ctrl+\``) in the project root and run:
   ```
   pip install -r requirements.txt
   python setup_env/check_env.py
   ```
4. All lines should show `[ OK ]` and end with `ENVIRONMENT READY`.
   The result is saved to `results/logs/environment.txt`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `No module named 'neuron'` | Add `C:\nrn\lib\python` to the **PYTHONPATH** user environment variable, restart VS Code. |
| `nrnivmodl not found on PATH` | Add `C:\nrn\bin` to **PATH**, restart VS Code. As a fallback, compile with Start menu → NEURON → **mknrndll** and select the folder with the .mod files. |
| `DLL load failed` / `ImportError` when importing neuron | Your Python version is not one the installer supports — install a matching version and re-select the interpreter. |
| Compilation errors mentioning `cygwin`/`mingw` | Reinstall NEURON; the installer ships its own compiler — don't mix with another MinGW on PATH. |

## Fallback: WSL2 (Linux inside Windows)
If the native install keeps failing:
```
wsl --install -d Ubuntu          # in an admin PowerShell, then reboot
# inside Ubuntu:
sudo apt update && sudo apt install -y python3-pip python3-venv build-essential
cd "/mnt/c/Users/Supernova/Desktop/college/Amrita/SEM 3/Comp_Neuro/Project_CompN"
python3 -m venv .venv && source .venv/bin/activate
pip install neuron -r requirements.txt
python setup_env/check_env.py
```
In VS Code install the **WSL** extension and use "Reopen Folder in WSL".
