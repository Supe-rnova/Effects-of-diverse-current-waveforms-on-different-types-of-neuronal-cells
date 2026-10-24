# Computational Analysis of Neuronal Responses to Diverse Current Injection Waveforms — Group 9

4 neuron models (HH, STN, Purkinje, CA1) × 8 current waveforms, simulated in
NEURON (Python), with amplitudes scaled to each cell's rheobase.

## Folder layout
```
config/        settings.py – paths, dt, seeds, spike-detection and model metadata
models/        mechanisms.py (compile/load helpers) + one subfolder per cell
waveforms/     waveform generator (Step 5)
simulations/   injection framework + batch runner (Steps 6–7)
analysis/      spike detection, metrics, plots (Steps 8–9)
results/       data/, figures/, states/, logs/  (generated – not hand-edited)
tests/         pytest unit tests
setup_env/     SETUP.md, check_env.py
```

## Revised pipeline (original 11 steps + adopted changes)
| Step | Original | Changes adopted |
|---|---|---|
| 1 Setup | pip install neuron, folders | Windows installer route; `check_env.py` verifies compile + HH run |
| 2 Models | ModelDB + nrnivmodl | **One model per Python process** (mechanism name clashes, e.g. `CaT`) |
| 3 Validate | reproduce 1 published result, set celsius | dt-convergence check (dt vs dt/2) |
| 4 Characterise | Rin, τm, rheobase, baseline | **Native vs Held** conditions for spontaneously firing STN/Purkinje (holding current that silences the cell; rheobase measured on top). Settle once and store with `h.SaveState` |
| 5 Waveforms | one function per waveform | **Step-exact edges**, pulse widths multiple of dt, net-charge = 0 test for biphasic pulses |
| 6 Injection | IClamp + Vector.play | restore saved state; random initial phase for repeated trials on spontaneous cells |
| 7 Experiments | rheobase-scaled amps, few freqs, 20 seeds | Sinusoid grid ~8 log-spaced freqs (1–100 Hz) × 3 amps; pulse trains add 5 & 200 Hz; **frozen-noise + independent background noise** for reliability/jitter; dendritic sites get their own rheobase; **in-silico channel blockade** (HCN, T-type Ca²⁺, dendritic Ca²⁺) |
| 8 Analysis | 0 mV threshold, metrics table | dV/dt + peak spike detector; dendritic Ca²⁺ spikes counted separately; chirp impedance from the time-domain FFT only (`h.Impedance` misses Ih gating in these models), subthreshold check, Q-factor; Rayleigh test for vector strength |
| 9 Visualisation | traces, f-I, impedance, rasters, heatmap | per-metric normalisation for heatmap; mean ± 95 % CI |
| 10 Interpretation | link to channels, literature | causal evidence from channel blockade; HH temperature confound stated |
| 11 Reporting | seeds, versions, main.py | unchanged |

## How to run
1. Set up the environment once: `setup_env/SETUP.md` (WSL2 recommended - ~10x faster than native Windows NEURON).
2. Run everything:  `python main.py`   (≈1.5 h on a 12-core PC; Step 7 is resumable)
   * `python main.py --list` shows the steps; `--from <step>`, `--only <step>`, `--skip demo`.
3. Outputs
   * `results/figures/final/` – report figures (Step 9)
   * `results/data/analysis/` – metrics (`metrics_runs.csv`, `summary.json`, `heatmap_table.csv`)
   * `results/logs/provenance.md` – versions, seeds, model checksums, full design
   * `analysis/INTERPRETATION.md` – cross-model interpretation (Step 10)
   * `analysis/README.md` – metric definitions; `models/README.md` – model choices
