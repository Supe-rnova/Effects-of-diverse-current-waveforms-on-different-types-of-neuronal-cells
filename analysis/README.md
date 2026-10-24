# Analysis (Step 8) – metric definitions

Run: `python analysis/run_analysis.py` (≈1 min). Outputs in `results/data/analysis/`.

Spikes: upward dV/dt ≥ 20 mV/ms, peak ≥ −20 mV, 1 ms refractory (`analysis/spikes.py`).
Dendritic Ca²⁺ spikes (Purkinje): dendritic V ≥ −10 mV for ≥ 3 ms.
Time 0 = waveform start; stimuli start at 200 ms (chirp 500 ms).

| Protocol | Metric | Definition |
|---|---|---|
| Step | rate, steady rate | spikes / step duration; spikes in 2nd half / half duration |
| | latency | first spike − step onset |
| | adaptation index | last ISI / first ISI |
| | depolarisation block | firing stops > max(200 ms, 3 mean ISIs) before step end |
| | spike threshold | V at the dV/dt crossing of the first spike |
| Hyperpolarising step | sag ratio | (V_base − V_ss) / (V_base − V_peak); 1 = no sag |
| | rebound spikes / latency / max rate | spikes in 300 ms after release; first-spike delay; 1/min ISI |
| Ramp | threshold current | injected current at the first spike |
| Sinusoid | vector strength (VS) | \|mean e^{iφ}\| of spike phases, first cycle (≥100 ms) excluded |
| | Rayleigh p | Zar (1999) approximation; p < 0.05 = significant locking |
| | spikes/cycle, cycles with spike | entrainment ratio; fraction of cycles with ≥ 1 spike |
| Chirp (subthreshold) | \|Z(f)\| | cross-spectral FFT(V)·FFT(I)* / \|FFT(I)\|², Hann window, 60 log bins |
| | f_R, Q | frequency of max \|Z\|; Q = \|Z(f_R)\| / \|Z(lowest f)\| |
| | validity | run flagged invalid if any spike occurred |
| Pulse train | fidelity | fraction of pulses followed by a spike within min(period, 5 ms) |
| | fidelity first 10 / 2nd half | onset vs sustained following |
| Noise (variability) | ISI CV, rate | 20 different seeds, first 100 ms excluded; mean ± 95 % CI (t) |
| | spike-triggered average | mean injected current in the 50 ms before each spike |
| Noise (reliability) | Schreiber R | mean pairwise cosine similarity of Gaussian-filtered (σ = 3 ms) spike trains across 20 frozen-noise trials; 95 % CI by bootstrap |
| | event jitter | SD of spike times within PSTH events reached by ≥ 50 % of trials |
| Synaptic | spike probability | P(spike within 20 ms of an event) over 5 events × 20 trials; Wilson 95 % CI |
| | jitter | SD of first-spike latency across trials, averaged over events |

Notes for interpretation
* STN *held* rheobase (0.067 nA) is the threshold for a transient onset spike; sustained firing
  from −70 mV needs ≈3× that, so suprathreshold rate/noise/synaptic metrics for STN are taken
  from the *native* condition and the held condition is used for sag, rebound and resonance.
* Pulse trains at 2× single-pulse threshold drive most cells near-perfectly; the 1.2× level
  discriminates between cells.
