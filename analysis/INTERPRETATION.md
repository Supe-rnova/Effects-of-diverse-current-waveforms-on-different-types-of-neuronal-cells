# Step 10 – Cross-model comparison and interpretation

Data: 979 simulations (Step 7), metrics from `analysis/run_analysis.py` (Step 8), figures
in `results/figures/final/` (Step 9). Amplitudes are multiples of each cell's own rheobase
(pulse trains: of the single-pulse threshold). "Fig. N" refers to `NN_*.png`.

---

## 1. Key findings

1. **T-type Ca²⁺ is the STN's signature.** From −70 mV, a −2× hyperpolarising step evokes a
   15-spike rebound burst at 170–200 Hz; blocking T-type Ca²⁺ abolishes it (15 → 0), blocking
   Ih does not (15 → 15). The burst appears only when the trough passes ≈ −78 mV (none at
   −76 mV, full burst at −81 mV) – matching the experimental rebound threshold of
   −78 ± 3 mV (Bevan et al. 2000). (Fig. 4, 9)
2. **Two different routes to theta resonance.** STN (held, −70 mV) resonates at 5.2 Hz with
   the strongest peak of all cells (Q = 2.53); removing T-type Ca²⁺ cuts Q to 1.23, so the
   T-current *amplifies* the resonance. CA1 resonates at 10 Hz (Q = 1.39); removing Ih
   drops it to 3.3 Hz / Q 1.08 and removing the M-current to Q 1.19 – the H- and
   M-resonances described by Hu et al. (2002). Purkinje is a pure low-pass filter
   (no peak) and HH resonates far above theta (43 Hz, Q 2.38). (Fig. 3, 9)
3. **Dendritic Ca²⁺ spikes govern the Purkinje cell.** Strong somatic or any dendritic input
   triggers P-type-dependent dendritic Ca²⁺ spikes that reshape firing into bursts
   (dendritic-injection ISI CV 1.41, the most irregular response in the study). Blocking
   P-type Ca²⁺ leaves only one spike at 2× rheobase (86 Hz → 1 Hz). Blocking the D-type
   (Kv1-like) K⁺ current turns the silent cell into a 50 Hz spontaneous burster with ~30
   dendritic Ca²⁺ spikes/s – the hyperexcitability reported by Khavandgar et al. (2005). (Fig. 2, 9)
4. **Integrators vs resonators.** HH (type II) has an abrupt f-I onset (0 → 57 Hz between
   2.5× and 3× rheobase), does not fire to slow ramps up to 3× (accommodation), fires an
   anode-break spike after hyperpolarisation, and its spike-triggered average is biphasic
   (a hyperpolarising lobe ~10 ms before the spike). CA1 behaves as an integrator: a
   graded f-I curve, a monophasic, slowly rising STA (~20 ms) and firing on ramps. (Fig. 2, 7)
5. **High-frequency pulses: STN follows, CA1 and HH fail.** At 1.2× single-pulse threshold,
   STN (native) follows 1:1 up to 200 Hz with either polarity, Purkinje 0.88 at 130 Hz,
   while CA1 (0.16) and HH (0.12) fail in the DBS range. Charge-balanced biphasic pulses at
   1.2× never fire HH (0/all), and roughly halve following in held STN (1.0 → 0.53 at
   130 Hz) – the cathodic phase cancels part of the depolarisation. (Fig. 6)
6. **Variability ≠ reliability.** STN fires most regularly to noise (ISI CV 0.22, its
   pacemaker dominates) while HH is the most irregular (CV 0.73) yet the most reliable to
   repeated frozen noise (R = 0.87, jitter 0.5 ms): HH only fires on specific fast input
   transients, so when it fires, it fires at the same time. (Fig. 7)

---

## 2. Response features linked to channels

| Cell | Response feature | Channel responsible | Causal evidence (this study) |
|---|---|---|---|
| HH | Abrupt f-I onset, high resonance (43 Hz), biphasic STA, anode-break spike | Fast Na⁺ inactivation + delayed-rectifier K⁺ (type-II dynamics) | Only two conductances present; features follow from their kinetics |
| HH | No spike to slow ramps (≤ 3×) | Na⁺ inactivation + K⁺ activation during slow depolarisation (accommodation) | Fast ramp (0.5 s) also fails – threshold rises with the stimulus |
| STN | Pacemaking ~10 Hz, linear f-I without adaptation (index 1.07–1.14) | Persistent Na⁺ (NaL) + Kv3.1/sKCa | Native condition; regular even under noise (CV 0.22) |
| STN | Rebound burst (15 spikes, 170–200 Hz) | **T-type Ca²⁺** | CaT block → 0 spikes; HCN block → unchanged |
| STN | Sag (ratio 0.52) | **Ih (HCN)** | – |
| STN | Theta resonance 5.2 Hz, Q 2.53 | Ih resonance **amplified by T-type Ca²⁺** | CaT block → Q 1.23 |
| Purkinje | Somatic onset spike, then bursting at ≥ 2× | Dendritic **P-type Ca²⁺** spikes driving somatic Na⁺ bursts | CaP block → 1 spike; Ca²⁺ spikes 17–18 /s in control |
| Purkinje | Silent at rest; low dendritic excitability | Dendritic **D-type (Kv1-like) K⁺** | KD block → 50 Hz spontaneous firing, ~30 Ca²⁺ spikes/s |
| Purkinje | No resonance (low-pass) | Ih confined to the soma, small | – |
| CA1 | Graded, low f-I (17–25 Hz) with strong adaptation (index ≈ 3) | **M-current** + Ca²⁺-activated K⁺ | KM block → 29 Hz at 2×, adaptation 2.3 |
| CA1 | Sag (0.76) and theta resonance 10 Hz | **Ih** (somato-dendritic gradient) | HCN block → sag ratio 1.0, fR 3.3 Hz, Q 1.08 |
| CA1 | Residual resonance without Ih | **M-current** (M-resonance) | KM block → Q 1.19; HCN block leaves Q 1.08 |
| CA1 | Higher resonance in dendrite (13.4 vs 10 Hz) | Ih density increasing with distance | – |

---

## 3. Comparison across waveform families

**Steps (Fig. 2).** Rheobase hides very different gains. STN (native) rises 52 → 109 Hz over
0.8–3× without adaptation; Purkinje jumps from a single onset spike (≤ 1.5×) to 86–122 Hz once
dendritic Ca²⁺ spikes engage; CA1 stays low (17–25 Hz) with the strongest adaptation; HH gives
only an onset spike until it crosses its repetitive-firing threshold (6.5 µA/cm², 56 Hz).
In the *held* STN, sustained firing from −70 mV needs ≈ 3× the transient rheobase (30 Hz at 3×,
ramp threshold 2.78×) – the hold current must be overcome first.

**Hyperpolarising steps (Fig. 4).** Sag strength ranks STN > CA1 > Purkinje (0.52, 0.76, 0.89),
following Ih density. Only STN rebounds substantially; HH's single rebound spike is an
anode-break spike from Na⁺ de-inactivation (HH has no Ih) – its apparent "sag" at small
steps is relaxation of its own gating variables, not Ih.

**Ramps.** HH never fires (accommodation); STN (held) needs 2.8× rheobase; Purkinje 2.0×; CA1
fires at 1.5× (slow) and 1.4× (fast) – the integrator fires as soon as current accumulates.

**Sinusoids (Fig. 5).** Each cell's spikes prefer its intrinsic band: CA1 locks one spike per
cycle at 4–10 Hz (VS 1.0) and stops firing above 20 Hz; held STN locks at 2–10 Hz
(VS 0.85–0.92); HH and Purkinje only fire for ≥ 20 Hz inputs (fast membranes). Native STN
shows weaker locking at low frequencies (VS 0.6–0.8) because its own pacemaker competes with
the input, but locks 1:1 at 50 Hz and fires once every 3–6 cycles at 100 Hz.

**Chirp (Fig. 3).** Subthreshold impedance mirrors the sinusoid results: band-pass at theta
for STN and CA1, high-frequency band-pass for HH, low-pass for Purkinje.

**Pulse trains (Fig. 6).** See Section 5.

**Noise (Fig. 7).** Rates under the same relative drive (mean 1×, SD 0.4×) differ ~20-fold
(HH 3.4 Hz, CA1 5.2 Hz, STN 58 Hz, Purkinje 66 Hz). Reliability across frozen-noise trials is
high for all (R 0.75–0.87; jitter 0.5–1.3 ms), consistent with the precise spike timing for
fluctuating input reported in cortical neurons (Mainen & Sejnowski 1995). STAs separate
resonators (HH, Purkinje: brief depolarisation preceded by a hyperpolarising lobe) from the
integrator (CA1: slow monophasic rise), as predicted for type II vs type I dynamics
(Mato & Samengo 2008; Ermentrout et al. 2007).

**Synaptic-like input (Fig. 8).** Brief EPSC-shaped inputs need more than step rheobase:
HH, Purkinje and STN (native) respond reliably at 2×, CA1 only at 4× (low input resistance,
29 MΩ, and fast apparent τ shunt brief inputs). STN is the only cell that responds
sub-rheobase (P = 0.76 at 1×) because its pacemaker keeps it near threshold, but its timing is
the least precise (jitter 1.9 → 0.5 ms with amplitude) – it depends on the pacemaker phase at
which the input arrives. Held STN never responds: at −70 mV brief inputs cannot reach threshold.

---

## 4. Somatic vs dendritic injection

| | Purkinje (100 µm) | CA1 (300 µm, apical trunk) |
|---|---|---|
| Step rheobase | dendrite **lower** (0.72 vs 0.85 nA) | dendrite higher (0.39 vs 0.26 nA) |
| Single 0.5 ms pulse threshold | dendrite much higher (6.8 vs 1.9 nA) | dendrite **lower** (1.4 vs 2.8 nA) |
| Firing to dendritic steps | delayed (200–570 ms) Ca²⁺-spike bursts, CV 1.41 | prompt, 45–51 Hz, weaker adaptation (1.3) |
| Impedance | low-pass (both sites) | dendrite higher |Z| and higher fR (13.4 Hz) |

Purkinje dendrites are excitable on a slow time scale (sustained current recruits Ca²⁺ spikes)
but poorly coupled to the soma for brief inputs; CA1's thin trunk has high input impedance, so
brief dendritic pulses depolarise it more than the same current at the soma.

---

## 5. Implications for DBS waveform design (STN)

* Brief (0.5 ms) intracellular pulses drive the STN **soma** 1:1 up to 200 Hz even at 1.2×
  threshold, with no depolarisation block and no post-stimulus silencing (post-train rate
  5–20 Hz ≈ baseline 10 Hz). Slice studies instead report silencing of STN during
  high-frequency stimulation (Beurrier et al. 2001), and modelling shows axons can be driven at
  the stimulus rate while the soma is suppressed (McIntyre et al. 2004). Together this suggests
  that STN silencing during DBS arises from mechanisms absent from single-site intracellular
  injection – extracellular fields acting on axons, synaptic GABA release and ionic
  accumulation – rather than from the soma's intrinsic inability to follow.
* **Charge balance costs efficacy:** at 1.2× threshold, biphasic pulses roughly halve STN
  (held) following and fail completely in HH. Clinically, charge-balanced pulses protect
  tissue; this study quantifies the extra amplitude they need.
* **STN is the most "DBS-followable" cell; CA1 and HH are the least.** The ranking reflects
  recovery speed: STN's fast Kv3 repolarisation vs CA1's M/AHP currents and HH's slow
  recovery at 6.3 °C.
* **Hyperpolarising/low-frequency input is dangerous for STN:** any input that pushes it below
  ≈ −78 mV (inhibitory pauses, low-frequency sinusoids) triggers T-type rebound bursts –
  the pattern associated with parkinsonian beta activity. Its 5 Hz resonance makes it most
  responsive to low-frequency drive.

---

## 6. Agreement with published data

| Cell | Measure | This study | Published | Verdict |
|---|---|---|---|---|
| HH | Repetitive-firing onset | 6.5 µA/cm², 56 Hz | ≈ 6.3 µA/cm², abrupt onset (Rinzel & Miller 1980) | agrees |
| STN | Spontaneous firing | 9.6 Hz, CV 0.007 | ~10 Hz regular pacemaking (Gillies & Willshaw 2006) | agrees |
| STN | Rebound threshold | between −76 and −81 mV | −78 ± 3 mV (Bevan et al. 2000) | agrees |
| STN | Rebound requires T-type Ca²⁺ | abolished by CaT block | T-type mechanism of the model (Gillies & Willshaw 2006) | agrees |
| Purkinje | Dendritic Ca²⁺ spikes P-type | abolished by CaP block | Llinás & Sugimori (1980); Miyasho et al. (2001) | agrees |
| Purkinje | Kv1 (D-type) block | spontaneous 50 Hz Ca²⁺-spike bursting | Kv1 prevents spontaneous dendritic Ca²⁺ spikes / bursts (Khavandgar et al. 2005) | agrees |
| Purkinje | Firing at rest | silent | Purkinje cells fire spontaneously in vitro | **disagrees** (model limitation) |
| CA1 | Somatic fR | 10 Hz at −70 mV | 2–5 Hz at 33 °C (Hu et al. 2002); ~3–5 Hz at −65/−75 mV (Narayanan & Johnston 2007) | **model ~2× too high** |
| CA1 | Dendritic vs somatic fR | 13.4 vs 10 Hz | ~9–11 vs ~3–5 Hz (Narayanan & Johnston 2007) | trend agrees, gradient weaker |
| CA1 | Ih and M dependence of resonance | HCN block Q 1.39 → 1.08; KM block → 1.19 | ZD7288 abolishes H-, XE991 M-resonance (Hu et al. 2002) | agrees |
| All | Spike-time precision for fluctuating input | jitter 0.5–1.3 ms | sub-ms to ms precision (Mainen & Sejnowski 1995) | agrees |
| HH vs CA1 | STA shape | biphasic vs monophasic | type II vs type I (Mato & Samengo 2008) | agrees |

---

## 7. Limitations

1. **Intracellular injection is not DBS.** Clinical DBS uses extracellular, 60–90 µs pulses that
   preferentially activate axons; our 0.5 ms intracellular pulses probe only somatic excitability.
2. **Temperatures differ** (HH 6.3 °C, mammalian cells 34–37 °C), so absolute frequencies are not
   comparable between HH and the others; HH serves as a reference.
3. **Model choices:** the Purkinje model (Miyasho 2001) is silent at rest, unlike real Purkinje
   cells (a spontaneously active model was ~100× too slow for this batch); the CA1 model's
   resonance is ~2× faster than experimental values.
4. **Normalisation:** rheobase from a 1 s step is not the relevant threshold for brief inputs
   (pulses, EPSCs), which is why pulse trains were scaled to a separate single-pulse threshold;
   for STN the held-condition rheobase is a transient-spike threshold (sustained firing ≈ 3×).
5. **Channel blocks** used control holding currents and stimulus amplitudes; two blocked chirps
   (STN–Ih, Purkinje–KD) fired and were excluded from impedance analysis.
6. **Deterministic models:** trial-to-trial variability comes only from the added background
   noise (frozen-noise protocol), so reliability values depend on its chosen size (5 % of rheobase).
7. Single cell morphologies and parameter sets – no population variability.

---

## References

* Bevan MD, Wilson CJ, Bolam JP, Magill PJ (2000) Equilibrium potential of GABA_A current and implications for rebound burst firing in rat subthalamic neurons in vitro. *J Neurophysiol* 83:3169–3172.
* Beurrier C, Bioulac B, Audin J, Hammond C (2001) High-frequency stimulation produces a transient blockade of voltage-gated currents in subthalamic neurons. *J Neurophysiol* 85:1351–1356.
* Ermentrout GB, Galán RF, Urban NN (2007) Relating neural dynamics to neural coding. *Phys Rev Lett* 99:248103.
* Gillies A, Willshaw D (2006) Membrane channel interactions underlying rat subthalamic projection neuron rhythmic and bursting activity. *J Neurophysiol* 95:2352–2365.
* Hodgkin AL, Huxley AF (1952) A quantitative description of membrane current and its application to conduction and excitation in nerve. *J Physiol* 117:500–544.
* Hu H, Vervaeke K, Storm JF (2002) Two forms of electrical resonance at theta frequencies, generated by M-current, h-current and persistent Na⁺ current in rat hippocampal pyramidal cells. *J Physiol* 545:783–805.
* Khavandgar S, Walter JT, Sageser K, Khodakhah K (2005) Kv1 channels selectively prevent dendritic hyperexcitability in rat Purkinje cells. *J Physiol* 569:545–557.
* Llinás R, Sugimori M (1980) Electrophysiological properties of in vitro Purkinje cell dendrites in mammalian cerebellar slices. *J Physiol* 305:197–213.
* Mainen ZF, Sejnowski TJ (1995) Reliability of spike timing in neocortical neurons. *Science* 268:1503–1506.
* Mato G, Samengo I (2008) Type I and type II neuron models are selectively driven by differential stimulus features. *Neural Comput* 20:2418–2440 (arXiv:0801.3963).
* McIntyre CC, Grill WM, Sherman DL, Thakor NV (2004) Cellular effects of deep brain stimulation: model-based analysis of activation and inhibition. *J Neurophysiol* 91:1457–1469.
* Migliore R, Lupascu CA, Bologna LL, et al. (2018) The physiological variability of channel density in hippocampal CA1 pyramidal cells and interneurons explored using a unified data-driven modeling workflow. *PLoS Comput Biol* 14:e1006423.
* Miyasho T, Takagi H, Suzuki H, et al. (2001) Low-threshold potassium channels and a low-threshold calcium channel regulate Ca²⁺ spike firing in the dendrites of cerebellar Purkinje neurons: a modeling study. *Brain Res* 891:106–115.
* Narayanan R, Johnston D (2007) Long-term potentiation in rat hippocampal neurons is accompanied by spatially widespread changes in intrinsic oscillatory dynamics and excitability. *Neuron* 56:1061–1075.
* Rinzel J, Miller RN (1980) Numerical calculation of stable and unstable periodic solutions to the Hodgkin-Huxley equations. *Math Biosci* 49:27–59.
