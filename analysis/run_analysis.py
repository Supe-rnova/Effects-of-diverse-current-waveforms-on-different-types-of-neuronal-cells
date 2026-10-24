"""
Step 8 - run the analysis over every batch result.

    python analysis/run_analysis.py              # all models
    python analysis/run_analysis.py HH STN       # subset

Outputs (results/data/analysis/):
    metrics_runs.csv     one row per run: identifiers + every per-run metric
    impedance.npz        |Z(f)| and phase for every chirp run
    sta.npz              spike-triggered averages (noise_var runs)
    summary.json         per model/condition/block/site: f-I, sag/rebound, ramp thresholds,
                         phase locking, resonance, pulse fidelity, ISI CV, reliability,
                         spike probability/jitter - with 95 % CIs where trials exist
Log: results/logs/step8_analysis.log
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import DATA_DIR, LOGS_DIR  # noqa: E402

OUT_DIR = DATA_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = LOGS_DIR / "step8_analysis.log"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def key_of(t):
    return f"{t['model']}|{t['condition']}|{t['block'] or 'ctrl'}|{t['site']}"


def main(models=None):
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN slices for silent cells
    import pandas as pd
    from analysis import metrics as M
    from simulations.injection import load_result

    LOG.write_text("")
    tasks = json.loads((DATA_DIR / "tasks.json").read_text())
    if models:
        tasks = [t for t in tasks if t["model"] in models]
    char = json.loads((DATA_DIR / "characterisation.json").read_text())
    log(f"analysing {len(tasks)} runs ...")
    t0 = time.perf_counter()

    rows, imp, groups = [], {}, defaultdict(list)
    n_err = 0
    for k, t in enumerate(tasks):
        f = DATA_DIR / "runs" / f"{t['id']}.npz"
        if not f.exists():
            log(f"  missing result: {t['id']}")
            continue
        try:
            res = load_result(f)
            base = {"id": t["id"], "model": t["model"], "condition": t["condition"],
                    "block": t["block"] or "ctrl", "site": t["site"], "protocol": t["protocol"],
                    "level": json.dumps(t["level"]), "trial": t["trial"]}
            if t["protocol"] == "chirp":
                m, (fz, Z, ph) = M.m_chirp(res)
                imp[t["id"]] = np.vstack([fz, Z, ph])
            else:
                m = M.PER_RUN[t["protocol"]](res)
            rows.append({**base, **{a: (json.dumps(b) if isinstance(b, list) else b) for a, b in m.items()}})
            if t["protocol"] in ("noise_var", "noise_rel", "syn"):
                groups[(key_of(t), t["protocol"], json.dumps(t["level"]))].append(res)
        except Exception:
            n_err += 1
            log(f"  ERROR {t['id']}:\n{traceback.format_exc()}")
        if (k + 1) % 100 == 0:
            log(f"  {k + 1}/{len(tasks)} runs ({time.perf_counter() - t0:.0f} s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "metrics_runs.csv", index=False)
    np.savez_compressed(OUT_DIR / "impedance.npz", **{k.replace("/", "__"): v for k, v in imp.items()})

    # --------------------------------------------------------------- across-trial metrics
    summary = defaultdict(dict)
    sta_out = {}
    for (key, proto, level), runs in groups.items():
        trains = [M.spikes_of(r) for r in runs]
        if proto == "noise_rel":
            p = runs[0]["meta"]["params"]
            t_a, t_b = p["onset"] + 100, p["onset"] + p["dur"]
            rel = M.schreiber_reliability(trains, t_a, t_b)
            # bootstrap CI over trials
            rng = np.random.default_rng(0)
            boots = [M.schreiber_reliability([trains[i] for i in rng.integers(0, len(trains), len(trains))], t_a, t_b)
                     for _ in range(100)]
            jit, part, nev = M.event_precision(trains, t_a, t_b)
            summary[key]["reliability"] = {"schreiber_R": rel, "ci95": [float(np.nanpercentile(boots, 2.5)),
                                                                       float(np.nanpercentile(boots, 97.5))],
                                           "event_jitter_ms": jit, "event_participation": part,
                                           "n_events": nev, "n_trials": len(trains),
                                           "rate_Hz": M.mean_ci([1000 * len(x[(x >= t_a) & (x < t_b)]) / (t_b - t_a)
                                                                 for x in trains])}
        elif proto == "noise_var":
            p = runs[0]["meta"]["params"]
            t_a, t_b = p["onset"] + 100, p["onset"] + p["dur"]
            cvs, rates = [], []
            for x in trains:
                s = x[(x >= t_a) & (x < t_b)]
                rates.append(1000 * len(s) / (t_b - t_a))
                if len(s) >= 3:
                    isi = np.diff(s)
                    cvs.append(isi.std() / isi.mean())
            lag, sta, nsp = M.spike_triggered_average(runs)
            if sta is not None:
                sta_out[key] = np.vstack([lag, sta])
            summary[key]["variability"] = {"isi_cv": M.mean_ci(cvs), "rate_Hz": M.mean_ci(rates),
                                           "n_seeds": len(runs), "n_spikes_sta": nsp,
                                           "sta_peak_nA": float(sta.max()) if sta is not None else None,
                                           "sta_peak_lag_ms": float(lag[np.argmax(sta)]) if sta is not None else None}
        elif proto == "syn":
            lat = np.array([M.m_syn(r)["latencies_ms"] for r in runs], float)    # trials x events
            resp = ~np.isnan(lat)
            k_resp, n_tot = int(resp.sum()), resp.size
            per_event_jit = [np.nanstd(lat[:, e]) for e in range(lat.shape[1]) if resp[:, e].sum() >= 3]
            summary[key].setdefault("synaptic", {})[level] = {
                "spike_prob": k_resp / n_tot, "ci95": M.wilson_ci(k_resp, n_tot),
                "latency_ms": float(np.nanmean(lat)) if k_resp else None,
                "jitter_ms": float(np.mean(per_event_jit)) if per_event_jit else None,
                "n_trials": len(runs), "n_events": int(lat.shape[1])}

    # --------------------------------------------------------------- per-run summaries
    for key, g in df.groupby(df["model"] + "|" + df["condition"] + "|" + df["block"] + "|" + df["site"]):
        s = summary[key]
        st = g[g.protocol == "step"].sort_values("level", key=lambda c: c.map(json.loads))
        if len(st):
            s["fI"] = {"x_rheobase": st["level"].map(json.loads).tolist(), "rate_Hz": st["rate_Hz"].tolist(),
                       "rate_ss_Hz": st["rate_ss_Hz"].tolist(), "latency_ms": st["latency_ms"].tolist(),
                       "adaptation_index": st.get("adaptation_index", st["rate_Hz"] * np.nan).tolist(),
                       "depol_block": st["depol_block"].tolist(),
                       "n_dend_ca": st["n_dend_ca"].tolist()}
        hy = g[g.protocol == "hyper"].sort_values("level", key=lambda c: c.map(json.loads))
        if len(hy):
            s["hyper"] = {c: hy[c].tolist() for c in ("level", "sag_ratio", "sag_mV", "rebound_spikes",
                                                      "rebound_latency_ms", "rebound_max_rate_Hz", "v_peak_mV")}
            s["hyper"]["level"] = [json.loads(x) for x in s["hyper"]["level"]]
        rp = g[g.protocol == "ramp"]
        if len(rp):
            s["ramp"] = {json.loads(r.level): {"threshold_current_nA": r.threshold_current_nA,
                                               "n_spikes": r.n_spikes, "slope_nA_per_s": r.slope_nA_per_s}
                         for r in rp.itertuples()}
        sn = g[g.protocol == "sine"]
        if len(sn):
            s["sine"] = [{"freq_Hz": json.loads(r.level)[0], "amp_x": json.loads(r.level)[1],
                          "vector_strength": r.vector_strength, "rayleigh_p": r.rayleigh_p,
                          "rate_Hz": r.rate_Hz, "spikes_per_cycle": r.spikes_per_cycle,
                          "cycles_with_spike": r.cycles_with_spike, "v_p2p_mV": r.v_p2p_mV}
                         for r in sn.itertuples()]
        ch = g[g.protocol == "chirp"]
        if len(ch):
            r = ch.iloc[0]
            s["resonance"] = {c: r[c] for c in ("fR_Hz", "Q", "Z_low_MOhm", "Z_max_MOhm", "f3dB_Hz",
                                                 "phase_zero_crossing_Hz", "valid_subthreshold", "n_spikes")}
        pu = g[g.protocol == "pulse"]
        if len(pu):
            s["pulse"] = [{"freq_Hz": json.loads(r.level)[0], "polarity": json.loads(r.level)[1],
                           "amp_x_thr": json.loads(r.level)[2], "fidelity": r.fidelity,
                           "fidelity_first10": r.fidelity_first10, "fidelity_2nd_half": r.fidelity_2nd_half,
                           "rate_Hz": r.rate_Hz, "rate_pre_Hz": r.rate_pre_Hz, "rate_post_Hz": r.rate_post_Hz}
                          for r in pu.itertuples()]
        m, c = key.split("|")[0], key.split("|")[1]
        cc = char[m]["conditions"].get(c, {})
        s["cell"] = {"Rin_MOhm": cc.get("Rin_MOhm"), "tau_m_ms": cc.get("tau_m_ms"),
                     "baseline_rate_Hz": cc.get("baseline", {}).get("rate_Hz"),
                     "rheobase_soma_nA": char[m].get("amplitude_unit_nA"),
                     "rheobase_dend_nA": char[m].get("amplitude_unit_dend_nA")}

    np.savez_compressed(OUT_DIR / "sta.npz", **{k.replace("|", "__"): v for k, v in sta_out.items()})
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1, default=_jsonable))
    log(f"done: {len(df)} runs analysed, {n_err} errors, {time.perf_counter() - t0:.0f} s")
    log(f"outputs in {OUT_DIR}")
    _print_headlines(summary)
    log("ANALYSIS COMPLETE" if n_err == 0 else "ANALYSIS FINISHED WITH ERRORS - see log")


def _print_headlines(summary):
    log("\nheadline numbers (control, soma):")
    for key in sorted(summary):
        m, c, b, site = key.split("|")
        if b != "ctrl":
            continue
        s = summary[key]
        parts = []
        if "fI" in s:
            parts.append(f"f@2x={s['fI']['rate_Hz'][s['fI']['x_rheobase'].index(2.0)]:.0f}Hz"
                         if 2.0 in s["fI"]["x_rheobase"] else "")
        if "hyper" in s:
            parts.append(f"rebound@-2x={s['hyper']['rebound_spikes'][s['hyper']['level'].index(-2.0)]}")
        if "resonance" in s:
            r = s["resonance"]
            parts.append(f"fR={r['fR_Hz']:.1f}Hz Q={r['Q']:.2f}{'' if r['valid_subthreshold'] else ' (SPIKED!)'}")
        if "pulse" in s:
            f130 = [p["fidelity"] for p in s["pulse"] if p["freq_Hz"] == 130 and p["polarity"] == "mono"
                    and p["amp_x_thr"] == 2.0]
            if f130:
                parts.append(f"fid130={f130[0]:.2f}")
        if "variability" in s:
            parts.append(f"CV={s['variability']['isi_cv'][0]:.2f}")
        if "reliability" in s:
            parts.append(f"R={s['reliability']['schreiber_R']:.2f}")
        log(f"  {m:9s} {c:7s} {site:5s} " + "  ".join(p for p in parts if p))


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return None if np.isnan(o) else o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    raise TypeError(type(o))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
