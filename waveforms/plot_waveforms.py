"""
Step 5 - plot every waveform for visual checking.

Run from the project root:
    python waveforms/plot_waveforms.py
Output: results/figures/waveforms/all_waveforms.png and pulse_detail.png
"""
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import FIGURES_DIR, LOGS_DIR  # noqa: E402
from waveforms import generator as g  # noqa: E402

OUT = FIGURES_DIR / "waveforms"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = 1500
    examples = [
        ("1 Long rectangular step", g.step(0.2, 200, 1000, T)),
        ("2 Hyperpolarising step + release", g.hyperpolarising_step(-0.2, 200, 800, T)),
        ("3 Slow linear ramp", g.ramp(0.0, 0.3, 200, 1200, T)),
        ("4 Sinusoid (10 Hz, DC 0.05)", g.sinusoid(0.1, 10, 200, 1000, T, dc=0.05)),
        ("5 Chirp / ZAP (0.5-20 Hz, linear)", g.chirp(0.05, 0.5, 20, 200, 1200, T)),
        ("6a Pulse train, monophasic 20 Hz", g.pulse_train(0.5, 1.0, 20, 200, 1000, T)),
        ("6b Pulse train, biphasic 20 Hz", g.pulse_train(0.5, 1.0, 20, 200, 1000, T, polarity="biphasic", interphase=0.1)),
        ("7a OU noise (tau 5 ms)", g.ou_noise(0.05, 0.05, 5.0, 200, 1000, T, seed=1)),
        ("7b Frozen noise: 2 trials (same signal)", None),
        ("8 Synaptic-like (double exp, Poisson 20 Hz)",
         g.synaptic(0.2, g.poisson_times(20, 200, 1000, seed=2), T, tau_rise=0.5, tau_decay=5.0)),
    ]
    fig, axes = plt.subplots(5, 2, figsize=(14, 13), sharex=True)
    for ax, (title, w) in zip(axes.flat, examples):
        if w is None:
            for ts, col in ((10, "C0"), (11, "C3")):
                w2 = g.frozen_noise_trial(0.05, 0.05, 5.0, 200, 1000, T, signal_seed=1,
                                          trial_seed=ts, sigma_bg=0.01)
                ax.plot(w2.t, w2.i, lw=.5, color=col, alpha=.8, label=f"trial seed {ts}")
            ax.legend(fontsize=7, loc="upper right")
        else:
            ax.plot(w.t, w.i, lw=.6)
            ax.text(0.01, 0.92, f"charge {w.charge_pC():.2f} pC", transform=ax.transAxes, fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("nA")
        ax.axhline(0, color="k", lw=.3)
    for ax in axes[-1]:
        ax.set_xlabel("ms")
    fig.suptitle("Current waveforms (dt = %.3f ms)" % examples[0][1].dt, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "all_waveforms.png", dpi=120)

    # zoom on a few DBS-rate pulses: exact edges and charge balance
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.2))
    for a, pol in zip(ax, ("mono", "biphasic")):
        w = g.pulse_train(1.0, 0.1, 130, 0, 30, 30, polarity=pol, interphase=0.05)
        a.step(w.t, w.i, where="post", lw=1)
        a.plot(w.t, w.i, ".", ms=3)
        a.set(xlim=(0, 16), title=f"130 Hz, 0.1 ms, {pol}: net charge {w.charge_pC():+.4f} pC",
              xlabel="ms", ylabel="nA")
    fig.tight_layout()
    fig.savefig(OUT / "pulse_detail.png", dpi=120)
    msg = f"waveform figures written to {OUT}"
    print(msg)
    (LOGS_DIR / "step5_waveforms.log").write_text(msg + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        (LOGS_DIR / "step5_waveforms.log").write_text(tb)
        print(tb)
        sys.exit(1)
