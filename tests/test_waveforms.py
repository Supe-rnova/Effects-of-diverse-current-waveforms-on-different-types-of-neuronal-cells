"""Unit tests for the waveform generator (Step 5).   Run:  python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import DT  # noqa: E402
from waveforms import generator as g  # noqa: E402


def test_grid_and_length():
    w = g.step(0.1, 100, 500, 1000)
    assert len(w.t) == round(1000 / DT)
    assert np.allclose(np.diff(w.t), DT)
    assert w.t[0] == 0


def test_step_edges_exact():
    w = g.step(0.2, 100, 500, 1000)
    on = w.i > 0
    assert w.t[on][0] == pytest.approx(100)
    assert w.t[on][-1] == pytest.approx(600 - DT)
    assert w.charge_pC() == pytest.approx(0.2 * 500)


def test_off_grid_time_rejected():
    with pytest.raises(ValueError):
        g.step(0.1, 100.01, 500, 1000)


def test_hyperpolarising_requires_negative():
    with pytest.raises(ValueError):
        g.hyperpolarising_step(0.1, 100, 500, 1000)
    w = g.hyperpolarising_step(-0.2, 100, 500, 1000)
    assert w.params["release"] == 600 and w.i.min() == -0.2


def test_ramp_endpoints():
    w = g.ramp(0.0, 1.0, 100, 1000, 1200)
    act = w.i[(w.t >= 100) & (w.t < 1100)]
    assert act[0] == 0 and act[-1] == pytest.approx(1.0 - 1.0 / len(act))
    assert np.all(np.diff(act) > 0)


def test_sinusoid_frequency():
    w = g.sinusoid(0.1, 10, 0, 1000, 1000)
    spec = np.abs(np.fft.rfft(w.i))
    f = np.fft.rfftfreq(len(w.i), DT / 1000)
    assert f[np.argmax(spec)] == pytest.approx(10, abs=0.5)
    assert w.i.max() == pytest.approx(0.1, rel=1e-3)


@pytest.mark.parametrize("method", ["linear", "log"])
def test_chirp_sweep(method):
    w = g.chirp(0.05, 0.5, 50, 100, 10000, 10200, method=method)
    fi = g.chirp_inst_freq(w)
    assert fi[0] == pytest.approx(0.5) and fi[-1] == pytest.approx(50, rel=1e-3)


def test_monophasic_charge():
    w = g.pulse_train(1.0, 0.1, 130, 100, 1000, 1200, polarity="mono")
    assert w.params["n_pulses"] == 130
    assert w.charge_pC() == pytest.approx(130 * 1.0 * 0.1)
    # every pulse is exactly 4 samples (0.1 ms / 0.025 ms) with vertical edges
    assert int((w.i > 0).sum()) == 130 * 4


@pytest.mark.parametrize("freq", [5, 20, 60, 130, 200])
def test_biphasic_net_charge_zero(freq):
    w = g.pulse_train(1.0, 0.1, freq, 100, 1000, 1200, polarity="biphasic", interphase=0.05)
    assert abs(w.charge_pC()) < 1e-12
    assert w.i.max() == 1.0 and w.i.min() == -1.0


def test_pulse_must_fit_period():
    with pytest.raises(ValueError):
        g.pulse_train(1.0, 3.0, 200, 0, 100, 100, polarity="biphasic")   # 6 ms > 5 ms period


def test_pulse_width_multiple_of_dt():
    with pytest.raises(ValueError):
        g.pulse_train(1.0, 0.11, 100, 0, 100, 100)


def test_ou_statistics():
    w = g.ou_noise(0.1, 0.05, 5.0, 0, 20000, 20000, seed=1)
    assert w.i.mean() == pytest.approx(0.1, abs=0.005)
    assert w.i.std() == pytest.approx(0.05, rel=0.05)
    x = w.i - w.i.mean()
    lag = int(5.0 / DT)          # autocorrelation at lag tau should be ~ 1/e
    r = np.dot(x[:-lag], x[lag:]) / np.dot(x, x)
    assert r == pytest.approx(np.exp(-1), abs=0.05)


def test_seeds_reproducible_and_distinct():
    a = g.ou_noise(0, 0.05, 5, 0, 1000, 1000, seed=7)
    b = g.ou_noise(0, 0.05, 5, 0, 1000, 1000, seed=7)
    c = g.ou_noise(0, 0.05, 5, 0, 1000, 1000, seed=8)
    assert np.array_equal(a.i, b.i) and not np.array_equal(a.i, c.i)


def test_frozen_noise_shares_signal():
    a = g.frozen_noise_trial(0.1, 0.05, 5, 0, 1000, 1000, signal_seed=1, trial_seed=10, sigma_bg=0.005)
    b = g.frozen_noise_trial(0.1, 0.05, 5, 0, 1000, 1000, signal_seed=1, trial_seed=11, sigma_bg=0.005)
    assert not np.array_equal(a.i, b.i)
    assert np.corrcoef(a.i, b.i)[0, 1] > 0.98      # same signal, small independent background


@pytest.mark.parametrize("kind", ["alpha", "double_exp"])
def test_synaptic_peak(kind):
    w = g.synaptic(0.3, [100.0], 300, kind=kind, tau_rise=0.5, tau_decay=5.0)
    assert w.i.max() == pytest.approx(0.3, rel=1e-2)
    assert np.all(w.i[w.t < 100] == 0)


def test_poisson_rate():
    ev = g.poisson_times(20, 0, 100000, seed=3)
    assert len(ev) / 100 == pytest.approx(20, rel=0.05)


def test_make_registry():
    w = g.make("sinusoid", amp=0.1, freq=10, onset=100, dur=1000, total=1200)
    assert w.kind == "sinusoid"


def test_make_synaptic_with_kind_param():
    w = g.make("synaptic", peak=0.2, event_times=[50.0], total=200, kind="alpha", tau_rise=0.5, tau_decay=5.0)
    assert w.kind == "synaptic_alpha"
