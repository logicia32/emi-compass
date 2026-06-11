import numpy as np
import pytest

from emi_compass.cli import parse_value
from emi_compass.models import (
    CapacitorModel,
    antiresonance_peak,
    ideal_cap_impedance,
    parallel_impedance,
    shunt_insertion_loss_db,
    shunt_insertion_loss_db_sys,
)

MLCC = CapacitorModel(c=0.1e-6, esl=0.5e-9, esr=0.02)


def test_srf_matches_formula():
    # 1 / (2*pi*sqrt(0.5n * 0.1u)) = 22.5 MHz
    assert MLCC.srf == pytest.approx(22.5e6, rel=0.01)


def test_low_frequency_is_capacitive():
    f = np.array([1e4])
    expected = 1.0 / (2 * np.pi * f[0] * MLCC.c)
    assert abs(MLCC.impedance(f)[0]) == pytest.approx(expected, rel=1e-3)


def test_high_frequency_is_inductive():
    f = np.array([3e9])
    expected = 2 * np.pi * f[0] * MLCC.esl
    assert abs(MLCC.impedance(f)[0]) == pytest.approx(expected, rel=1e-2)


def test_valley_depth_equals_esr():
    f = np.array([MLCC.srf])
    assert abs(MLCC.impedance(f)[0]) == pytest.approx(MLCC.esr, rel=1e-6)


def test_right_slope_independent_of_capacitance():
    f = np.array([1e9])
    mags = [abs(CapacitorModel(c=c, esl=0.5e-9, esr=0.02).impedance(f)[0])
            for c in (0.01e-6, 0.1e-6, 1e-6)]
    assert max(mags) / min(mags) < 1.01


def test_three_terminal_wins_at_100mhz():
    f = np.array([100e6])
    z2t = abs(MLCC.impedance(f)[0])
    z3t = abs(CapacitorModel(c=0.1e-6, esl=0.05e-9, esr=0.02).impedance(f)[0])
    assert z3t < z2t / 5


def test_antiresonance_peak_between_srfs():
    small = CapacitorModel(c=0.01e-6, esl=0.5e-9, esr=0.02)
    f = np.logspace(4, 9.5, 6001)
    zp = parallel_impedance(MLCC.impedance(f), small.impedance(f))
    fpk, zpk = antiresonance_peak(f, zp, MLCC.srf, small.srf)
    assert MLCC.srf < fpk < small.srf
    # the hill rises well above both ESRs
    assert zpk > 5 * MLCC.esr


def test_ideal_cap_keeps_falling():
    f = np.logspace(4, 9, 100)
    mag = np.abs(ideal_cap_impedance(f, 0.1e-6))
    assert np.all(np.diff(mag) < 0)


def test_insertion_loss_bigger_for_lower_z():
    z_low = np.array([0.1 + 0j])
    z_high = np.array([10.0 + 0j])
    assert shunt_insertion_loss_db(z_low)[0] > shunt_insertion_loss_db(z_high)[0]


def test_insertion_loss_known_value():
    # shunt 0.25 Ohm in 50 Ohm system: S21 = 0.5/50.5 -> ~40 dB
    il = shunt_insertion_loss_db(np.array([0.25 + 0j]))[0]
    assert il == pytest.approx(20 * np.log10(50.5 / 0.5), rel=1e-6)


def test_il_sys_reduces_to_50ohm_bench():
    z = np.array([0.25 + 0j, 1.0 + 2j, 5.0 - 3j])
    np.testing.assert_allclose(
        shunt_insertion_loss_db_sys(z, 25.0), shunt_insertion_loss_db(z, 50.0), rtol=1e-9
    )


def test_il_shrinks_on_low_impedance_rail():
    # 0.3 ohm cap: ~40 dB in the 50-ohm bench, far less against a 1-ohm rail
    z = np.array([0.3 + 0j])
    il_bench = shunt_insertion_loss_db_sys(z, 25.0)[0]
    il_rail = shunt_insertion_loss_db_sys(z, 1.0)[0]
    assert il_bench > 35
    assert il_rail < 15
    assert il_bench - il_rail > 20


def test_parse_value_suffixes():
    assert parse_value("0.1u") == pytest.approx(1e-7)
    assert parse_value("0.5n") == pytest.approx(5e-10)
    assert parse_value("50p") == pytest.approx(5e-11)
    assert parse_value("3G") == pytest.approx(3e9)
    assert parse_value("0.02") == pytest.approx(0.02)
