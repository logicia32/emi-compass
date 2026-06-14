import numpy as np
import pytest

from emi_compass.cli import parse_value
from emi_compass.models import (
    TWO_PI,
    BeadModel,
    CapacitorModel,
    LCFilterModel,
    antiresonance_peak,
    ideal_cap_impedance,
    insertion_loss_from_s21,
    parallel_impedance,
    peak_in_band,
    read_touchstone,
    s_series_element,
    series_insertion_loss_db,
    series_shunt_gain_db,
    series_z_from_s21,
    shunt_insertion_loss_db,
    shunt_insertion_loss_db_sys,
    write_touchstone,
)

MLCC = CapacitorModel(c=0.1e-6, esl=0.5e-9, esr=0.02)
BEAD = BeadModel(l=0.35e-6, r_loss=112.0, r_dc=0.05, c_par=1e-12)


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


def test_bead_crossover_matches_formula():
    # wL = R_loss  ->  f = R_loss / (2*pi*L)
    assert BEAD.crossover == pytest.approx(112.0 / (TWO_PI * 0.35e-6), rel=1e-9)


def test_bead_low_frequency_is_inductive():
    f = np.array([1e5])  # 100 kHz, far below the crossover
    z = BEAD.impedance(f)[0]
    assert abs(z.imag) > z.real  # X dominates
    assert abs(z) == pytest.approx(TWO_PI * f[0] * BEAD.l, rel=0.05)


def test_bead_is_resistive_above_crossover():
    f = np.array([100e6])  # above the ~51 MHz crossover
    z = BEAD.impedance(f)[0]
    assert z.real > abs(z.imag)  # R dominates -> the bead "woke up"


def test_bead_z_at_100mhz_is_about_100ohm():
    z100 = abs(BEAD.impedance(np.array([100e6]))[0])
    assert z100 == pytest.approx(100.0, abs=12.0)  # "100 ohm @ 100 MHz" class


def test_bead_parasitic_cap_rolls_off_top_end():
    # parasitic C bypasses the part: |Z| falls back down at the very top
    z_mid = abs(BEAD.impedance(np.array([3e8]))[0])
    z_top = abs(BEAD.impedance(np.array([3e9]))[0])
    assert z_top < z_mid


def test_series_insertion_loss_known_values():
    z = np.array([100.0 + 0j])
    assert series_insertion_loss_db(z, 100.0)[0] == pytest.approx(20 * np.log10(2.0), rel=1e-9)
    assert series_insertion_loss_db(z, 2.0)[0] == pytest.approx(20 * np.log10(51.0), rel=1e-9)


def test_series_insertion_loss_bigger_for_lower_rsys():
    # opposite of a shunt part: a series bead works better against a stiff rail
    z = np.array([100.0 + 0j])
    assert series_insertion_loss_db(z, 2.0)[0] > series_insertion_loss_db(z, 100.0)[0]


def test_bead_cap_resonance_peaks_then_damps():
    f = np.logspace(4, 9, 6001)
    zb = BEAD.impedance(f)
    f_res = 1.0 / (TWO_PI * np.sqrt(BEAD.l * 10e-6))
    # low-ESR MLCC: the LC pair peaks above 0 dB (noise amplified)
    low = CapacitorModel(c=10e-6, esl=0.5e-9, esr=0.005)
    _, gpk_low = peak_in_band(f, series_shunt_gain_db(zb, low.impedance(f)), f_res / 5, f_res * 5)
    assert gpk_low > 6.0
    # damped by a lossy (higher-ESR) cap: the peak is gone
    damped = CapacitorModel(c=10e-6, esl=0.5e-9, esr=0.6)
    _, gpk_damped = peak_in_band(f, series_shunt_gain_db(zb, damped.impedance(f)), f_res / 5, f_res * 5)
    assert gpk_damped < 1.0


def test_parse_value_suffixes():
    assert parse_value("0.1u") == pytest.approx(1e-7)
    assert parse_value("0.5n") == pytest.approx(5e-10)
    assert parse_value("50p") == pytest.approx(5e-11)
    assert parse_value("3G") == pytest.approx(3e9)
    assert parse_value("0.02") == pytest.approx(0.02)


# --- LC (pi) filter, article 3 -------------------------------------------

LCF = LCFilterModel(l=22e-6, c=100e-6)  # the case-study choke + bulk cap


def test_lc_filter_fc_matches_formula():
    # fc = 1/(2*pi*sqrt(L*C)); 22uH + 100uF -> ~3.39 kHz
    assert LCF.fc == pytest.approx(1.0 / (TWO_PI * np.sqrt(22e-6 * 100e-6)), rel=1e-9)
    assert LCF.fc == pytest.approx(3.39e3, rel=0.02)


def test_lc_filter_dc_gain_is_unity():
    att = LCF.attenuation_db(np.array([1.0]))[0]  # ~DC, far below fc
    assert att == pytest.approx(0.0, abs=0.2)  # passes DC through


def test_lc_filter_ideal_rolls_off_40db_per_decade():
    # ideal shunt cap (esr=0): two decades above fc -> ~40 dB/dec
    f = np.array([1e5, 1e6])
    att = LCF.attenuation_db(f)
    assert (att[1] - att[0]) == pytest.approx(40.0, abs=1.0)


def test_lc_filter_reaches_about_100db_at_1mhz():
    # 1 MHz is ~300x above fc -> ~99 dB for an ideal 2nd-order section
    att = LCF.attenuation_db(np.array([1e6]))[0]
    assert att == pytest.approx(99.0, abs=3.0)


def test_lc_filter_underdamped_peaks_above_zero():
    # low total loss -> the LC amplifies near fc (gain > 0 dB)
    f = np.logspace(2, 5, 4001)
    low_loss = LCFilterModel(l=22e-6, c=100e-6, esr=0.005, dcr=0.05, r_src=0.1)
    _, gpk = peak_in_band(f, low_loss.gain_db(f), low_loss.fc / 4, low_loss.fc * 4)
    assert gpk > 5.0


def test_lc_filter_damping_removes_peak():
    f = np.logspace(2, 5, 4001)
    damped = LCFilterModel(l=22e-6, c=100e-6, esr=0.6, dcr=0.05, r_src=0.1)
    _, gpk = peak_in_band(f, damped.gain_db(f), damped.fc / 4, damped.fc * 4)
    assert gpk < 1.0


def test_lc_filter_esr_flattens_stopband():
    # a real cap ESR limits the deep stop-band: the slope drops below 40 dB/dec
    # (this is exactly why a pi filter needs a second, low-ESR cap)
    f = np.array([1e5, 1e6])
    leaky = LCFilterModel(l=22e-6, c=100e-6, esr=0.5)
    att = leaky.attenuation_db(f)
    assert (att[1] - att[0]) < 30.0


# --- S-parameters / Touchstone, article 3 --------------------------------

def test_series_s_roundtrip():
    z = np.array([0.5 + 0j, 10.0 + 30j, 100.0 - 20j])
    _, s21 = s_series_element(z, z0=50.0)
    np.testing.assert_allclose(series_z_from_s21(s21, 50.0), z, rtol=1e-9)


def test_series_open_has_zero_transmission():
    # a huge series impedance blocks the line: S21 -> 0, IL large
    _, s21 = s_series_element(np.array([1e9 + 0j]), z0=50.0)
    assert insertion_loss_from_s21(s21)[0] > 60.0


def test_series_short_passes_through():
    # a tiny series impedance is invisible: S21 -> 1, IL ~ 0
    _, s21 = s_series_element(np.array([1e-6 + 0j]), z0=50.0)
    assert insertion_loss_from_s21(s21)[0] == pytest.approx(0.0, abs=1e-3)


def test_touchstone_write_read_roundtrip(tmp_path):
    f = np.logspace(5, 9, 50)
    z = BEAD.impedance(f)
    s11, s21 = s_series_element(z, z0=50.0)
    p = tmp_path / "bead.s2p"
    write_touchstone(p, f, s11, s21, z0=50.0, comment="synthetic bead")
    f2, s = read_touchstone(p)
    np.testing.assert_allclose(f2, f, rtol=1e-5)
    np.testing.assert_allclose(s["s21"], s21, rtol=1e-4, atol=1e-6)
    # and the impedance recovered through the .s2p matches the model
    np.testing.assert_allclose(series_z_from_s21(s["s21"], s["z0"]), z, rtol=1e-3)


def test_touchstone_reads_mhz_and_ri(tmp_path):
    # a hand-written file in MHz / real-imag format parses to the same place
    p = tmp_path / "ri.s2p"
    p.write_text(
        "! demo\n# MHz S RI R 50\n"
        "100 0.5 0.0 0.5 0.0 0.5 0.0 0.5 0.0\n"
        "200 0.0 0.5 0.0 0.5 0.0 0.5 0.0 0.5\n"
    )
    f, s = read_touchstone(p)
    assert f[0] == pytest.approx(100e6)
    assert f[1] == pytest.approx(200e6)
    assert s["s11"][0] == pytest.approx(0.5 + 0j)
    assert s["s21"][1] == pytest.approx(0.5j)
