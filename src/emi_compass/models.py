"""Equivalent-circuit models for EMI suppression parts.

All models are deliberately simple textbook equivalents: a real capacitor is a
series RLC, a parallel bank is the parallel combination of branch impedances.
They are meant for *reading* datasheet-style curves, not for sign-off accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class CapacitorModel:
    """A real capacitor as a series C + ESL + ESR."""

    c: float  # farads
    esl: float = 0.5e-9  # henries
    esr: float = 0.02  # ohms
    label: str = ""

    def impedance(self, f: np.ndarray) -> np.ndarray:
        w = TWO_PI * np.asarray(f, dtype=float)
        return self.esr + 1j * (w * self.esl - 1.0 / (w * self.c))

    @property
    def srf(self) -> float:
        """Series self-resonant frequency in Hz."""
        return 1.0 / (TWO_PI * np.sqrt(self.esl * self.c))


@dataclass(frozen=True)
class BeadModel:
    """A ferrite chip bead as R_dc in series with (L .parallel. R_loss), the whole
    thing shunted by a parasitic capacitance C_par.

    - Low frequency: ``jwL`` wins, so it looks like a small inductor (X dominates).
    - Above the R/X crossover (``wL ~= R_loss``): the magnetic-loss resistance
      ``R_loss`` dominates -- the bead "becomes a resistor" that turns noise into
      heat. ``R_loss`` is roughly the resistive plateau height.
    - Very high frequency: ``C_par`` bypasses the part and |Z| falls again.
    """

    l: float  # noqa: E741 -- L is the inductance symbol; henries (low-frequency inductance)
    r_loss: float  # ohms (parallel magnetic-loss resistance, ~ resistive plateau)
    r_dc: float = 0.05  # ohms (DC resistance)
    c_par: float = 1.0e-12  # farads (parasitic capacitance)
    label: str = ""

    def core_impedance(self, f: np.ndarray) -> np.ndarray:
        """R_dc in series with (L .parallel. R_loss), before the parasitic cap."""
        w = TWO_PI * np.asarray(f, dtype=float)
        z_lr = (1j * w * self.l * self.r_loss) / (self.r_loss + 1j * w * self.l)
        return self.r_dc + z_lr

    def impedance(self, f: np.ndarray) -> np.ndarray:
        w = TWO_PI * np.asarray(f, dtype=float)
        y = 1.0 / self.core_impedance(f) + 1j * w * self.c_par
        return 1.0 / y

    @property
    def crossover(self) -> float:
        """R/X 'wake-up' frequency in Hz where wL = R_loss (ignoring R_dc, C_par)."""
        return self.r_loss / (TWO_PI * self.l)


def ideal_cap_impedance(f: np.ndarray, c: float) -> np.ndarray:
    w = TWO_PI * np.asarray(f, dtype=float)
    return -1j / (w * c)


def parallel_impedance(*zs: np.ndarray) -> np.ndarray:
    inv = sum(1.0 / z for z in zs)
    return 1.0 / inv


def shunt_insertion_loss_db(z: np.ndarray, z0: float = 50.0) -> np.ndarray:
    """Insertion loss (positive dB) of a shunt impedance across a matched z0 line.

    S21 of a shunt element is 2Z / (2Z + Z0); insertion loss is -20*log10|S21|.
    """
    s21 = (2.0 * z) / (2.0 * z + z0)
    return -20.0 * np.log10(np.abs(s21))


def shunt_insertion_loss_db_sys(z: np.ndarray, r_sys: float) -> np.ndarray:
    """Insertion loss (positive dB) of a shunt impedance against a system
    impedance ``r_sys = Rsource || Rload``.

    IL = 20*log10|1 + r_sys / Z|. The part only attenuates relative to the
    impedance it competes against: a stiff (low r_sys) power rail gives far
    less dB than the 50-ohm bench, where r_sys = 50||50 = 25 ohm. That case
    reduces to ``shunt_insertion_loss_db(z, z0=50)``.
    """
    return 20.0 * np.log10(np.abs(1.0 + r_sys / np.asarray(z)))


def series_insertion_loss_db(z: np.ndarray, r_sys: float = 100.0) -> np.ndarray:
    """Insertion loss (positive dB) of a SERIES impedance z against a system
    impedance ``r_sys = Rsource + Rload``.

    IL = 20*log10|1 + z / r_sys|. This is the mirror image of a shunt element:
    a series part (a bead) works *better* against a low r_sys (a stiff power
    rail) and barely at all in the 50-ohm bench, where r_sys = 50 + 50 = 100 ohm.
    """
    return 20.0 * np.log10(np.abs(1.0 + np.asarray(z) / r_sys))


def series_shunt_gain_db(z_series: np.ndarray, z_shunt: np.ndarray) -> np.ndarray:
    """Voltage transfer |Vout/Vin| in dB of a series element feeding a shunt
    element, with the output taken across the shunt: H = Z_shunt / (Z_series +
    Z_shunt). Above 0 dB means the L-C pair amplifies -- the resonant peak that
    makes "just add a bead" backfire.
    """
    h = np.asarray(z_shunt) / (np.asarray(z_series) + np.asarray(z_shunt))
    return 20.0 * np.log10(np.abs(h))


def antiresonance_peak(
    f: np.ndarray, z_parallel: np.ndarray, srf_low: float, srf_high: float
) -> tuple[float, float]:
    """Locate the |Z| peak between two SRFs. Returns (frequency, |Z|)."""
    mask = (f > srf_low) & (f < srf_high)
    if not mask.any():
        raise ValueError("no frequency samples between the two SRFs")
    zmag = np.abs(z_parallel[mask])
    i = int(np.argmax(zmag))
    return float(f[mask][i]), float(zmag[i])


def peak_in_band(
    f: np.ndarray, y: np.ndarray, f_low: float, f_high: float
) -> tuple[float, float]:
    """Locate the max of a real curve y between f_low and f_high. Returns
    (frequency, value). Used to find the bead+capacitor resonance peak.
    """
    mask = (f > f_low) & (f < f_high)
    if not mask.any():
        raise ValueError("no frequency samples in the requested band")
    yb = np.asarray(y)[mask]
    i = int(np.argmax(yb))
    return float(f[mask][i]), float(yb[i])
