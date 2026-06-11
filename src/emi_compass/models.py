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
