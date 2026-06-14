"""Equivalent-circuit models for EMI suppression parts.

All models are deliberately simple textbook equivalents: a real capacitor is a
series RLC, a parallel bank is the parallel combination of branch impedances.
They are meant for *reading* datasheet-style curves, not for sign-off accuracy.
"""

from __future__ import annotations

import pathlib
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


@dataclass(frozen=True)
class LCFilterModel:
    """A 2nd-order LC low-pass: a series choke (L + DCR) feeding a shunt
    capacitor (C + ESR), driven from a source resistance into a load. Output is
    taken across the shunt cap.

    This is the working section of a pi filter: in practice the input bulk
    electrolytic keeps the source stiff (small ``r_src``) and a small output
    MLCC cleans up the very top (not modeled here).

    - With an *ideal* shunt cap (``esr=0``) the stop-band falls at 40 dB/dec.
    - A real ESR eventually flattens that slope -- which is exactly why a pi
      filter adds a second, low-ESR cap to carry the high end.
    - The peak at ``fc`` is set by how the total series loss
      (``r_src + dcr + esr``) compares to the characteristic ``sqrt(L/C)``.
    """

    l: float  # noqa: E741 -- choke inductance, henries
    c: float  # shunt capacitance, farads
    dcr: float = 0.05  # choke DC resistance, ohms
    esr: float = 0.0  # shunt-cap ESR, ohms (0 = ideal -> clean 40 dB/dec)
    r_src: float = 0.1  # source resistance (a stiff rail), ohms
    r_load: float = 1.0e9  # load resistance, ohms (default ~ open circuit)

    def _z_series(self, f: np.ndarray) -> np.ndarray:
        w = TWO_PI * np.asarray(f, dtype=float)
        return self.dcr + 1j * w * self.l

    def _z_shunt(self, f: np.ndarray) -> np.ndarray:
        w = TWO_PI * np.asarray(f, dtype=float)
        z_c = self.esr + 1.0 / (1j * w * self.c)
        return (z_c * self.r_load) / (z_c + self.r_load)

    def transfer(self, f: np.ndarray) -> np.ndarray:
        """Voltage transfer Vout/Vin (complex)."""
        zs = self._z_series(f)
        zsh = self._z_shunt(f)
        return zsh / (self.r_src + zs + zsh)

    def gain_db(self, f: np.ndarray) -> np.ndarray:
        """|Vout/Vin| in dB. Above 0 dB near fc means the LC peaks (worse)."""
        return 20.0 * np.log10(np.abs(self.transfer(f)))

    def attenuation_db(self, f: np.ndarray) -> np.ndarray:
        """Attenuation in positive dB (= -gain_db). Negative where the LC peaks."""
        return -self.gain_db(f)

    @property
    def fc(self) -> float:
        """LC corner 1/(2*pi*sqrt(L*C)) in Hz."""
        return 1.0 / (TWO_PI * np.sqrt(self.l * self.c))

    @property
    def z0(self) -> float:
        """Characteristic impedance sqrt(L/C); sets the peak height vs the loss."""
        return float(np.sqrt(self.l / self.c))


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


# ---------------------------------------------------------------------------
# S-parameters / Touchstone (.s2p): the "real measured data" of article 3.
# ---------------------------------------------------------------------------

def s_series_element(z: np.ndarray, z0: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """S11, S21 of a single SERIES impedance ``z`` in a ``z0`` system.

    ``S21 = 2*z0 / (2*z0 + z)`` and ``S11 = z / (z + 2*z0)``. A series element
    is reciprocal and symmetric, so ``S12 = S21`` and ``S22 = S11``.
    """
    z = np.asarray(z, dtype=complex)
    s21 = (2.0 * z0) / (2.0 * z0 + z)
    s11 = z / (z + 2.0 * z0)
    return s11, s21


def series_z_from_s21(s21: np.ndarray, z0: float = 50.0) -> np.ndarray:
    """Recover a series impedance from its S21 (inverse of ``s_series_element``).

    ``z = 2*z0*(1 - S21)/S21``. Lets you read a vendor .s2p of a series part
    (a bead) back into the |Z| curve the rest of emi-compass speaks.
    """
    s21 = np.asarray(s21, dtype=complex)
    return 2.0 * z0 * (1.0 - s21) / s21


def insertion_loss_from_s21(s21: np.ndarray) -> np.ndarray:
    """Insertion loss in positive dB from S21: ``-20*log10|S21|``."""
    return -20.0 * np.log10(np.abs(np.asarray(s21, dtype=complex)))


_TOUCHSTONE_UNITS = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}


def read_touchstone(path: str | pathlib.Path) -> tuple[np.ndarray, dict]:
    """Read a 2-port Touchstone (.s2p) file.

    Returns ``(f_hz, S)`` where ``S`` is a dict with complex arrays ``s11``,
    ``s21``, ``s12``, ``s22`` and the reference impedance ``z0``. Supports the
    MA / RI / DB data formats and Hz/kHz/MHz/GHz frequency units; ``!`` starts a
    comment and ``#`` is the option line.
    """
    text = pathlib.Path(path).read_text()
    unit_scale, fmt, z0 = 1.0, "ma", 50.0
    rows: list[list[float]] = []
    for raw in text.splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            toks = line[1:].lower().split()
            for i, t in enumerate(toks):
                if t in _TOUCHSTONE_UNITS:
                    unit_scale = _TOUCHSTONE_UNITS[t]
                elif t in ("ma", "ri", "db"):
                    fmt = t
                elif t == "r" and i + 1 < len(toks):
                    z0 = float(toks[i + 1])
            continue
        rows.append([float(x) for x in line.split()])
    data = np.array(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] < 9:
        raise ValueError("not a 2-port Touchstone file (need 9 columns)")
    f = data[:, 0] * unit_scale

    def to_complex(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if fmt == "ri":
            return a + 1j * b
        if fmt == "db":
            return 10.0 ** (a / 20.0) * np.exp(1j * np.deg2rad(b))
        return a * np.exp(1j * np.deg2rad(b))  # ma

    s = {
        "s11": to_complex(data[:, 1], data[:, 2]),
        "s21": to_complex(data[:, 3], data[:, 4]),
        "s12": to_complex(data[:, 5], data[:, 6]),
        "s22": to_complex(data[:, 7], data[:, 8]),
        "z0": z0,
    }
    return f, s


def write_touchstone(
    path: str | pathlib.Path,
    f: np.ndarray,
    s11: np.ndarray,
    s21: np.ndarray,
    s12: np.ndarray | None = None,
    s22: np.ndarray | None = None,
    z0: float = 50.0,
    comment: str | None = None,
) -> None:
    """Write a 2-port .s2p in MA (magnitude / angle-deg) format, frequency in Hz.

    ``s12`` and ``s22`` default to ``s21`` and ``s11`` (a reciprocal, symmetric
    element such as a series bead).
    """
    s11 = np.asarray(s11, dtype=complex)
    s21 = np.asarray(s21, dtype=complex)
    s12 = s21 if s12 is None else np.asarray(s12, dtype=complex)
    s22 = s11 if s22 is None else np.asarray(s22, dtype=complex)
    lines: list[str] = []
    if comment:
        lines.extend(f"! {c}" for c in comment.splitlines())
    lines.append(f"# Hz S MA R {int(z0)}")

    def ma(z: complex) -> str:
        return f"{abs(z):.6e} {np.degrees(np.angle(z)):.4f}"

    for i in range(len(f)):
        lines.append(
            f"{f[i]:.6e} {ma(s11[i])} {ma(s21[i])} {ma(s12[i])} {ma(s22[i])}"
        )
    pathlib.Path(path).write_text("\n".join(lines) + "\n")
