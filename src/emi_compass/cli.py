"""emi-compass command line interface.

Currently ships the `cap` subcommand (article 1 of the series). Bead and
filter checks arrive with articles 2 and 3.
"""

from __future__ import annotations

import argparse

import numpy as np

from .models import (
    TWO_PI,
    BeadModel,
    CapacitorModel,
    antiresonance_peak,
    parallel_impedance,
    peak_in_band,
    series_shunt_gain_db,
)

_SUFFIX = {
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "µ": 1e-6,
    "m": 1e-3,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
}


def parse_value(text: str) -> float:
    """Parse '0.1u', '0.5n', '22' etc. into a float."""
    text = text.strip()
    if text and text[-1] in _SUFFIX:
        return float(text[:-1]) * _SUFFIX[text[-1]]
    return float(text)


def _eng(value: float, unit: str) -> str:
    for factor, prefix in ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "u"), (1e-9, "n"), (1e-12, "p")):
        if abs(value) >= factor:
            return f"{value / factor:.3g} {prefix}{unit}"
    return f"{value:.3g} {unit}"


def cmd_cap(args: argparse.Namespace) -> int:
    f = np.logspace(np.log10(parse_value(args.fmin)), np.log10(parse_value(args.fmax)), 4001)

    cap1 = CapacitorModel(parse_value(args.c), parse_value(args.esl), parse_value(args.esr))
    print(f"C1: C={_eng(cap1.c, 'F')}, ESL={_eng(cap1.esl, 'H')}, ESR={_eng(cap1.esr, 'Ohm')}")
    print(f"    SRF = {_eng(cap1.srf, 'Hz')}  (|Z| there ~= ESR)")

    curves = [(np.abs(cap1.impedance(f)), "C1")]

    if args.c2 is not None:
        esl2 = parse_value(args.esl2) if args.esl2 else cap1.esl
        esr2 = parse_value(args.esr2) if args.esr2 else cap1.esr
        cap2 = CapacitorModel(parse_value(args.c2), esl2, esr2)
        print(f"C2: C={_eng(cap2.c, 'F')}, ESL={_eng(cap2.esl, 'H')}, ESR={_eng(cap2.esr, 'Ohm')}")
        print(f"    SRF = {_eng(cap2.srf, 'Hz')}")
        z_par = parallel_impedance(cap1.impedance(f), cap2.impedance(f))
        curves.append((np.abs(cap2.impedance(f)), "C2"))
        curves.append((np.abs(z_par), "C1 // C2"))
        lo, hi = sorted([cap1.srf, cap2.srf])
        try:
            fpk, zpk = antiresonance_peak(f, z_par, lo, hi)
            print(f"anti-resonance: |Z| peaks at {_eng(fpk, 'Hz')} ({zpk:.3g} Ohm)")
        except ValueError:
            print("anti-resonance: no peak inside the plotted range")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mag, label in curves:
        ax.loglog(f, mag, label=label)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("|Z| [Ohm]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"saved: {args.out}")
    return 0


def cmd_bead(args: argparse.Namespace) -> int:
    f = np.logspace(np.log10(parse_value(args.fmin)), np.log10(parse_value(args.fmax)), 4001)

    bead = BeadModel(
        l=parse_value(args.l),
        r_loss=parse_value(args.r_loss),
        r_dc=parse_value(args.r_dc),
        c_par=parse_value(args.c_par),
    )
    z = bead.impedance(f)
    f100 = np.array([100e6])
    z100 = abs(bead.impedance(f100)[0])
    print(f"bead: L={_eng(bead.l, 'H')}, R_loss={_eng(bead.r_loss, 'Ohm')}, "
          f"R_dc={_eng(bead.r_dc, 'Ohm')}, C_par={_eng(bead.c_par, 'F')}")
    print(f"    |Z| at 100 MHz = {z100:.1f} Ohm")
    print(f"    R/X crossover (wake-up) = {_eng(bead.crossover, 'Hz')}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    two_panel = args.filter_c is not None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5)) if two_panel else plt.subplots(figsize=(7, 4.5))
    ax = axes[0] if two_panel else axes

    ax.loglog(f, np.abs(z), color="#1f77b4", lw=2.2, label="|Z|")
    ax.loglog(f, np.real(z), color="#d62728", lw=1.6, label="R (heat)")
    ax.loglog(f, np.abs(np.imag(z)), color="#2ca02c", lw=1.6, label="X (store/return)")
    ax.axvline(bead.crossover, color="#888", ls="--", lw=0.8)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("Z, R, X [Ohm]")
    ax.set_title("bead Z / R / X")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    if two_panel:
        esr = parse_value(args.filter_esr)
        mlcc = CapacitorModel(parse_value(args.filter_c), esl=0.5e-9, esr=esr)
        gain = series_shunt_gain_db(z, mlcc.impedance(f))
        f_res = 1.0 / (TWO_PI * np.sqrt(bead.l * mlcc.c))
        fpk, gpk = peak_in_band(f, gain, f_res / 5, f_res * 5)
        print(f"bead + {_eng(mlcc.c, 'F')} (ESR {_eng(esr, 'Ohm')}): "
              f"resonance ~ {_eng(f_res, 'Hz')}, peak {gpk:+.1f} dB at {_eng(fpk, 'Hz')}")
        ax2 = axes[1]
        ax2.semilogx(f, gain, color="#d62728", lw=2)
        ax2.axhline(0, color="#888", lw=0.8)
        ax2.annotate(f"+{gpk:.1f} dB\n(noise amplified)", xy=(fpk, gpk),
                     xytext=(fpk * 6, gpk - 4), fontsize=9, color="#d62728",
                     arrowprops=dict(arrowstyle="->", color="#d62728"))
        ax2.set_xlabel("frequency [Hz]")
        ax2.set_ylabel("gain Vout/Vin [dB]  (>0 = worse)")
        ax2.set_title(f"bead + {_eng(mlcc.c, 'F')} MLCC: resonance peak")
        ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"saved: {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="emi-compass", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_cap = sub.add_parser("cap", help="plot |Z| of a real capacitor (and an optional parallel pair)")
    p_cap.add_argument("--c", required=True, help="capacitance, e.g. 0.1u")
    p_cap.add_argument("--esl", default="0.5n", help="ESL (default 0.5n)")
    p_cap.add_argument("--esr", default="0.02", help="ESR in ohms (default 0.02)")
    p_cap.add_argument("--c2", default=None, help="second capacitor for a parallel pair")
    p_cap.add_argument("--esl2", default=None, help="ESL of C2 (default: same as C1)")
    p_cap.add_argument("--esr2", default=None, help="ESR of C2 (default: same as C1)")
    p_cap.add_argument("--fmin", default="10k", help="plot start frequency (default 10k)")
    p_cap.add_argument("--fmax", default="3G", help="plot stop frequency (default 3G)")
    p_cap.add_argument("-o", "--out", default="emi_compass_cap.png", help="output PNG path")
    p_cap.set_defaults(func=cmd_cap)

    p_bead = sub.add_parser("bead", help="plot Z/R/X of a ferrite bead (and an optional bead+C resonance check)")
    p_bead.add_argument("--l", default="0.35u", help="low-frequency inductance (default 0.35u)")
    p_bead.add_argument("--r-loss", default="112", help="parallel loss resistance / resistive plateau (default 112)")
    p_bead.add_argument("--r-dc", default="0.05", help="DC resistance in ohms (default 0.05)")
    p_bead.add_argument("--c-par", default="1p", help="parasitic capacitance (default 1p)")
    p_bead.add_argument("--filter-c", default=None, help="downstream shunt MLCC for a bead+C resonance check, e.g. 10u")
    p_bead.add_argument("--filter-esr", default="5m", help="ESR of the downstream MLCC (default 5m)")
    p_bead.add_argument("--fmin", default="10k", help="plot start frequency (default 10k)")
    p_bead.add_argument("--fmax", default="3G", help="plot stop frequency (default 3G)")
    p_bead.add_argument("-o", "--out", default="emi_compass_bead.png", help="output PNG path")
    p_bead.set_defaults(func=cmd_bead)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
