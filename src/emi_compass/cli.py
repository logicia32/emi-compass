"""emi-compass command line interface.

Ships the `cap` (article 1), `bead` (article 2), and `filter` / `touchstone`
(article 3) subcommands.
"""

from __future__ import annotations

import argparse

import numpy as np

from .models import (
    TWO_PI,
    BeadModel,
    CapacitorModel,
    LCFilterModel,
    antiresonance_peak,
    insertion_loss_from_s21,
    parallel_impedance,
    peak_in_band,
    read_touchstone,
    series_shunt_gain_db,
    series_z_from_s21,
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


def draw_smith(ax, s11: np.ndarray, freqs: np.ndarray | None = None,
               marks: tuple[float, ...] = (), mark_fs: float = 8.0) -> None:
    """Draw a (simplified) Smith chart of a reflection trajectory ``s11`` on
    ``ax``: the unit circle, a few constant-R circles and constant-X arcs, the
    real axis, and the S11 curve. Optional ``marks`` are frequencies (Hz) to dot.
    """
    th = np.linspace(0.0, 2.0 * np.pi, 400)
    cos, sin = np.cos(th), np.sin(th)
    ax.plot(cos, sin, color="#333", lw=1.3)  # unit circle (|Gamma| = 1)
    ax.plot([-1.0, 1.0], [0.0, 0.0], color="#ccc", lw=0.7)  # real axis
    for r in (0.2, 0.5, 1.0, 2.0, 5.0):  # constant-resistance circles
        cx, rad = r / (1.0 + r), 1.0 / (1.0 + r)
        ax.plot(cx + rad * cos, rad * sin, color="#ddd", lw=0.7)
    for x in (0.5, 1.0, 2.0):  # constant-reactance arcs (clipped to the disk)
        for sign in (1.0, -1.0):
            cy, rad = sign / x, 1.0 / x
            xx, yy = 1.0 + rad * cos, cy + rad * sin
            inside = xx * xx + yy * yy <= 1.0 + 1e-9
            ax.plot(xx[inside], yy[inside], color="#ddd", lw=0.7)
    s11 = np.asarray(s11, dtype=complex)
    ax.plot(s11.real, s11.imag, color="#d62728", lw=2.0, label="S11 (reflection)")
    if freqs is not None:
        for fm in marks:
            i = int(np.argmin(np.abs(np.asarray(freqs) - fm)))
            ax.plot(s11[i].real, s11[i].imag, "o", color="#1f77b4")
            ax.annotate(_eng(fm, "Hz"), (s11[i].real, s11[i].imag),
                        textcoords="offset points", xytext=(5, 4), fontsize=mark_fs)
    ax.set_aspect("equal")
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.axis("off")


def cmd_filter(args: argparse.Namespace) -> int:
    f = np.logspace(np.log10(parse_value(args.fmin)), np.log10(parse_value(args.fmax)), 4001)

    filt = LCFilterModel(
        l=parse_value(args.l),
        c=parse_value(args.c),
        dcr=parse_value(args.dcr),
        esr=parse_value(args.esr),
        r_src=parse_value(args.r_src),
        r_load=parse_value(args.r_load),
    )
    gain = filt.gain_db(f)
    f_at = parse_value(args.at)
    att_at = float(filt.attenuation_db(np.array([f_at]))[0])
    fpk, gpk = peak_in_band(f, gain, filt.fc / 8, filt.fc * 8)

    print(f"LC filter: L={_eng(filt.l, 'H')}, C={_eng(filt.c, 'F')}, "
          f"DCR={_eng(filt.dcr, 'Ohm')}, ESR={_eng(filt.esr, 'Ohm')}")
    print(f"    cutoff fc = {_eng(filt.fc, 'Hz')}  (sqrt(L/C) = {filt.z0:.3g} Ohm)")
    print(f"    attenuation at {_eng(f_at, 'Hz')} = {att_at:.1f} dB")
    print(f"    resonance peak {gpk:+.1f} dB at {_eng(fpk, 'Hz')}  (>0 = amplified)")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.semilogx(f, gain, color="#1f77b4", lw=2)
    ax.axhline(0, color="#888", lw=0.9)
    ax.axvline(filt.fc, color="#bbb", ls="--", lw=0.9)
    ax.plot([f_at], [-att_at], "o", color="#d62728")
    ax.annotate(f"{-att_at:.0f} dB\nat {_eng(f_at, 'Hz')}", xy=(f_at, -att_at),
                xytext=(f_at / 6, -att_at + 12), fontsize=9, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("transfer Vout/Vin [dB]  (down = quieter)")
    ax.set_title(f"LC low-pass: fc ~ {_eng(filt.fc, 'Hz')}, 40 dB/dec roll-off")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"saved: {args.out}")
    return 0


def cmd_touchstone(args: argparse.Namespace) -> int:
    f, s = read_touchstone(args.path)
    z = series_z_from_s21(s["s21"], s["z0"])  # read a series part back into |Z|
    il = insertion_loss_from_s21(s["s21"])
    print(f"touchstone: {args.path}  ({len(f)} points, "
          f"{_eng(f[0], 'Hz')}..{_eng(f[-1], 'Hz')}, Z0={s['z0']:.0f} Ohm)")
    f_at = parse_value(args.at)
    i = int(np.argmin(np.abs(f - f_at)))
    print(f"    at {_eng(f[i], 'Hz')}: |Z| = {abs(z[i]):.1f} Ohm, "
          f"insertion loss = {il[i]:.1f} dB (series, 50-ohm system)")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    draw_smith(axes[0], s["s11"], f, marks=(f[0], f[len(f) // 2], f[-1]))
    axes[0].set_title("Smith chart (S11): the round one")

    axes[1].loglog(f, np.abs(z), color="#1f77b4", lw=2.2, label="|Z| from S21")
    axes[1].set_xlabel("frequency [Hz]")
    axes[1].set_ylabel("|Z| [Ohm]")
    axes[1].set_title("frequency x |Z|: the same data, read in one glance")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="upper left")
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

    p_filter = sub.add_parser("filter", help="plot an LC (pi-filter) low-pass: cutoff, roll-off, resonance peak")
    p_filter.add_argument("--l", default="22u", help="choke inductance (default 22u)")
    p_filter.add_argument("--c", default="100u", help="shunt capacitance (default 100u)")
    p_filter.add_argument("--dcr", default="0.05", help="choke DC resistance in ohms (default 0.05)")
    p_filter.add_argument("--esr", default="0", help="shunt-cap ESR in ohms (0 = ideal, clean 40 dB/dec)")
    p_filter.add_argument("--r-src", default="0.1", help="source resistance / rail stiffness (default 0.1)")
    p_filter.add_argument("--r-load", default="1G", help="load resistance (default ~open)")
    p_filter.add_argument("--at", default="1M", help="frequency to report attenuation at (default 1M)")
    p_filter.add_argument("--fmin", default="100", help="plot start frequency (default 100)")
    p_filter.add_argument("--fmax", default="10M", help="plot stop frequency (default 10M)")
    p_filter.add_argument("-o", "--out", default="emi_compass_filter.png", help="output PNG path")
    p_filter.set_defaults(func=cmd_filter)

    p_ts = sub.add_parser("touchstone", help="read a vendor .s2p and draw |Z| (and the Smith chart)")
    p_ts.add_argument("path", help="path to a 2-port Touchstone (.s2p) file")
    p_ts.add_argument("--at", default="100M", help="frequency to report |Z| / IL at (default 100M)")
    p_ts.add_argument("-o", "--out", default="emi_compass_touchstone.png", help="output PNG path")
    p_ts.set_defaults(func=cmd_touchstone)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
