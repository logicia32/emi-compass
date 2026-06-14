"""Generate the figures for series article 2 (chip-bead edition).

Outputs land in outputs/. All labels are in English (the article body carries
the Japanese context), matching make_article1_figures.py.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow, FancyBboxPatch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from emi_compass.models import (  # noqa: E402
    TWO_PI,
    BeadModel,
    CapacitorModel,
    peak_in_band,
    series_insertion_loss_db,
    series_shunt_gain_db,
)

OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

F = np.logspace(4, 9.5, 6001)  # 10 kHz .. ~3 GHz

# A typical "100 ohm @ 100 MHz" class bead.
NOMINAL = BeadModel(l=0.35e-6, r_loss=112.0, r_dc=0.05, c_par=1e-12, label="bead")
# Two beads that both pass ~100 ohm at 100 MHz but split R/X differently.
R_TYPE = BeadModel(l=0.55e-6, r_loss=105.0, r_dc=0.05, c_par=1e-12, label="R-type")
X_TYPE = BeadModel(l=0.21e-6, r_loss=156.0, r_dc=0.05, c_par=1e-12, label="X-type")

summary: list[str] = []


def grid(ax):
    ax.grid(True, which="both", alpha=0.3)


def at(bead, f_hz):
    z = bead.impedance(np.array([f_hz]))[0]
    return abs(z), z.real, abs(z.imag)


# ---------------------------------------------------------------- fig 01
# Three strategies (diagram): divert / block & return / turn to heat
def fig01():
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 8.4))
    titles = ["Capacitor: divert  (energy stays)",
              "Inductor: block & return  (energy stays)",
              "Bead: turn to heat  (energy gone)"]
    for ax, title in zip(axes, titles):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis("off")
        ax.set_title(title, fontsize=14, pad=8)
        ax.plot([0.5, 9.5], [3.6, 3.6], color="#1f77b4", lw=2.8)
        ax.plot([0.5, 9.5], [1.0, 1.0], color="#555", lw=2.8)
        ax.text(0.5, 4.3, "noise in ->", fontsize=12, color="#d62728")

    # capacitor: shunt to GND
    ax = axes[0]
    ax.add_patch(FancyBboxPatch((4.4, 2.1), 1.2, 0.8, boxstyle="round,pad=0.04",
                                fc="#cfe3f7", ec="#1f77b4"))
    ax.plot([5.0, 5.0], [3.6, 2.9], color="#1f77b4", lw=2.3)
    ax.plot([5.0, 5.0], [2.1, 1.0], color="#1f77b4", lw=2.3)
    ax.add_patch(FancyArrow(5.0, 3.2, 0, -1.4, width=0.05, head_width=0.34,
                            head_length=0.32, color="#d62728"))
    ax.text(5.7, 1.9, "to GND", fontsize=12, color="#d62728")

    # inductor: series, bounce back
    ax = axes[1]
    ax.add_patch(FancyBboxPatch((4.2, 3.2), 1.6, 0.8, boxstyle="round,pad=0.04",
                                fc="#d8f0d8", ec="#2ca02c"))
    ax.text(5.0, 3.6, "L", ha="center", va="center", fontsize=14, color="#2ca02c")
    ax.add_patch(FancyArrow(3.7, 3.6, -1.7, 0, width=0.05, head_width=0.34,
                            head_length=0.4, color="#d62728"))
    ax.text(2.3, 2.9, "reflected back", fontsize=12, color="#d62728")

    # bead: series, becomes heat
    ax = axes[2]
    ax.add_patch(FancyBboxPatch((4.2, 3.2), 1.6, 0.8, boxstyle="round,pad=0.04",
                                fc="#f7d9d9", ec="#d62728"))
    ax.text(5.0, 3.6, "bead", ha="center", va="center", fontsize=13, color="#d62728")
    for dx in (-0.3, 0.0, 0.3):
        ax.plot([5.0 + dx, 5.0 + dx + 0.14], [2.9, 2.3], color="#e08a00", lw=2.0)
        ax.plot([5.0 + dx + 0.14, 5.0 + dx], [2.3, 1.7], color="#e08a00", lw=2.0)
    ax.text(5.0, 1.2, "heat", ha="center", fontsize=12, color="#e08a00")

    fig.tight_layout(h_pad=2.5)
    fig.savefig(OUT / "01-three-strategies.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- fig 02
def fig02():
    z = NOMINAL.impedance(F)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.loglog(F, np.abs(z), color="#1f77b4", lw=2.4, label="Z (total)")
    ax.loglog(F, np.real(z), color="#d62728", lw=1.7, label="R (turns to heat)")
    ax.loglog(F, np.abs(np.imag(z)), color="#2ca02c", lw=1.7, label="X (stores / returns)")
    fc = NOMINAL.crossover
    ax.axvline(fc, color="#888", ls="--", lw=0.9)
    ax.annotate(f"wake-up freq\nR = X  (~{fc/1e6:.0f} MHz)", xy=(fc, 60),
                xytext=(1.2e6, 135), fontsize=10, color="#444",
                arrowprops=dict(arrowstyle="->", color="#888"))
    ax.text(1.3e4, 0.45, "X dominates\n(inductor)", fontsize=10, color="#2ca02c", va="top")
    ax.text(1.8e7, 0.5, "R dominates\n(turns to heat)", fontsize=10, color="#d62728", va="top")
    ax.text(4.5e8, 0.45, "parasitic C:\n|Z| falls", fontsize=10, color="#777", va="top")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("Z, R, X [Ohm]")
    ax.set_title("the bead's three lines: Z / R / X")
    ax.set_ylim(0.04, 3.2e2)
    ax.legend(loc="upper left")
    grid(ax)
    fig.tight_layout()
    fig.savefig(OUT / "02-bead-zrx.png", dpi=140)
    plt.close(fig)
    zt, rt, xt = at(NOMINAL, 100e6)
    summary.append(f"NOMINAL bead: crossover {fc/1e6:.1f} MHz, "
                   f"|Z|@100MHz {zt:.1f} Ohm (R {rt:.0f}, X {xt:.0f})")


# ---------------------------------------------------------------- fig 03
def fig03():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, bead in zip(axes, (R_TYPE, X_TYPE)):
        z = bead.impedance(F)
        ax.loglog(F, np.abs(z), color="#1f77b4", lw=2.2, label="Z")
        ax.loglog(F, np.real(z), color="#d62728", lw=1.5, label="R")
        ax.loglog(F, np.abs(np.imag(z)), color="#2ca02c", lw=1.5, label="X")
        zt, rt, xt = at(bead, 100e6)
        ax.plot([100e6], [zt], "o", color="#1f77b4")
        ax.axvline(100e6, color="#bbb", ls=":", lw=0.8)
        ax.set_title(f"{bead.label}: |Z|@100MHz ~ {zt:.0f} Ohm  (R {rt:.0f}, X {xt:.0f})",
                     fontsize=10)
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylim(0.05, 3e2)
        grid(ax)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("Z, R, X [Ohm]")
    fig.suptitle("same '100 Ohm @ 100 MHz', different character", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "03-same-z-different-rx.png", dpi=140)
    plt.close(fig)
    for b in (R_TYPE, X_TYPE):
        zt, rt, xt = at(b, 100e6)
        summary.append(f"{b.label}: |Z|@100MHz {zt:.0f} (R {rt:.0f} / X {xt:.0f}), "
                       f"crossover {b.crossover/1e6:.0f} MHz")


# ---------------------------------------------------------------- fig 04
def fig04():
    z = NOMINAL.impedance(F)
    il_bench = series_insertion_loss_db(z, 100.0)  # 50 + 50 ohm bench
    il_rail = series_insertion_loss_db(z, 2.0)  # stiff 2-ohm rail, example
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.semilogx(F, il_bench, color="#1f77b4", lw=2,
                label="50-ohm bench (R_sys = 100 ohm)  <- the datasheet")
    ax.semilogx(F, il_rail, color="#d62728", lw=2,
                label="stiff power rail (R_sys = 2 ohm, example)")
    b = float(series_insertion_loss_db(NOMINAL.impedance(np.array([100e6])), 100.0)[0])
    r = float(series_insertion_loss_db(NOMINAL.impedance(np.array([100e6])), 2.0)[0])
    ax.annotate("", xy=(100e6, r), xytext=(100e6, b),
                arrowprops=dict(arrowstyle="<->", color="#555"))
    ax.text(1.25e8, (b + r) / 2, f"{b:.0f} dB -> {r:.0f} dB\nat 100 MHz", fontsize=10,
            color="#555")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("insertion loss [dB]  (bigger = quieter)")
    ax.set_title("same 100-ohm bead: a series part works better on a stiff rail")
    ax.set_ylim(0, 45)
    ax.legend(loc="upper left", fontsize=9)
    grid(ax)
    fig.tight_layout()
    fig.savefig(OUT / "04-series-il-context.png", dpi=140)
    plt.close(fig)
    summary.append(f"series IL of NOMINAL bead at 100 MHz: 50ohm-bench {b:.1f} dB / 2ohm-rail {r:.1f} dB")


# ---------------------------------------------------------------- fig 05
def fig05():
    """Square wave through source(50) - bead(series) - load capacitance (IC pin):
    the bead + load cap form a low-pass, so the signal edges round off."""
    fs = 2e10
    n = 20000
    t = np.arange(n) / fs
    f0 = 30e6
    sq = np.sign(np.sin(TWO_PI * f0 * t))
    spec = np.fft.rfft(sq)
    fr = np.fft.rfftfreq(n, 1 / fs)
    fr_safe = np.maximum(fr, 1.0)
    zb = NOMINAL.impedance(fr_safe)
    cload = CapacitorModel(c=47e-12, esl=0.0, esr=0.0)  # IC input + stray, to GND
    zc = cload.impedance(fr_safe)
    h = zc / (50.0 + zb + zc)  # output across the load cap
    out = np.fft.irfft(spec * h, n)
    sq_n = sq / np.max(np.abs(sq))
    out_n = out / np.max(np.abs(out))

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    sl = slice(0, int(fs / f0 * 3))  # ~3 periods
    ax.plot(t[sl] * 1e9, sq_n[sl], color="#888", lw=1.4, label="input (fast square)")
    ax.plot(t[sl] * 1e9, out_n[sl], color="#d62728", lw=2,
            label="after the bead, into a load cap (edges rounded)")
    ax.set_xlabel("time [ns]")
    ax.set_ylabel("signal (shape, normalized)")
    ax.set_title("a bead rounds signal edges too (30 MHz square, 47 pF load)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "05-edge-rounding.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- fig 06
def fig06():
    """DC-bias derating (illustrative): more DC current sinks the whole Z curve."""
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for scale, color, label in ((1.0, "#1f77b4", "no DC bias (catalog)"),
                                (0.6, "#9467bd", "half rated current"),
                                (0.35, "#d62728", "near rated current")):
        b = BeadModel(l=NOMINAL.l * scale, r_loss=NOMINAL.r_loss * scale,
                      r_dc=NOMINAL.r_dc, c_par=NOMINAL.c_par)
        ax.loglog(F, np.abs(b.impedance(F)), color=color, lw=2, label=label)
    ax.annotate("more DC current\n-> ferrite saturates\n-> |Z| sinks",
                xy=(1.1e8, 38), xytext=(3e7, 0.55), fontsize=10, color="#444",
                ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color="#888"))
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("|Z| [Ohm]")
    ax.set_title("DC-bias derating (illustrative shape)")
    ax.set_ylim(0.05, 2e2)
    ax.legend(loc="upper left", fontsize=9)
    grid(ax)
    fig.tight_layout()
    fig.savefig(OUT / "06-dc-bias.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- fig 07
def fig07():
    zb = NOMINAL.impedance(F)
    mlcc = CapacitorModel(c=10e-6, esl=0.5e-9, esr=0.005)  # low-ESR MLCC
    gain = series_shunt_gain_db(zb, mlcc.impedance(F))
    f_res = 1.0 / (TWO_PI * np.sqrt(NOMINAL.l * mlcc.c))
    fpk, gpk = peak_in_band(F, gain, f_res / 5, f_res * 5)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.semilogx(F, gain, color="#d62728", lw=2)
    ax.axhline(0, color="#888", lw=0.9)
    ax.fill_between(F, 0, gain, where=(gain > 0), color="#d62728", alpha=0.15)
    ax.annotate(f"+{gpk:.1f} dB peak\n(~{fpk/1e3:.0f} kHz): noise amplified",
                xy=(fpk, gpk), xytext=(fpk * 4, gpk - 6), fontsize=10, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("gain Vout/Vin [dB]  (>0 = worse)")
    ax.set_title("bead (0.35 uH) + low-ESR 10 uF MLCC: a resonance peak")
    ax.set_xlim(1e4, 1e7)
    ax.set_ylim(-30, 15)
    grid(ax)
    fig.tight_layout()
    fig.savefig(OUT / "07-bead-c-resonance.png", dpi=140)
    plt.close(fig)
    summary.append(f"bead + 10uF (ESR 5mOhm): resonance ~{f_res/1e3:.0f} kHz, peak {gpk:+.1f} dB")


# ---------------------------------------------------------------- fig 08
def fig08():
    zb = NOMINAL.impedance(F)
    low = CapacitorModel(c=10e-6, esl=0.5e-9, esr=0.005)
    damped = CapacitorModel(c=10e-6, esl=0.5e-9, esr=0.6)  # e.g. an electrolytic
    g_low = series_shunt_gain_db(zb, low.impedance(F))
    g_damped = series_shunt_gain_db(zb, damped.impedance(F))
    f_res = 1.0 / (TWO_PI * np.sqrt(NOMINAL.l * low.c))
    _, gpk_d = peak_in_band(F, g_damped, f_res / 5, f_res * 5)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.semilogx(F, g_low, color="#d62728", lw=2, label="low-ESR MLCC (peaks)")
    ax.semilogx(F, g_damped, color="#1f77b4", lw=2,
                label="damped: add ESR / lossy cap (flat)")
    ax.axhline(0, color="#888", lw=0.9)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("gain Vout/Vin [dB]  (>0 = worse)")
    ax.set_title("damping removes the peak")
    ax.set_xlim(1e4, 1e7)
    ax.set_ylim(-30, 15)
    ax.legend(loc="lower left", fontsize=9)
    grid(ax)
    fig.tight_layout()
    fig.savefig(OUT / "08-bead-c-damped.png", dpi=140)
    plt.close(fig)
    summary.append(f"damped (ESR 0.6 Ohm): peak now {gpk_d:+.1f} dB")


def main() -> None:
    for fn in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08):
        fn()
        print(f"done: {fn.__name__}")
    text = "\n".join(summary)
    (OUT / "summary2.txt").write_text(text + "\n")
    print("--- summary ---")
    print(text)


if __name__ == "__main__":
    main()
