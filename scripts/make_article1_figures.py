"""Generate the figures for series article 1 (capacitor edition).

Outputs land in outputs/. All labels are in English (the article body carries
the Japanese context).
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow, FancyBboxPatch, Rectangle

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from emi_compass.models import (  # noqa: E402
    CapacitorModel,
    antiresonance_peak,
    ideal_cap_impedance,
    parallel_impedance,
    shunt_insertion_loss_db,
    shunt_insertion_loss_db_sys,
)

OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

F = np.logspace(4, 9.5, 6001)  # 10 kHz .. ~3 GHz

MLCC = CapacitorModel(c=0.1e-6, esl=0.5e-9, esr=0.02, label="MLCC 0.1uF")
MLCC_SMALL = CapacitorModel(c=0.01e-6, esl=0.5e-9, esr=0.02, label="MLCC 0.01uF")
MLCC_BIG = CapacitorModel(c=1e-6, esl=0.5e-9, esr=0.02, label="MLCC 1uF")
THREE_T = CapacitorModel(c=0.1e-6, esl=0.05e-9, esr=0.02, label="3-terminal 0.1uF")

summary: list[str] = []


def grid(ax):
    ax.grid(True, which="both", alpha=0.3)


# ---------------------------------------------------------------- fig 01
# Two noise modes (diagram)
def fig01():
    # Stacked vertically so each panel uses the full article column width;
    # on Zenn a wide side-by-side diagram gets scaled down and the labels
    # become hard to read.
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.2))
    for ax, title in zip(axes, ["Differential (normal) mode", "Common mode"]):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis("off")
        ax.set_title(title, fontsize=15, pad=10)
        # source / load boxes
        for x, name in ((0.4, "noise\nsource"), (8.2, "circuit")):
            ax.add_patch(FancyBboxPatch((x, 1.4), 1.4, 2.2, boxstyle="round,pad=0.08",
                                        fc="#eef2f7", ec="#444"))
            ax.text(x + 0.7, 2.5, name, ha="center", va="center", fontsize=12)
        # two conductors
        ax.plot([1.8, 8.2], [3.2, 3.2], color="#1f77b4", lw=2.4)
        ax.plot([1.8, 8.2], [1.8, 1.8], color="#555", lw=2.4)
        ax.text(5.0, 3.5, "line", ha="center", fontsize=12, color="#1f77b4")
        ax.text(5.0, 1.42, "return (GND)", ha="center", fontsize=12, color="#555")

    ax = axes[0]
    ax.add_patch(FancyArrow(3.6, 3.2, 1.6, 0, width=0.07, head_width=0.34,
                            head_length=0.42, color="#d62728"))
    ax.add_patch(FancyArrow(6.4, 1.8, -1.6, 0, width=0.07, head_width=0.34,
                            head_length=0.42, color="#d62728"))
    ax.text(5.0, 2.5, "goes out on the line,\ncomes back on the return",
            ha="center", va="center", fontsize=12, color="#d62728")

    ax = axes[1]
    for y in (3.2, 1.8):
        ax.add_patch(FancyArrow(3.6, y, 1.6, 0, width=0.07, head_width=0.34,
                                head_length=0.42, color="#d62728"))
    # stray return path
    ax.plot([8.9, 8.9, 0.9, 0.9], [1.4, 0.4, 0.4, 1.4], color="#d62728",
            lw=1.8, ls="--")
    ax.text(5.0, 0.66, "returns via stray capacitance / chassis", ha="center",
            fontsize=12, color="#d62728")
    ax.text(5.0, 2.5, "both conductors swing together\n(vs. the outside world)",
            ha="center", va="center", fontsize=12, color="#d62728")

    fig.tight_layout(h_pad=2.0)
    fig.savefig(OUT / "01-two-noise-modes.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- fig 02
# Time domain vs frequency domain
def fig02():
    rng = np.random.default_rng(7)
    fs = 2e9
    n = 40000
    t = np.arange(n) / fs
    vdd = 3.3
    f_noise = 80e6
    # before: 25 mV switching ripple + harmonics + floor
    noise = (0.025 * np.sin(2 * np.pi * f_noise * t)
             + 0.008 * np.sin(2 * np.pi * 3 * f_noise * t + 0.7)
             + 0.004 * rng.standard_normal(n))
    after = (0.0025 * np.sin(2 * np.pi * f_noise * t)
             + 0.0008 * np.sin(2 * np.pi * 3 * f_noise * t + 0.7)
             + 0.004 * rng.standard_normal(n))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    sl = slice(0, 400)
    scope_fuzz = 0.012 * rng.standard_normal(n)  # oscilloscope front-end noise
    ax.plot(t[sl] * 1e9, vdd + noise[sl] + scope_fuzz[sl], lw=0.8, label="no filter")
    ax.plot(t[sl] * 1e9, vdd + after[sl] + scope_fuzz[sl] - 0.3, lw=0.8,
            label="with filter (offset -0.3 V)")
    ax.set_ylim(2.5, 3.8)
    ax.set_xlabel("time [ns]")
    ax.set_ylabel("voltage [V]")
    ax.set_title("time domain: hard to tell apart")
    ax.legend(fontsize=8, loc="upper right")
    grid(ax)

    ax = axes[1]
    seg = 20
    for sig, label in ((noise, "no filter"), (after, "with filter")):
        chunks = sig.reshape(seg, n // seg)
        spec = np.mean([np.abs(np.fft.rfft(c)) / (n // seg) * 2 for c in chunks], axis=0)
        freq = np.fft.rfftfreq(n // seg, 1 / fs)
        m = (freq > 2e6) & (freq < 1e9)
        ax.semilogx(freq[m], 20 * np.log10(spec[m] + 1e-12), lw=1.2, label=label)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("noise level [dBV]")
    ax.set_title("frequency domain: the difference is obvious")
    ax.legend(fontsize=8)
    grid(ax)

    fig.tight_layout()
    fig.savefig(OUT / "02-time-vs-freq.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- fig 03
def fig03():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(F, np.abs(ideal_cap_impedance(F, MLCC.c)), "--", color="#888",
              label="ideal 0.1 uF (keeps falling)")
    ax.loglog(F, np.abs(MLCC.impedance(F)), color="#1f77b4", lw=2,
              label="real MLCC 0.1 uF (ESL 0.5 nH, ESR 20 mOhm)")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("|Z| [Ohm]")
    ax.set_title("ideal vs real: the V-curve")
    ax.legend()
    grid(ax)
    fig.tight_layout()
    fig.savefig(OUT / "03-ideal-vs-real.png", dpi=140)
    plt.close(fig)
    summary.append(f"MLCC 0.1uF/0.5nH/20mOhm SRF = {MLCC.srf/1e6:.1f} MHz")


# ---------------------------------------------------------------- fig 04
def fig04():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    z = np.abs(MLCC.impedance(F))
    ax.loglog(F, z, color="#1f77b4", lw=2)
    srf = MLCC.srf

    ax.annotate("left slope:\ncapacitive  1/(2*pi*f*C)", xy=(3e5, 5.3), xytext=(2e4, 0.08),
                fontsize=10, color="#2ca02c",
                arrowprops=dict(arrowstyle="->", color="#2ca02c"))
    ax.annotate("right slope:\ninductive  2*pi*f*ESL", xy=(7e8, 2.2), xytext=(6e6, 30),
                fontsize=10, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax.annotate(f"valley = SRF ({srf/1e6:.0f} MHz)\ndepth = ESR (20 mOhm)",
                xy=(srf, MLCC.esr), xytext=(1.5e7, 0.0015),
                fontsize=10, color="#9467bd",
                arrowprops=dict(arrowstyle="->", color="#9467bd"))
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("|Z| [Ohm]")
    ax.set_title("anatomy of the V-curve (MLCC 0.1 uF)")
    ax.set_ylim(1e-3, 3e2)
    grid(ax)
    fig.tight_layout()
    fig.savefig(OUT / "04-v-curve-anatomy.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- fig 05
def fig05():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cap, color in ((MLCC_SMALL, "#2ca02c"), (MLCC, "#1f77b4"), (MLCC_BIG, "#d62728")):
        ax.loglog(F, np.abs(cap.impedance(F)), color=color, lw=1.8,
                  label=f"{cap.c*1e6:g} uF (SRF {cap.srf/1e6:.0f} MHz)")
    ax.annotate("right slopes all overlap:\nset by ESL, not by C",
                xy=(1e9, 3.1), xytext=(2e7, 60), fontsize=10,
                arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("|Z| [Ohm]")
    ax.set_title("same size & ESL (0.5 nH), capacitance x100")
    ax.legend(loc="lower left")
    grid(ax)
    fig.tight_layout()
    fig.savefig(OUT / "05-capacitance-family.png", dpi=140)
    plt.close(fig)
    for cap in (MLCC_SMALL, MLCC, MLCC_BIG):
        z1g = abs(cap.impedance(np.array([1e9]))[0])
        summary.append(f"|Z| at 1 GHz, C={cap.c*1e6:g}uF: {z1g:.3f} Ohm")


# ---------------------------------------------------------------- fig 06
def fig06():
    z1 = MLCC.impedance(F)
    z2 = MLCC_SMALL.impedance(F)
    zp = parallel_impedance(z1, z2)
    fpk, zpk = antiresonance_peak(F, zp, MLCC.srf, MLCC_SMALL.srf)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(F, np.abs(z1), color="#1f77b4", lw=1.2, alpha=0.6, label="0.1 uF alone")
    ax.loglog(F, np.abs(z2), color="#2ca02c", lw=1.2, alpha=0.6, label="0.01 uF alone")
    ax.loglog(F, np.abs(zp), color="#d62728", lw=2.2, label="0.1 uF // 0.01 uF")
    ax.annotate(f"anti-resonance peak\n({fpk/1e6:.0f} MHz, {zpk:.2f} Ohm)",
                xy=(fpk, zpk), xytext=(8e6, 40), fontsize=11, color="#d62728",
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color="#d62728",
                                shrinkA=6, shrinkB=4))
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("|Z| [Ohm]")
    ax.set_title("two valleys, and the hill between them")
    ax.legend(loc="lower left")
    grid(ax)
    fig.tight_layout()
    fig.savefig(OUT / "06-parallel-antiresonance.png", dpi=140)
    plt.close(fig)
    summary.append(f"anti-resonance (0.1uF // 0.01uF): {fpk/1e6:.1f} MHz, {zpk:.2f} Ohm")


# ---------------------------------------------------------------- fig 07
# Three-terminal structure (diagram)
def fig07():
    # Stacked vertically (like fig01) so each schematic uses the full column
    # width on Zenn and the labels stay legible.
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 8.2))

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("2-terminal MLCC: the current takes a detour", fontsize=15, pad=8)
    ax.plot([0.5, 9.5], [4.6, 4.6], color="#1f77b4", lw=3.5)
    ax.text(2.4, 5.0, "power line", ha="center", fontsize=13, color="#1f77b4")
    ax.add_patch(Rectangle((4.2, 2.2), 1.6, 1.2, fc="#c9a36a", ec="#444"))
    ax.text(5.0, 2.8, "MLCC", ha="center", va="center", fontsize=12)
    ax.plot([5.0, 5.0], [4.6, 3.4], color="#d62728", lw=2.6)
    ax.plot([5.0, 5.0], [2.2, 1.0], color="#d62728", lw=2.6)
    ax.plot([0.5, 9.5], [1.0, 1.0], color="#555", lw=3.5)
    ax.text(2.4, 0.5, "GND", ha="center", fontsize=13, color="#555")
    ax.annotate("this whole detour\n(pads, vias, electrodes)\nacts as ESL",
                xy=(5.05, 3.9), xytext=(6.5, 2.7), fontsize=13, color="#d62728",
                va="center", arrowprops=dict(arrowstyle="->", color="#d62728",
                                             shrinkA=4, shrinkB=4))

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("3-terminal: the line passes through the part", fontsize=15, pad=8)
    ax.plot([0.5, 3.6], [3.0, 3.0], color="#1f77b4", lw=3.5)
    ax.plot([6.4, 9.5], [3.0, 3.0], color="#1f77b4", lw=3.5)
    ax.add_patch(Rectangle((3.6, 2.2), 2.8, 1.6, fc="#c9a36a", ec="#444"))
    ax.plot([3.6, 6.4], [3.0, 3.0], color="#1f77b4", lw=3.5)  # feedthrough electrode
    ax.text(5.0, 4.2, "feedthrough electrode", ha="center", fontsize=13, color="#1f77b4")
    # wide ground tabs on both sides
    for x in (4.3, 5.7):
        ax.plot([x, x], [2.2, 1.0], color="#2ca02c", lw=7, solid_capstyle="butt")
    ax.plot([0.5, 9.5], [1.0, 1.0], color="#555", lw=3.5)
    ax.text(2.4, 0.5, "GND", ha="center", fontsize=13, color="#555")
    ax.annotate("short, wide GND tabs\non both sides -> tiny ESL",
                xy=(5.75, 1.5), xytext=(6.8, 2.4), fontsize=13, color="#2ca02c",
                va="center", arrowprops=dict(arrowstyle="->", color="#2ca02c",
                                             shrinkA=4, shrinkB=4))

    fig.tight_layout(h_pad=2.2)
    fig.savefig(OUT / "07-three-terminal-structure.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- fig 08
def fig08():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(F, np.abs(MLCC.impedance(F)), color="#1f77b4", lw=2,
              label=f"2-terminal MLCC (ESL 0.5 nH, SRF {MLCC.srf/1e6:.0f} MHz)")
    ax.loglog(F, np.abs(THREE_T.impedance(F)), color="#d62728", lw=2,
              label=f"3-terminal (ESL 0.05 nH, SRF {THREE_T.srf/1e6:.0f} MHz)")
    f100 = np.array([100e6])
    z_n = abs(MLCC.impedance(f100)[0])
    z_3t = abs(THREE_T.impedance(f100)[0])
    ax.annotate("", xy=(100e6, z_3t), xytext=(100e6, z_n),
                arrowprops=dict(arrowstyle="<->", color="#555"))
    # Park the label in the open band between the two curves, just right of the
    # arrow, so it doesn't sit on the rising 3-terminal slope.
    ax.text(1.25e8, z_n * 0.6, f"x{z_n / z_3t:.0f}\nat 100 MHz",
            fontsize=11, color="#555", ha="left", va="center")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("|Z| [Ohm]")
    ax.set_title("same 0.1 uF, different structure")
    ax.legend(loc="upper left")
    grid(ax)
    fig.tight_layout()
    fig.savefig(OUT / "08-three-terminal-z.png", dpi=140)
    plt.close(fig)
    summary.append(f"|Z| at 100 MHz: 2-terminal {z_n:.3f} Ohm / 3-terminal {z_3t:.4f} Ohm"
                   f" (x{z_n/z_3t:.1f})")
    summary.append(f"3-terminal SRF = {THREE_T.srf/1e6:.0f} MHz")


# ---------------------------------------------------------------- fig 09
def fig09():
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for cap, color, label in ((MLCC, "#1f77b4", "2-terminal MLCC 0.1 uF"),
                              (THREE_T, "#d62728", "3-terminal 0.1 uF")):
        il = shunt_insertion_loss_db(cap.impedance(F))
        ax.semilogx(F, il, color=color, lw=2, label=label)
    for db, txt in ((20, "noise voltage 1/10"), (40, "1/100"), (60, "1/1000")):
        ax.axhline(db, color="#aaa", lw=0.8, ls="--")
        ax.text(1.3e4, db + 1, f"{db} dB = {txt}", fontsize=8, color="#777")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("insertion loss [dB]  (bigger = quieter)")
    ax.set_title("insertion loss in a 50-Ohm test setup (comparison use only)")
    ax.set_ylim(0, 75)
    ax.legend(loc="upper right")
    grid(ax)
    fig.tight_layout()
    fig.savefig(OUT / "09-insertion-loss.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- fig 10
def fig10():
    """Same capacitor, two surrounding impedances: 50-ohm bench vs stiff rail."""
    z = MLCC.impedance(F)
    il_bench = shunt_insertion_loss_db_sys(z, 25.0)  # 50 || 50
    il_rail = shunt_insertion_loss_db_sys(z, 1.0)  # stiff power line, example

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.semilogx(F, il_bench, color="#1f77b4", lw=2,
                label="50-ohm bench (R_sys = 25 ohm)  <- the datasheet")
    ax.semilogx(F, il_rail, color="#d62728", lw=2,
                label="stiff power rail (R_sys = 1 ohm, example)")
    f100 = np.array([100e6])
    b = float(shunt_insertion_loss_db_sys(MLCC.impedance(f100), 25.0)[0])
    r = float(shunt_insertion_loss_db_sys(MLCC.impedance(f100), 1.0)[0])
    ax.annotate("", xy=(100e6, r), xytext=(100e6, b),
                arrowprops=dict(arrowstyle="<->", color="#555"))
    # Sit the label just above the arrow's top, clear of the falling bench curve.
    ax.text(1.05e8, b + 3, f"{b:.0f} dB -> {r:.0f} dB\nat 100 MHz", fontsize=11,
            color="#555", ha="left", va="bottom")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("insertion loss [dB]  (bigger = quieter)")
    ax.set_title("same 0.1 uF cap: the dB shrinks on a low-impedance rail")
    ax.set_ylim(0, 75)
    ax.legend(loc="upper left", fontsize=9)
    grid(ax)
    fig.tight_layout()
    fig.savefig(OUT / "10-impedance-context.png", dpi=140)
    plt.close(fig)
    summary.append(f"insertion loss of 0.1uF MLCC at 100 MHz: 50ohm-bench {b:.1f} dB"
                   f" / 1ohm-rail {r:.1f} dB")


def main() -> None:
    for fn in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10):
        fn()
        print(f"done: {fn.__name__}")
    text = "\n".join(summary)
    (OUT / "summary.txt").write_text(text + "\n")
    print("--- summary ---")
    print(text)


if __name__ == "__main__":
    main()
