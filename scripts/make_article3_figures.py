"""Generate the figures for series article 3 (selection-flow edition).

Outputs land in outputs/. Also (re)generates samples/sample_bead.s2p, the
synthetic Touchstone file the article reads. Curve graphs keep English axis
labels (matching the other figure scripts); the flowcharts (fig01 / fig08) use
Japanese because they are prose decision diagrams, not axis-labelled plots.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emi_compass.cli import draw_smith  # noqa: E402
from emi_compass.models import (  # noqa: E402
    BeadModel,
    CapacitorModel,
    LCFilterModel,
    peak_in_band,
    read_touchstone,
    s_series_element,
    series_z_from_s21,
    write_touchstone,
)

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SAMPLES = ROOT / "samples"
SAMPLES.mkdir(exist_ok=True)

# palette
BLUE, RED, GREEN, PURPLE, ORANGE, GRAY = (
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#e08a00", "#888888",
)
FILL_B, FILL_G, FILL_R, FILL_Y, FILL_GR = (
    "#cfe3f7", "#d8f0d8", "#f7d9d9", "#fdf0c8", "#ededed",
)

NOMINAL = BeadModel(l=0.35e-6, r_loss=112.0, r_dc=0.05, c_par=1e-12, label="bead")

summary: list[str] = []


def grid(ax):
    ax.grid(True, which="both", alpha=0.3)


def box(ax, x, y, w, h, text, fc, ec=GRAY, fs=9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.04", fc=fc, ec=ec, lw=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="#222")


def diamond(ax, x, y, w, h, text, fc, ec=GRAY, fs=9):
    pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="#222")


def arrow(ax, x1, y1, x2, y2, label=None, lcolor="#666", lfs=8.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.4))
    if label:
        ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2, label, fontsize=lfs,
                color=lcolor, ha="left", va="center")


# ---------------------------------------------------------------- sample .s2p
def make_sample_s2p():
    """Write a synthetic 2-port .s2p of the nominal bead (a SERIES element)."""
    f = np.logspace(6, 9.3, 220)  # 1 MHz .. ~2 GHz
    z = NOMINAL.impedance(f)
    s11, s21 = s_series_element(z, z0=50.0)
    write_touchstone(
        SAMPLES / "sample_bead.s2p", f, s11, s21, z0=50.0,
        comment=("Synthetic sample for emi-compass article 3.\n"
                 "A nominal '100 ohm @ 100 MHz' ferrite bead as a series\n"
                 "element, written from the equivalent-circuit model -- NOT a\n"
                 "real vendor measurement. For trying the read-in flow only."),
    )
    summary.append("wrote samples/sample_bead.s2p (synthetic bead, 220 pts, 1 MHz..2 GHz)")


# ---------------------------------------------------------------- fig 01
def fig01():
    # The flowchart is the article's centre-piece decision diagram, so its
    # labels are Japanese (the curve graphs keep English axis labels). Fonts
    # are bumped because Zenn downscales wide figures.
    with plt.rc_context({"font.family": "IPAGothic", "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=(9.2, 8.0))
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 12.4)
        ax.axis("off")

        box(ax, 6.5, 11.6, 4.4, 0.8, "何を、どこに入れる?", FILL_GR, fs=13)
        diamond(ax, 6.5, 9.9, 4.8, 1.7, "アナログ信号を\n扱う?", FILL_Y, fs=13)
        box(ax, 2.7, 7.7, 4.9, 1.35,
            "いいえ（デジタルのみ）:\n一般的なデカップリングで足りる\n各 IC のピン直近に MLCC",
            FILL_B, BLUE, fs=11.5)
        box(ax, 9.7, 8.0, 5.0, 1.1,
            "はい（センサ / ADC / オペアンプ）:\nノイズ対策は実質「必須」", FILL_R, RED, fs=11.5)
        diamond(ax, 9.7, 6.0, 4.4, 1.5, "ノイズの\n周波数は?", FILL_Y, fs=13)

        box(ax, 2.55, 3.4, 3.9, 1.2, "数 kHz〜数百 kHz\nLC（π 型）フィルタ", FILL_G, GREEN, fs=11.5)
        box(ax, 7.0, 3.4, 3.9, 1.2, "数 MHz〜数十 MHz\nMLCC シャント / ビーズ", FILL_G, GREEN, fs=11)
        box(ax, 11.1, 3.4, 3.7, 1.2, "数十 MHz〜GHz\n三端子コンデンサ / ビーズの R", FILL_G, GREEN, fs=10)

        box(ax, 6.5, 1.0, 12.4, 0.9,
            "最後に必ず: 配線とレイアウトを確認（約 1nH/mm — ピン直近に置く）",
            FILL_GR, fs=12)

        arrow(ax, 6.5, 11.2, 6.5, 10.75)
        arrow(ax, 4.6, 9.5, 2.9, 8.4, label="いいえ", lfs=12)
        arrow(ax, 8.4, 10.1, 9.7, 8.6, label="はい", lfs=12)
        arrow(ax, 9.7, 7.45, 9.7, 6.75)
        arrow(ax, 8.4, 5.7, 4.1, 4.05)
        arrow(ax, 9.4, 5.35, 7.2, 4.05)
        arrow(ax, 10.6, 5.35, 11.1, 4.05)
        arrow(ax, 2.65, 7.0, 2.55, 4.05)
        for bx in (2.55, 7.0, 11.1):
            arrow(ax, bx, 2.8, bx, 1.5)

        ax.set_title("EMI 対策部品の選定フロー — 最初の分岐は「アナログ信号を扱うか?」",
                     fontsize=14)
        fig.tight_layout()
        fig.savefig(OUT / "01-flowchart.png", dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------- fig 02
def fig02():
    # Coverage map: a conceptual diagram (part names / band labels are prose),
    # so it carries Japanese labels like the flowcharts. Fonts bumped for Zenn.
    with plt.rc_context({"font.family": "IPAGothic", "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=(9.4, 4.9))
        rows = [
            ("LC（π 型）フィルタ", 1e3, 1e6, GREEN),
            ("大容量電解（バルク）", 1e3, 1e5, PURPLE),
            ("MLCC シャント", 1e5, 5e7, BLUE),
            ("三端子コンデンサ", 1e7, 1e9, ORANGE),
            ("ビーズ（R 帯域）", 3e7, 1e9, RED),
        ]
        for i, (name, f1, f2, color) in enumerate(rows):
            y = len(rows) - 1 - i
            ax.plot([f1, f2], [y, y], color=color, lw=12, solid_capstyle="round", alpha=0.85)
            ax.text(f1 * 0.7, y, name, ha="right", va="center", fontsize=12)
        # band shading
        for x1, x2, label, col in (
            (1e3, 1e5, "低域: スイッチング基本波・PWM", "#eef6ee"),
            (1e5, 5e7, "中域: 高調波・クロック", "#eef2fb"),
            (5e7, 1e9, "高域: クロック高調波・放射", "#fbeeee"),
        ):
            ax.axvspan(x1, x2, color=col, zorder=0)
            ax.text(np.sqrt(x1 * x2), 4.7, label, ha="center", fontsize=11, color="#555")
        ax.set_xscale("log")
        ax.set_xlim(5e2, 2e9)
        ax.set_ylim(-0.7, 5.2)
        ax.set_yticks([])
        ax.set_xlabel("周波数 [Hz]", fontsize=12.5)
        ax.tick_params(axis="x", labelsize=11)
        ax.set_title("部品ごとに効く周波数帯 — 横棒がその部品の守備範囲", fontsize=14)
        ax.grid(True, axis="x", which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "02-frequency-map.png", dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------- fig 03
def fig03():
    filt = LCFilterModel(l=22e-6, c=100e-6, dcr=0.05, esr=0.0, r_src=0.1)
    f = np.logspace(2, 7, 4001)
    gain = filt.gain_db(f)
    att_1m = float(filt.attenuation_db(np.array([1e6]))[0])
    fpk, gpk = peak_in_band(f, gain, filt.fc / 6, filt.fc * 6)

    # Curve only: the topology now lives in its own richer schematic (fig03b),
    # so this figure shows just the transfer characteristic. Japanese labels,
    # bigger fonts (single narrow panel reads well on Zenn).
    with plt.rc_context({"font.family": "IPAGothic", "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=(7.8, 4.5))
        ax.semilogx(f, gain, color=BLUE, lw=2.6)
        ax.axhline(0, color=GRAY, lw=0.9)
        ax.axvline(filt.fc, color="#bbb", ls="--", lw=0.9)
        ax.annotate(f"共振ピーク +{gpk:.0f} dB", xy=(fpk, gpk),
                    xytext=(fpk * 2.6, 19), fontsize=11.5, color="#444",
                    arrowprops=dict(arrowstyle="->", color="#888"))
        ax.annotate(f"fc ≒ {filt.fc / 1e3:.1f} kHz", xy=(filt.fc, -22),
                    xytext=(filt.fc * 1.7, -10), fontsize=12, color="#444",
                    arrowprops=dict(arrowstyle="->", color="#888"))
        ax.plot([1e6], [-att_1m], "o", color=RED)
        ax.annotate(f"1MHz: ≒ −{att_1m:.0f} dB", xy=(1e6, -att_1m),
                    xytext=(1.3e4, -att_1m + 20), fontsize=12.5, color=RED,
                    arrowprops=dict(arrowstyle="->", color=RED))
        ax.set_xlabel("周波数 [Hz]", fontsize=13)
        ax.set_ylabel("伝達 Vout/Vin [dB]（下ほど静か）", fontsize=12)
        ax.set_title("π 型 LC フィルタの伝達特性: 40dB/dec で落ちる", fontsize=13.5)
        ax.set_ylim(-120, 25)
        ax.tick_params(labelsize=11)
        grid(ax)
        fig.tight_layout()
        fig.savefig(OUT / "03-pi-filter.png", dpi=140)
        plt.close(fig)
    summary.append(f"pi filter (22uH+100uF): fc {filt.fc / 1e3:.2f} kHz, "
                   f"att@1MHz {att_1m:.1f} dB, peak +{gpk:.1f} dB")


# ------------------------------------------------------------- fig 03b (schem)
def fig03b():
    """Rich π-filter schematic (replaces the article's ASCII-art version)."""

    def ground(ax, x, y):
        ax.plot([x, x], [y, y - 0.16], color="#333", lw=1.8)
        for i, hw in enumerate((0.34, 0.20, 0.09)):
            yy = y - 0.16 - i * 0.12
            ax.plot([x - hw, x + hw], [yy, yy], color="#333", lw=1.8)

    def shunt_cap(ax, x, y_rail, color, label, polarized=False, fs=11):
        top, gap = 3.5, 0.26
        ax.plot([x, x], [y_rail, top + gap / 2], color="#333", lw=1.9)   # down
        ax.plot([x - 0.42, x + 0.42], [top + gap / 2] * 2, color=color, lw=3.4)
        ax.plot([x - 0.42, x + 0.42], [top - gap / 2] * 2, color=color, lw=3.4)
        if polarized:
            ax.text(x - 0.62, top + gap / 2 + 0.04, "+", color=color,
                    fontsize=13, ha="center", va="center")
        ax.plot([x, x], [top - gap / 2, 2.35], color="#333", lw=1.9)     # to gnd
        ground(ax, x, 2.35)
        ax.text(x, 1.5, label, ha="center", va="top", fontsize=fs, color="#222")

    def inductor(ax, x1, x2, y, color, n=4, lw=2.7):
        r = (x2 - x1) / (2 * n)
        th = np.linspace(np.pi, 0, 80)
        for k in range(n):
            cx = x1 + r + 2 * r * k
            ax.plot(cx + r * np.cos(th), y + r * np.sin(th),
                    color=color, lw=lw, solid_capstyle="round")

    with plt.rc_context({"font.family": "IPAGothic", "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 6.4)
        ax.axis("off")
        yr = 4.7
        # signal rail with a series choke in the middle
        ax.plot([1.2, 4.0], [yr, yr], color="#333", lw=2.4)
        inductor(ax, 4.0, 6.2, yr, GREEN, n=4)
        ax.plot([6.2, 11.6], [yr, yr], color="#333", lw=2.4)
        ax.text(5.1, yr + 0.52, "チョーク 10〜47μH", ha="center", fontsize=12, color=GREEN)
        # input / output terminals
        ax.annotate("", xy=(1.2, yr), xytext=(0.35, yr),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.0))
        ax.text(0.2, yr + 0.6, "入力（サーボ側）", ha="left", fontsize=12, color="#222")
        ax.text(0.2, yr + 0.22, "ノイズ →", ha="left", fontsize=11, color=RED)
        ax.annotate("", xy=(12.55, yr), xytext=(11.6, yr),
                    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.0))
        ax.text(12.75, yr + 0.6, "出力", ha="right", fontsize=12, color="#222")
        ax.text(12.75, yr + 0.22, "（アナログ回路へ）", ha="right", fontsize=11, color=GREEN)
        # shunt legs: input MLCC, output electrolytic (polarized), output MLCC
        shunt_cap(ax, 2.6, yr, BLUE, "入口 MLCC\n第一波を GND へ", fs=11)
        shunt_cap(ax, 7.8, yr, PURPLE, "電解 100μF\nLC を作り ESR がダンパ",
                  polarized=True, fs=11)
        shunt_cap(ax, 10.4, yr, BLUE, "出口 MLCC\n高域を仕上げ", fs=11)
        ax.set_title("π 型 LC フィルタの構成 — 入口 MLCC・チョーク・出口に電解＋MLCC",
                     fontsize=14)
        fig.tight_layout()
        fig.savefig(OUT / "03b-pi-filter-schematic.png", dpi=140)
        plt.close(fig)


# ---------------------------------------------------------------- fig 04
def fig04():
    # a *downsized* filter whose fc lands in the PWM carrier band (10-20 kHz)
    f = np.logspace(3, 6, 4001)
    sharp = LCFilterModel(l=10e-6, c=10e-6, dcr=0.05, esr=0.01, r_src=0.1)
    damped = LCFilterModel(l=10e-6, c=10e-6, dcr=0.05, esr=1.0, r_src=0.1)
    g_sharp, g_damped = sharp.gain_db(f), damped.gain_db(f)
    _, pk_s = peak_in_band(f, g_sharp, sharp.fc / 4, sharp.fc * 4)
    _, pk_d = peak_in_band(f, g_damped, damped.fc / 4, damped.fc * 4)

    with plt.rc_context({"font.family": "IPAGothic", "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.axvspan(10e3, 20e3, color="#f3e9c9", zorder=0)
        # label sits in the top headroom of the band (clear of the frame and
        # of the red +16 dB peak that lands inside the carrier band)
        ax.text(np.sqrt(10e3 * 20e3), 23, "PWM キャリア\n10〜20kHz", ha="center",
                va="center", fontsize=11, color="#9a7d1f")
        ax.semilogx(f, g_sharp, color=RED, lw=2.4,
                    label=f"低 ESR のみ: +{pk_s:.0f}dB のピーク（キャリアに直撃）")
        ax.semilogx(f, g_damped, color=BLUE, lw=2.4,
                    label=f"電解の ESR でダンピング: +{pk_d:.0f}dB")
        ax.axhline(0, color=GRAY, lw=0.9)
        ax.set_xlabel("周波数 [Hz]", fontsize=12.5)
        ax.set_ylabel("伝達 Vout/Vin [dB]（>0 は増幅）", fontsize=11.5)
        ax.set_title("部品を小さくすると共振ピークが PWM キャリア帯に乗る", fontsize=13.5)
        ax.set_xlim(1e3, 1e6)
        ax.set_ylim(-60, 27)
        ax.tick_params(labelsize=11)
        ax.legend(loc="lower left", fontsize=10.5)
        grid(ax)
        fig.tight_layout()
        fig.savefig(OUT / "04-lc-damping.png", dpi=140)
        plt.close(fig)
    summary.append(f"downsized LC (10uH+10uF, fc {sharp.fc/1e3:.1f} kHz): "
                   f"low-ESR +{pk_s:.1f} dB vs electrolytic +{pk_d:.1f} dB")


# ---------------------------------------------------------------- fig 05
def fig05():
    """Illustrative MLCC DC-bias derating: retained capacitance vs applied DC."""
    v = np.linspace(0, 50, 400)

    def retained(v, v_rated, n=2.2):
        # smooth illustrative roll-off; ~half gone near the rating for class-2
        return 100.0 / (1.0 + (v / (0.55 * v_rated)) ** n)

    with plt.rc_context({"font.family": "IPAGothic", "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=(7.9, 4.7))
        for v_rated, color, name, ytext in (
            (25.0, RED, "25V 定格", 28.0), (50.0, BLUE, "50V 定格", 64.0)
        ):
            ax.plot(v, retained(v, v_rated), color=color, lw=2.4, label=name)
            y24 = retained(np.array([24.0]), v_rated)[0]
            ax.plot([24.0], [y24], "o", color=color)
            ax.annotate(f"≒{y24:.0f}% @ 24V", xy=(24.0, y24), xytext=(27.5, ytext),
                        fontsize=12, color=color,
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.9))
        ax.axvline(24.0, color=GRAY, ls="--", lw=0.9)
        ax.text(23.2, 6, "24V ライン", color="#555", fontsize=11, ha="right")
        ax.set_xlabel("かける直流電圧 [V]", fontsize=12.5)
        ax.set_ylabel("残る容量 [%]（イメージ）", fontsize=12)
        ax.set_title("高誘電率系 MLCC は DC バイアスで容量が減る — 定格に余裕を", fontsize=12.5)
        ax.set_ylim(0, 110)
        ax.set_xlim(0, 50)
        ax.tick_params(labelsize=11)
        ax.legend(loc="upper right", fontsize=11.5)
        grid(ax)
        fig.tight_layout()
        fig.savefig(OUT / "05-mlcc-dc-bias.png", dpi=140)
        plt.close(fig)


# ---------------------------------------------------------------- fig 06
def fig06():
    f = np.logspace(4, 9, 6001)
    part = CapacitorModel(c=0.1e-6, esl=0.5e-9, esr=0.02)        # part only
    wired = CapacitorModel(c=0.1e-6, esl=5.5e-9, esr=0.02)       # + 5 mm wiring
    with plt.rc_context({"font.family": "IPAGothic", "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=(7.9, 4.8))
        ax.loglog(f, np.abs(part.impedance(f)), color=BLUE, lw=2.4,
                  label=f"ピン直近（ESL 0.5nH）: SRF {part.srf/1e6:.1f}MHz")
        ax.loglog(f, np.abs(wired.impedance(f)), color=RED, lw=2.4,
                  label=f"5mm 離す（ESL 5.5nH）: SRF {wired.srf/1e6:.1f}MHz")
        z_part = abs(part.impedance(np.array([100e6]))[0])
        z_wired = abs(wired.impedance(np.array([100e6]))[0])
        ax.axvline(100e6, color="#bbb", ls=":", lw=0.9)
        ax.annotate(f"100MHz で\n|Z| が ×{z_wired / z_part:.1f} 悪化",
                    xy=(100e6, z_wired), xytext=(1.2e7, z_wired * 3.2),
                    fontsize=11.5, color="#444",
                    arrowprops=dict(arrowstyle="->", color="#888"))
        ax.set_xlabel("周波数 [Hz]", fontsize=12.5)
        ax.set_ylabel("|Z| [Ω]", fontsize=12.5)
        ax.set_title("配線 5mm で SRF が落ち、高域が効かなくなる", fontsize=13)
        ax.set_ylim(0.01, 1e2)
        ax.tick_params(labelsize=11)
        ax.legend(loc="upper center", fontsize=11)
        grid(ax)
        fig.tight_layout()
        fig.savefig(OUT / "06-wiring-inductance.png", dpi=140)
        plt.close(fig)
    summary.append(f"wiring: SRF {part.srf/1e6:.1f} -> {wired.srf/1e6:.1f} MHz, "
                   f"|Z|@100MHz x{z_wired/z_part:.1f}")


# ---------------------------------------------------------------- fig 07
def fig07():
    f, s = read_touchstone(SAMPLES / "sample_bead.s2p")
    z = series_z_from_s21(s["s21"], s["z0"])
    z100 = abs(z[int(np.argmin(np.abs(f - 100e6)))])
    with plt.rc_context({"font.family": "IPAGothic", "axes.unicode_minus": False}):
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0),
                                 gridspec_kw={"width_ratios": [1, 1.15]})
        draw_smith(axes[0], s["s11"], f, marks=(1e6, 1e8, 1e9), mark_fs=11)
        axes[0].set_title("スミスチャート（S11）— 丸くて見慣れないほう", fontsize=13)
        axes[1].loglog(f, np.abs(z), color=BLUE, lw=2.6, label="|Z|（S21 から計算）")
        axes[1].axvline(100e6, color="#bbb", ls=":", lw=0.9)
        axes[1].annotate(f"≒{z100:.0f}Ω\n@ 100MHz", xy=(100e6, z100),
                         xytext=(2.5e6, z100 * 2.1), fontsize=12, color="#444",
                         arrowprops=dict(arrowstyle="->", color="#888"))
        axes[1].set_xlabel("周波数 [Hz]", fontsize=13)
        axes[1].set_ylabel("|Z| [Ω]", fontsize=13)
        axes[1].set_title("周波数 × |Z| — 同じデータ、ひと目で読める", fontsize=13)
        axes[1].tick_params(labelsize=11)
        grid(axes[1])
        axes[1].legend(loc="upper left", fontsize=11)
        fig.suptitle("同じ .s2p を 2 通りで見る — EMI には右が速い", fontsize=14.5)
        fig.tight_layout()
        fig.savefig(OUT / "07-smith-vs-simple.png", dpi=140)
        plt.close(fig)
    summary.append(f"sample_bead.s2p: |Z|@100MHz {z100:.1f} ohm (read back via S21)")


# ---------------------------------------------------------------- fig 08
def fig08():
    with plt.rc_context({"font.family": "IPAGothic", "axes.unicode_minus": False}):
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.6),
                                 gridspec_kw={"width_ratios": [1, 1.3]})
        # left: the spine of the flow
        ax = axes[0]
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 12)
        ax.axis("off")
        spine = [
            (10.8, "アナログ信号を扱う?", FILL_Y),
            (8.4, "ノイズの周波数は?", FILL_Y),
            (6.0, "その帯域の部品を選ぶ", FILL_G),
            (3.6, "直流条件を確かめる（DC バイアス・直流重畳）", FILL_R),
            (1.2, "配線とレイアウトを確認（約 1nH/mm）", FILL_GR),
        ]
        for y, text, fc in spine:
            box(ax, 3.0, y, 5.6, 1.2, text, fc, fs=12)
        for y0, y1 in ((10.2, 9.0), (7.8, 6.6), (5.4, 4.2), (3.0, 1.8)):
            arrow(ax, 3.0, y0, 3.0, y1)
        ax.set_title("選定の背骨（フロー）", fontsize=13)

        # right: the 4-point datasheet check
        ax = axes[1]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis("off")
        checks = [
            ("1. 周波数",
             "抑えたい周波数で本来の働きをするか。\nコンデンサは SRF より下、ビーズは目覚めの周波数より上"),
            ("2. 相手のインピーダンス",
             "効き目は相手との比。シャントは相手が高インピーダンス\nなほど、直列は相手が低インピーダンスなほど効く"),
            ("3. 直流条件",
             "カタログ値は電流・電圧ゼロのとき。MLCC は DC バイアスで\n容量が減り、ビーズ・チョークは直流重畳で飽和"),
            ("4. 組み合わせ",
             "共振はどこに立つか。反共振・共振ピークを計算で確かめ、\n必要なら ESR でダンピングする"),
        ]
        colors = [BLUE, GREEN, RED, PURPLE]
        for i, (head, body) in enumerate(checks):
            y = 10.6 - i * 2.7
            box(ax, 5.0, y, 9.4, 2.1, "", "#fbfbfb", colors[i])
            ax.text(0.5, y + 0.58, head, fontsize=13, color=colors[i],
                    weight="bold", ha="left", va="center")
            ax.text(0.5, y - 0.5, body, fontsize=11, color="#333",
                    ha="left", va="center")
        ax.set_title("データシート 4 点チェック", fontsize=13)

        fig.suptitle("羅針盤 — 1 つの分岐 ＋ 4 点チェック", fontsize=14.5)
        fig.tight_layout()
        fig.savefig(OUT / "08-summary-flowchart.png", dpi=140)
        plt.close(fig)


def main() -> None:
    make_sample_s2p()
    for fn in (fig01, fig02, fig03, fig03b, fig04, fig05, fig06, fig07, fig08):
        fn()
        print(f"done: {fn.__name__}")
    text = "\n".join(summary)
    (OUT / "summary3.txt").write_text(text + "\n")
    print("--- summary ---")
    print(text)


if __name__ == "__main__":
    main()
