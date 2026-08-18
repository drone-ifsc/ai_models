#!/usr/bin/env python3
"""Regenerate benchmark charts from benchmark_results.csv (dynamic — any set of
models). Called automatically by bench.py after new rows are added; can also be
run standalone. Writes 01..05 PNGs next to the CSV.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "benchmark_results.csv")

# Okabe-Ito, CVD-safe; cycled across however many models are present.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
           "#56B4E9", "#F0E442", "#999999"]
INK, MUTED, GRID = "#1a1a1a", "#666666", "#dddddd"

plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def load():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows.append({"name": r["name"], "threads": int(r["threads"]),
                         "input_px": float(r["input_px"]), "dtype": r["dtype"],
                         "fps": float(r["fps"]),
                         "cpu": float(r["cpu_pct_across_cores"]),
                         "ram": float(r["peak_ram_mb"])})
    return rows


def main():
    rows = load()
    names = sorted({r["name"] for r in rows})
    color = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}
    threads_all = sorted({r["threads"] for r in rows})

    # best (max-fps) row per model + derived metrics
    best = {}
    for n in names:
        rs = [r for r in rows if r["name"] == n]
        b = max(rs, key=lambda r: r["fps"])
        best[n] = {"fps": b["fps"], "thr": b["threads"], "cpu": b["cpu"],
                   "ram": b["ram"], "inp": b["input_px"], "dtype": b["dtype"],
                   "ms": 1000.0 / b["fps"] if b["fps"] else 0,
                   "mps": b["fps"] * b["input_px"] ** 2 / 1e6}
    order = sorted(names, key=lambda n: best[n]["fps"])  # ascending -> biggest on top in barh

    def barlabels(ax, bars, fmt, pad=0.01):
        xmax = max((b.get_width() for b in bars), default=1)
        for b in bars:
            ax.text(b.get_width() + pad * xmax, b.get_y() + b.get_height() / 2,
                    fmt.format(b.get_width()), va="center", ha="left",
                    color=INK, fontsize=10)

    def hbar(metric, title, subtitle, fmt, fname, note=None):
        fig, ax = plt.subplots(figsize=(8.6, max(2.6, 0.7 * len(order) + 1.2)))
        ys = list(range(len(order)))
        vals = [best[n][metric] for n in order]
        bars = ax.barh(ys, vals, color=[color[n] for n in order], height=0.62, zorder=3)
        labels = [f"{n}\n({int(best[n]['inp'])}px, {best[n]['thr']} thr, {best[n]['dtype']})"
                  for n in order]
        ax.set_yticks(ys); ax.set_yticklabels(labels)
        ax.set_title(title, fontweight="bold", color=INK, loc="left", pad=14)
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=MUTED, fontsize=9.5)
        ax.set_xlim(0, (max(vals) if vals else 1) * 1.18)
        ax.xaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        barlabels(ax, bars, fmt)
        if note:
            fig.text(0.01, -0.02, note, color=MUTED, fontsize=8.5, ha="left")
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, fname), dpi=140, bbox_inches="tight")
        plt.close(fig)

    hbar("fps", "Max FPS por modelo", "Maior FPS no melhor thread setting.",
         "{:.1f}", "01_fps.png",
         note="FPS nao e apples-to-apples se o input (px) difere entre modelos.")
    hbar("ms", "Latencia por inferencia (ms)", "1000 / FPS.", "{:.0f} ms", "02_latency.png")
    hbar("ram", "Pico de RAM do processo (MB)", "RSS maximo (inclui runtime).",
         "{:.0f} MB", "03_ram.png")
    hbar("mps", "Throughput normalizado (MPix/s)",
         "FPS x pixels de entrada -> custo de compute independente do input.",
         "{:.2f}", "04_throughput.png")

    # FPS vs threads (grouped) — only if more than one thread count exists
    if len(threads_all) > 1:
        fig, ax = plt.subplots(figsize=(8.6, 4.0))
        width = 0.8 / max(1, len(names))
        xs = list(range(len(threads_all)))
        for i, n in enumerate(names):
            fps_t = []
            for t in threads_all:
                m = [r["fps"] for r in rows if r["name"] == n and r["threads"] == t]
                fps_t.append(m[0] if m else 0)
            off = (i - (len(names) - 1) / 2) * width
            bars = ax.bar([x + off for x in xs], fps_t, width * 0.9,
                          label=n, color=color[n], zorder=3)
            for b, v in zip(bars, fps_t):
                if v:
                    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                            ha="center", va="bottom", color=INK, fontsize=8)
        ax.set_xticks(xs); ax.set_xticklabels([f"{t} threads" for t in threads_all])
        ax.set_ylabel("FPS")
        ax.set_title("Escalonamento com threads", fontweight="bold", color=INK, loc="left", pad=14)
        ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="upper left",
                  bbox_to_anchor=(0.0, -0.12), ncol=min(4, len(names)))
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "05_fps_vs_threads.png"), dpi=140, bbox_inches="tight")
        plt.close(fig)

    print("charts written to", HERE)
    for f in sorted(os.listdir(HERE)):
        if f.endswith(".png"):
            print("  ", f)


if __name__ == "__main__":
    main()
