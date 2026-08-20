"""Render the node-failure sweep as a grid heatmap of the EP expert layout.

One cell per node, laid out in the same order the experts are partitioned, so the
picture is the cluster: cell ``k`` is the contiguous expert block node ``k`` owns
in every MoE layer. Each cell carries its expert range, its accuracy, and its
delta against the measured baseline.

Color encodes the **delta**, diverging around the baseline, because that is the
polarity question the sweep asks: did losing this node's experts help, hurt, or
do nothing? A single-hue sequential ramp -- the obvious first choice for a
heatmap -- paints "1 point above baseline" and "1 point below" as almost the same
shade, which is exactly the distinction the reader is here for. Pass
``--scale sequential`` for a one-hue magnitude ramp instead.

The scale is symmetric by construction (``vmax = -vmin = max |delta|``) so the
neutral midpoint always lands exactly on the baseline; an asymmetric diverging
scale would put the "nothing happened" color somewhere other than "nothing
happened".

Usage::

    python experiments/node-failure/heatmap.py artifacts/glm-node-failure/bfcl \\
        --out fig/glm45-air-bfcl-node-failure --model glm-4.5-air

Writes ``<out>.png`` and ``<out>-dark.png`` (both modes are stepped for their own
surface, not flipped), plus ``<out>.csv`` as the table view.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Palette, by role. Light and dark are each stepped for their own surface.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "page": "#f9f9f7",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "ink_muted": "#898781",
        "ink_inverse": "#ffffff",
        "border": (0.043, 0.043, 0.043, 0.10),
        "neutral": "#f0efec",   # diverging midpoint: "nothing happened"
        "cool": "#184f95",      # above baseline  (blue ramp step 600)
        "warm": "#8f2422",      # below baseline  (red arm, matched in lightness)
        # Sequential ramp steps 100 -> 400. Deliberately NOT the full 100->700
        # range: these cells carry text, and a ramp spanning light to dark must
        # cross the point where black and white ink are equally bad (~4.5:1
        # either way, measured). Step 400 is the darkest step where black ink
        # still clears 4.5:1 (5.41), so the whole ramp takes one ink.
        "seq_lo": "#cde2fb",
        "seq_hi": "#3987e5",
    },
    "dark": {
        "surface": "#1a1a19",
        "page": "#0d0d0d",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "ink_muted": "#898781",
        "ink_inverse": "#0b0b0b",
        "border": (1.0, 1.0, 1.0, 0.10),
        "neutral": "#383835",
        "cool": "#86b6ef",
        "warm": "#e08a88",
        # Mirror of the light-mode reasoning, on the other side of the
        # crossover: steps 700 -> 500, the lightest step where white ink still
        # clears 4.5:1 (5.39). Narrower than the light ramp, which is the price
        # of keeping one ink; the diverging scale (the default) does not pay it.
        "seq_lo": "#0d366b",
        "seq_hi": "#256abf",
    },
}

SCORE_KEYS = ("bfcl", "math_500")
BENCHMARK_LABELS = {
    "bfcl": "BFCL non-live accuracy",
    "math_500": "MATH-500 accuracy",
}


def load(results_dir: pathlib.Path) -> tuple[str, float, list[float], dict[int, dict]]:
    """Return (benchmark, baseline mean, baseline scores, {node: cell})."""
    baselines, masked = [], {}
    benchmark = None
    for path in sorted(results_dir.glob("**/cell.json")):
        record = json.loads(path.read_text())
        key = record.get("benchmark") or next((k for k in SCORE_KEYS if k in record), None)
        if key is None:
            continue
        benchmark = benchmark or key
        block = record[key]
        if record.get("mode") == "off":
            baselines.append(block["score"])
        else:
            masked[record["node_id"]] = {
                "score": block["score"],
                "num": block["num"],
                "mode": record.get("mode"),
            }
    if not baselines:
        raise SystemExit(
            f"no baseline cells (mode=off) under {results_dir} -- the deltas would "
            "have nothing to be measured against."
        )
    if not masked:
        raise SystemExit(f"no masked cells under {results_dir}")
    return benchmark, statistics.fmean(baselines), baselines, masked


def build_cmap(theme: dict, scale: str) -> mcolors.LinearSegmentedColormap:
    """Diverging (delta polarity) or sequential (raw magnitude) ramp.

    The diverging ramp is two hues around a neutral gray. Low end is the warm
    pole so that "below baseline" reads as the alarming direction; a rainbow or a
    hue at the midpoint would both destroy the "nothing happened" reading.
    """
    if scale == "sequential":
        return mcolors.LinearSegmentedColormap.from_list(
            "nf_seq", [theme["seq_lo"], theme["seq_hi"]]
        )
    return mcolors.LinearSegmentedColormap.from_list(
        "nf_div", [theme["warm"], theme["neutral"], theme["cool"]]
    )


def _relative_luminance(rgb: tuple[float, float, float]) -> float:
    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(fg: tuple, bg: tuple) -> float:
    """WCAG contrast ratio. Computed, not eyeballed -- see pick_ink."""
    l1, l2 = _relative_luminance(fg), _relative_luminance(bg)
    lo, hi = sorted((l1, l2))
    return (hi + 0.05) / (lo + 0.05)


def pick_ink(fill: tuple, theme: dict) -> tuple[str, float]:
    """Choose the cell's text color by measured contrast against its own fill.

    A fixed ink color fails at one end of any ramp that spans light to dark, and
    which end depends on the data -- so it is decided per cell and the worst case
    is asserted below rather than assumed.
    """
    candidates = [theme["ink"], theme["ink_inverse"]]
    scored = [(contrast(mcolors.to_rgb(c), fill), c) for c in candidates]
    ratio, color = max(scored)
    return color, ratio


def render(
    out_path: pathlib.Path,
    mode: str,
    *,
    benchmark: str,
    base_mean: float,
    baselines: list[float],
    masked: dict[int, dict],
    num_experts: int,
    num_nodes: int,
    cols: int,
    scale: str,
    model: str,
    failure_mode: str,
) -> float:
    theme = THEMES[mode]
    cmap = build_cmap(theme, scale)
    per_node = num_experts // num_nodes
    rows = math.ceil(num_nodes / cols)

    deltas = {k: v["score"] - base_mean for k, v in masked.items()}
    span = max(abs(d) for d in deltas.values()) or 1e-9
    if scale == "sequential":
        lo, hi = min(v["score"] for v in masked.values()), max(
            v["score"] for v in masked.values()
        )
        pad = 0.1 * (hi - lo or 1e-9)
        norm = mcolors.Normalize(vmin=(lo - pad) * 100, vmax=(hi + pad) * 100)
        value_of = lambda node: masked[node]["score"] * 100  # noqa: E731
    else:
        # Symmetric, so the neutral midpoint lands exactly on the baseline.
        norm = mcolors.Normalize(vmin=-span * 100, vmax=span * 100)
        value_of = lambda node: deltas[node] * 100  # noqa: E731

    n = masked[next(iter(masked))]["num"]
    # The floor below which no single cell's delta means anything.
    se = math.sqrt(base_mean * (1 - base_mean) / n) * 100 if n else float("nan")

    fig_w, fig_h = 1.42 * cols + 0.9, 1.30 * rows + 2.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(theme["page"])
    ax.set_facecolor(theme["surface"])
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()
    ax.axis("off")

    worst_contrast = math.inf
    pad = 0.045  # surface gap between fills, so adjacent cells never touch
    for node in range(num_nodes):
        if node not in masked:
            continue
        r, c = divmod(node, cols)
        fill = cmap(norm(value_of(node)))
        ink, ratio = pick_ink(fill, theme)
        worst_contrast = min(worst_contrast, ratio)

        ax.add_patch(
            FancyBboxPatch(
                (c + pad, r + pad),
                1 - 2 * pad,
                1 - 2 * pad,
                boxstyle="round,pad=0,rounding_size=0.06",
                linewidth=0.8,
                edgecolor=theme["border"],
                facecolor=fill,
                mutation_aspect=1,
            )
        )
        lo_e = node * per_node
        ax.text(
            c + 0.5, r + 0.22,
            f"{lo_e}-{lo_e + per_node - 1}",
            ha="center", va="center", fontsize=7.5, color=ink, alpha=0.75,
        )
        ax.text(
            c + 0.5, r + 0.50,
            f"{masked[node]['score'] * 100:.1f}",
            ha="center", va="center", fontsize=15, fontweight="bold", color=ink,
        )
        delta = deltas[node] * 100
        ax.text(
            c + 0.5, r + 0.76,
            f"{delta:+.1f}",
            ha="center", va="center", fontsize=8.5, color=ink, alpha=0.85,
        )

    bench_label = BENCHMARK_LABELS.get(benchmark, f"{benchmark} accuracy")
    fig.suptitle(
        f"{bench_label} under single-node failure — {failure_mode}",
        x=0.035, y=0.975, ha="left", fontsize=13.5, fontweight="bold",
        color=theme["ink"],
    )
    fig.text(
        0.035, 0.930,
        f"{model}, EP={num_nodes} ({per_node} experts/node), n={n}.  "
        f"Baseline {base_mean * 100:.1f} "
        f"(mean of {len(baselines)}, SD {statistics.stdev(baselines) * 100:.2f} pts)"
        if len(baselines) > 1
        else f"{model}, EP={num_nodes} ({per_node} experts/node), n={n}.  "
        f"Baseline {base_mean * 100:.1f}",
        ha="left", fontsize=8.5, color=theme["ink_secondary"],
    )
    fig.text(
        0.035, 0.902,
        "cell = expert group, accuracy, Δ vs baseline   ·   "
        f"±1σ sampling noise = {se:.2f} pts",
        ha="left", fontsize=8.5, color=theme["ink_muted"],
    )

    cbar_label = (
        "Δ vs baseline (accuracy points)"
        if scale == "diverging"
        else f"{bench_label} (%)"
    )
    # The bar sits low; its reference markers are annotated ABOVE it and its
    # tick labels and axis label below, so the two label rows never collide.
    bar_y, bar_h = 0.085, 0.026
    cax = fig.add_axes([0.035, bar_y, 0.93, bar_h])
    bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax, orientation="horizontal",
    )
    bar.set_label(cbar_label, fontsize=8.5, color=theme["ink_secondary"], labelpad=4)
    bar.outline.set_edgecolor(theme["border"])
    bar.ax.tick_params(labelsize=7.5, colors=theme["ink_muted"], length=2)

    def annotate(value: float, label: str, weight: str, color: str, size: float) -> None:
        fig.text(
            0.035 + 0.93 * norm(value), bar_y + bar_h + 0.012, label,
            ha="center", va="bottom", fontsize=size, color=color, fontweight=weight,
        )

    if scale == "diverging":
        # Mark the noise floor, so a reader can see at a glance which cells are
        # even candidates for being real rather than scatter.
        for x in (-se, se):
            bar.ax.axvline(x, color=theme["ink"], linewidth=1.0, alpha=0.5)
            annotate(x, f"{'+' if x > 0 else '−'}1σ", "normal", theme["ink_muted"], 7)
        bar.ax.axvline(0, color=theme["ink"], linewidth=1.4)
        annotate(0, "baseline", "bold", theme["ink_secondary"], 7.5)
    else:
        bar.ax.axvline(base_mean * 100, color=theme["ink"], linewidth=1.4)
        annotate(base_mean * 100, "baseline", "bold", theme["ink_secondary"], 7.5)

    fig.subplots_adjust(left=0.035, right=0.965, top=0.875, bottom=0.205)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return worst_contrast


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_dir", type=pathlib.Path)
    p.add_argument("--out", type=pathlib.Path, required=True, help="path stem, no extension")
    p.add_argument("--model", default="model", help="label for the subtitle")
    p.add_argument("--num-experts", type=int, default=128)
    p.add_argument("--num-nodes", type=int, default=32)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument(
        "--scale",
        default="diverging",
        choices=["diverging", "sequential"],
        help="`diverging` colors the delta around the baseline (default); "
        "`sequential` colors raw accuracy on a one-hue ramp.",
    )
    args = p.parse_args()

    benchmark, base_mean, baselines, masked = load(args.results_dir)
    modes = {v["mode"] for v in masked.values()}
    failure_mode = modes.pop() if len(modes) == 1 else "+".join(sorted(modes))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    worst = math.inf
    for mode, suffix in (("light", ""), ("dark", "-dark")):
        out = args.out.with_name(args.out.name + suffix).with_suffix(".png")
        worst = min(
            worst,
            render(
                out, mode,
                benchmark=benchmark, base_mean=base_mean, baselines=baselines,
                masked=masked, num_experts=args.num_experts,
                num_nodes=args.num_nodes, cols=args.cols, scale=args.scale,
                model=args.model, failure_mode=failure_mode,
            ),
        )
        print(f"wrote {out}")

    # The table view the figure is required to have an equivalent of.
    csv_path = args.out.with_suffix(".csv")
    per_node = args.num_experts // args.num_nodes
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node", "experts", "accuracy", "delta_pts", "delta_rel_pct", "n"])
        for node in sorted(masked):
            lo = node * per_node
            score = masked[node]["score"]
            writer.writerow([
                node, f"{lo}-{lo + per_node - 1}", round(score, 6),
                round((score - base_mean) * 100, 3),
                round((score - base_mean) / base_mean * 100, 3),
                masked[node]["num"],
            ])
    print(f"wrote {csv_path}")

    # Contrast is computable, so it is checked rather than assumed. 4.5:1 is the
    # WCAG AA body-text floor; the cell values are the whole point of the figure.
    print(f"worst measured text contrast on any cell: {worst:.2f}:1")
    if worst < 4.5:
        print("FAIL: a cell's value text is below 4.5:1 on its own fill", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
