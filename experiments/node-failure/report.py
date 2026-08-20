"""Summarise a node-failure sweep: per-node delta against the measured baseline.

Reads every ``cell.json`` under a results tree and prints one row per masked
cell, sorted worst-first, with the delta expressed in units of the *measured*
baseline noise rather than only in points. That framing is the point of the
exercise: a 3.125% expert loss is a small perturbation, so "is this drop real?"
is a question about the noise floor, and a table of raw deltas invites reading
run-to-run scatter as a finding.

Works for either benchmark -- the cell records name their score under the
benchmark key (``math_500`` or ``bfcl``), so the reader takes whichever is
present.

Usage::

    python experiments/node-failure/report.py artifacts/glm-node-failure/bfcl
    python experiments/node-failure/report.py <dir> --csv summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys

# Keys a cell record may carry its score under, in the order tried.
SCORE_KEYS = ("bfcl", "math_500")


def load_cells(results_dir: pathlib.Path) -> list[dict]:
    """Load every cell record, annotated with its score, n and mode."""
    cells = []
    for path in sorted(results_dir.glob("**/cell.json")):
        record = json.loads(path.read_text())
        key = record.get("benchmark") or next(
            (k for k in SCORE_KEYS if k in record), None
        )
        if key is None or key not in record:
            print(f"skipping {path}: no recognised score key", file=sys.stderr)
            continue
        score_block = record[key]
        cells.append(
            {
                "path": path,
                "benchmark": key,
                "mode": record.get("mode"),
                "node_id": record.get("node_id"),
                "score": score_block.get("score"),
                "num": score_block.get("num"),
                "categories": score_block.get("categories") or {},
                "elapsed_s": record.get("elapsed_s"),
                "dead_slot_rate": (record.get("routing") or {}).get("dead_slot_rate"),
                "lost_gate_mass_frac": (record.get("routing") or {}).get(
                    "lost_gate_mass_frac"
                ),
                "calls": (record.get("routing") or {}).get("calls"),
            }
        )
    return cells


def binomial_se(p: float, n: int) -> float:
    """Sampling SE of a proportion, the floor below which no delta is meaningful."""
    if not n or p is None:
        return float("nan")
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=pathlib.Path)
    parser.add_argument("--csv", type=pathlib.Path, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="acceptance bar, in %% relative to baseline (default 5)",
    )
    args = parser.parse_args()

    cells = load_cells(args.results_dir)
    if not cells:
        raise SystemExit(f"no cell.json found under {args.results_dir}")

    baselines = [c for c in cells if c["mode"] == "off"]
    masked = [c for c in cells if c["mode"] != "off"]
    if not baselines:
        raise SystemExit(
            "no baseline cells (mode=off) -- without them the deltas have "
            "nothing to be measured against. Rerun with --baseline-repeats >= 2."
        )

    benchmark = cells[0]["benchmark"]
    n = baselines[0]["num"]
    base_scores = [c["score"] for c in baselines]
    base_mean = statistics.fmean(base_scores)
    # Two sources of spread, and they answer different questions. The observed
    # baseline SD is the honest empirical floor but is worthless at 2-3 repeats;
    # the binomial SE is the theoretical floor at this sample size. Report both
    # and use the larger, so a noisy harness cannot be mistaken for a signal.
    base_sd = statistics.stdev(base_scores) if len(base_scores) > 1 else float("nan")
    se = binomial_se(base_mean, n)
    noise = max(x for x in (base_sd, se) if not math.isnan(x))

    print(f"benchmark: {benchmark}   n = {n}   cells: {len(masked)} masked, "
          f"{len(baselines)} baseline")
    print(f"baseline: mean {base_mean:.4f}  " + "  ".join(f"{s:.4f}" for s in base_scores))
    print(
        f"noise floor: binomial SE {se * 100:.2f} pts, "
        f"observed baseline SD {base_sd * 100:.2f} pts "
        f"(n={len(base_scores)}) -> using {noise * 100:.2f} pts"
    )
    bar = args.threshold / 100.0 * base_mean
    print(
        f"acceptance bar: {args.threshold:.0f}% relative = {bar * 100:.2f} pts "
        f"= {bar / noise:.1f} sigma\n"
    )

    rows = []
    for cell in masked:
        delta = cell["score"] - base_mean
        rows.append(
            {
                "mode": cell["mode"],
                "node": cell["node_id"],
                "score": cell["score"],
                "delta_pts": delta * 100,
                "delta_rel_pct": (delta / base_mean * 100) if base_mean else float("nan"),
                "sigma": delta / noise if noise else float("nan"),
                "dead_slot_rate": cell["dead_slot_rate"],
                "lost_gate_mass_frac": cell["lost_gate_mass_frac"],
                "num": cell["num"],
                "elapsed_s": cell["elapsed_s"],
            }
        )
    rows.sort(key=lambda r: r["score"])

    header = (
        f"{'mode':<12} {'node':>4} {'score':>7} {'delta':>7} {'rel%':>7} "
        f"{'sigma':>6} {'dead_slot':>10} {'lost_gate':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['mode']:<12} {row['node']:>4} {row['score']:>7.4f} "
            f"{row['delta_pts']:>+7.2f} {row['delta_rel_pct']:>+7.2f} "
            f"{row['sigma']:>+6.1f} {(row['dead_slot_rate'] or 0):>10.4f} "
            f"{(row['lost_gate_mass_frac'] or 0):>10.4f}"
        )

    breaches = [r for r in rows if r["delta_rel_pct"] < -args.threshold]
    print()
    if breaches:
        print(
            f"{len(breaches)} cell(s) exceed the {args.threshold:.0f}% bar: "
            + ", ".join(f"{r['mode']}/node {r['node']}" for r in breaches)
        )
    else:
        worst = rows[0]
        print(
            f"no cell exceeds the {args.threshold:.0f}% bar. Worst is "
            f"{worst['mode']}/node {worst['node']} at {worst['delta_rel_pct']:+.2f}% "
            f"({worst['sigma']:+.1f} sigma) -- "
            + (
                "within the noise floor."
                if abs(worst["sigma"]) < 3
                else "beyond 3 sigma; rerun that node before treating it as real."
            )
        )

    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
