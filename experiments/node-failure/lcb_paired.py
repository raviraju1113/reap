"""Paired per-problem analysis of a LiveCodeBench node-failure sweep.

Why this exists
---------------
``report.py`` compares pooled pass@1: one number per cell against the baseline
mean. On BFCL that works, because repeated unmasked runs land within 0.18 points
of each other. On LiveCodeBench they do not -- measured here, two unmasked runs
of the *same* configuration differ by ~2.6 points, which is larger than the 5%
relative bar the experiment is trying to resolve.

The reason is not sampling in the usual sense (decoding is greedy). It is that
vLLM's batch composition varies run to run, that changes a few tokens, and in
code generation one token flips a whole program between passing and failing.
About 9% of baseline answers also sit right at the ``max_tokens`` truncation
boundary, where a small difference flips the whole answer.

An unpaired comparison throws away the information that makes this tractable:
*which* problems changed. Almost every problem gives the same verdict in both
runs, and the pooled difference is a small residue of a few flips in each
direction. So this script does the paired test instead:

* ``fail_to_pass`` / ``pass_to_fail`` -- discordant problems versus the baseline.
* The net delta is ``fail_to_pass - pass_to_fail`` problems, which is exactly the
  pooled delta, but now with its own standard error: under the null that masking
  changes nothing, each discordant problem is an independent coin flip, so
  ``SE = sqrt(b + c)`` problems (McNemar). That SE is typically 3-4x smaller
  than the unpaired one, because it depends on the *discordant* count rather
  than on the variance of two whole-benchmark proportions.
* A two-sided exact binomial p-value on the discordant pairs (McNemar's test,
  exact form -- the chi-square approximation is not safe at these counts).

The baseline is the union of the baseline repeats: a problem is scored as
"baseline pass" by majority vote across the repeats, which removes the
single-run flip noise from the reference itself. Problems that flip *between
baseline repeats* are reported separately as ``unstable`` -- they are the
measured flip rate of the harness, and the honest denominator for judging
whether a node's flips mean anything.

Usage::

    python experiments/node-failure/lcb_paired.py artifacts/glm-node-failure/livecodebench
    python experiments/node-failure/lcb_paired.py <dir> --csv paired.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys
from collections import Counter

# LCB writes its per-problem record next to the cell's summary. The filename
# encodes scenario/n/temperature, so it is matched by glob rather than spelled.
EVAL_ALL_GLOB = "Scenario.codegeneration_*_eval_all.json"


def load_grades(cell_dir: pathlib.Path) -> dict[str, bool]:
    """``{question_id: passed}`` for one cell, from LCB's own graded record.

    Reads the per-problem file rather than the summary, because the whole point
    here is the identity of the problems that changed.
    """
    candidates = sorted(cell_dir.glob(EVAL_ALL_GLOB))
    if not candidates:
        raise FileNotFoundError(f"no {EVAL_ALL_GLOB} under {cell_dir}")
    graded = json.loads(candidates[-1].read_text())
    return {
        instance["question_id"]: bool(any(instance.get("graded_list") or []))
        for instance in graded
    }


def load_difficulties(cell_dir: pathlib.Path) -> dict[str, str]:
    candidates = sorted(cell_dir.glob(EVAL_ALL_GLOB))
    graded = json.loads(candidates[-1].read_text())
    return {i["question_id"]: i.get("difficulty") or "unknown" for i in graded}


# Coarse buckets over LCB's per-test error messages. The point is not the
# taxonomy but the channels: a mask can cost accuracy by producing wrong logic,
# by producing slower code, or by rambling past `max_tokens` and emitting a
# fragment that will not even parse. Those have different implications and the
# pooled pass@1 shows none of them.
FAILURE_BUCKETS = (
    ("time_limit", ("Time Limit Exceeded",)),
    ("truncated", ("expected an indent", "unexpected EOF", "invalid syntax",
                   "was never closed", "EOF while scanning")),
    ("runtime", ("Runtime Error", "TestRunnerError")),
    ("wrong_answer", ("Wrong Answer", "Wrong answer")),
)


def failure_modes(cell_dir: pathlib.Path) -> Counter:
    """Count how the failures in one cell failed.

    LCB stores each per-test metadata entry as a **JSON string inside a list**,
    not as a dict, so it has to be parsed twice; reading it as a dict silently
    finds nothing at all, which is a quiet way to conclude "no timeouts".
    """
    candidates = sorted(cell_dir.glob(EVAL_ALL_GLOB))
    if not candidates:
        raise FileNotFoundError(f"no {EVAL_ALL_GLOB} under {cell_dir}")
    counts: Counter = Counter()
    for instance in json.loads(candidates[-1].read_text()):
        if any(instance.get("graded_list") or []):
            counts["pass"] += 1
            continue
        message = ""
        for entry in instance.get("metadata") or []:
            if isinstance(entry, str):
                try:
                    entry = json.loads(entry)
                except json.JSONDecodeError:
                    continue
            if isinstance(entry, dict):
                message = str(entry.get("error_message") or entry.get("error") or "")
                if message:
                    break
        for name, needles in FAILURE_BUCKETS:
            if any(needle in message for needle in needles):
                counts[name] += 1
                break
        else:
            counts["other" if message else "no_message"] += 1
    return counts


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact binomial p-value on the discordant pairs.

    Under the null, each of the ``b + c`` discordant problems is equally likely
    to have flipped in either direction. The chi-square approximation is not
    trustworthy at the counts this sweep produces (often < 25 discordant), so
    the exact form is used.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def majority(votes: list[bool]) -> bool | None:
    """Majority verdict, or ``None`` when the votes tie.

    The tie case is the whole reason this is a function. A strict
    ``sum * 2 > len`` rule silently resolves an even split to *fail*, which with
    an even number of repeats drags the reference below every individual run --
    measured at 6 repeats: majority 0.3987 against a per-run range of
    0.4141-0.4405, which would flatter every masked cell compared to it. A tied
    problem carries no information about whether the model solves it, so it is
    excluded rather than broken toward either verdict.
    """
    passes = sum(votes)
    if passes * 2 == len(votes):
        return None
    return passes * 2 > len(votes)


def majority_baseline(reps: list[dict[str, bool]]) -> tuple[dict[str, bool], set[str]]:
    """Majority-vote reference plus the set of problems that disagreed.

    Using one repeat as the reference would import that repeat's flip noise into
    every single delta. Majority vote across repeats is the cheapest way to get
    a reference that is stabler than any one run; the disagreements are returned
    because they are the measurement's own flip rate.

    Problems whose votes tie are left out of the reference entirely, so with an
    even number of repeats the comparison set is slightly smaller than the
    benchmark. Every row reports its own ``n``.
    """
    if not reps:
        raise SystemExit("no baseline repeats found -- nothing to compare against")
    questions = set(reps[0])
    for rep in reps[1:]:
        questions &= set(rep)
    reference: dict[str, bool] = {}
    unstable: set[str] = set()
    for question in questions:
        votes = [rep[question] for rep in reps]
        if len(set(votes)) > 1:
            unstable.add(question)
        verdict = majority(votes)
        if verdict is not None:
            reference[question] = verdict
    return reference, unstable


def loo_null(reps: list[dict[str, bool]], questions: list[str]) -> list[dict]:
    """The same paired statistic, computed on baselines that had no mask.

    This is the null distribution the per-node rows have to be read against.
    Each repeat is compared to a reference built from the *other* repeats only
    -- leaving it out, because a majority that includes the run being tested is
    biased toward agreeing with it, which would understate the flip rate.

    Problems where the remaining repeats disagree have no majority and are
    excluded from that comparison, so the denominators differ slightly between
    rows; they are reported per row rather than assumed equal.
    """
    if len(reps) < 3:
        return []
    null_rows = []
    for index, held_out in enumerate(reps):
        others = [rep for i, rep in enumerate(reps) if i != index]
        fail_to_pass = pass_to_fail = compared = 0
        for question in questions:
            reference = majority([rep[question] for rep in others])
            if reference is None:
                continue  # the remaining repeats tie, so there is no reference
            compared += 1
            if held_out[question] and not reference:
                fail_to_pass += 1
            elif reference and not held_out[question]:
                pass_to_fail += 1
        null_rows.append(
            {
                "rep": index,
                "n": compared,
                "fail_to_pass": fail_to_pass,
                "pass_to_fail": pass_to_fail,
                "net_problems": fail_to_pass - pass_to_fail,
                "delta_pts": 100.0 * (fail_to_pass - pass_to_fail) / compared,
                "p_value": exact_mcnemar_p(fail_to_pass, pass_to_fail),
            }
        )
    return null_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=pathlib.Path)
    parser.add_argument("--csv", type=pathlib.Path, default=None)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="significance level for the per-node flag. Note 32 nodes are "
        "tested, so ~1.6 nodes are expected to clear 0.05 by chance; the "
        "Bonferroni-corrected level is printed alongside.",
    )
    args = parser.parse_args()

    baseline_dirs = sorted((args.results_dir / "baseline").glob("rep_*"))
    reps = []
    for path in baseline_dirs:
        try:
            reps.append(load_grades(path))
        except FileNotFoundError:
            print(f"skipping incomplete baseline {path.name}", file=sys.stderr)
    reference, unstable = majority_baseline(reps)
    difficulties = load_difficulties(baseline_dirs[0])

    n = len(reference)
    base_pass = sum(reference.values())
    print(f"baseline: {len(reps)} repeat(s), {n} problems in common")
    print(f"  majority-vote pass@1: {base_pass / n:.4f} ({base_pass}/{n})")
    for path, rep in zip(baseline_dirs, reps):
        passed = sum(rep[q] for q in reference)
        print(f"    {path.name}: {passed / n:.4f} ({passed}/{n})")
    if len(reps) > 1:
        # The harness's own flip rate. Every per-node flip count has to be read
        # against this: a node that flips fewer problems than two baselines flip
        # against each other has not been shown to do anything.
        print(
            f"  unstable across baselines: {len(unstable)} problem(s) "
            f"({len(unstable) / n:.1%}) -- the measurement's own flip rate"
        )
        pairwise = [
            sum(1 for q in reference if a[q] != b[q])
            for i, a in enumerate(reps)
            for b in reps[i + 1 :]
        ]
        print(
            f"  baseline-vs-baseline discordant pairs: {pairwise} "
            f"(mean {statistics.mean(pairwise):.1f})"
        )
        scores = [sum(rep[q] for q in reference) / n for rep in reps]
        if len(scores) > 1:
            print(
                f"  baseline pass@1 SD: {statistics.stdev(scores) * 100:.2f} pts "
                f"-> unpaired noise floor"
            )

    # The measured null: what the per-node statistic reads on cells that had no
    # mask at all. A masked node has to beat this band, not zero.
    null_rows = loo_null(reps, list(reference))
    if null_rows:
        print("\n  null (each baseline vs the majority of the others, no mask):")
        print(f"    {'rep':<6}{'n':>5}{'net':>6}{'+':>5}{'-':>5}{'delta':>8}{'p':>9}")
        for row in null_rows:
            print(
                f"    {row['rep']:<6}{row['n']:>5}{row['net_problems']:>6}"
                f"{row['fail_to_pass']:>5}{row['pass_to_fail']:>5}"
                f"{row['delta_pts']:>+8.2f}{row['p_value']:>9.3f}"
            )
        null_nets = [row["net_problems"] for row in null_rows]
        print(
            f"    null net range: {min(null_nets):+d} to {max(null_nets):+d} "
            f"problems -- a node inside this band has shown nothing"
        )

    rows = []
    for cell_dir in sorted(args.results_dir.glob("*/node_*")):
        mode = cell_dir.parent.name
        try:
            grades = load_grades(cell_dir)
        except FileNotFoundError:
            continue
        common = [q for q in reference if q in grades]
        fail_to_pass = sum(1 for q in common if grades[q] and not reference[q])
        pass_to_fail = sum(1 for q in common if reference[q] and not grades[q])
        net = fail_to_pass - pass_to_fail
        discordant = fail_to_pass + pass_to_fail
        # McNemar SE, in problems: the sqrt of the discordant count.
        se_problems = math.sqrt(discordant) if discordant else 0.0
        rows.append(
            {
                "mode": mode,
                "node": int(cell_dir.name.split("_")[1]),
                "n": len(common),
                "score": sum(grades[q] for q in common) / len(common),
                "fail_to_pass": fail_to_pass,
                "pass_to_fail": pass_to_fail,
                "net_problems": net,
                "delta_pts": 100.0 * net / len(common),
                "sigma": (net / se_problems) if se_problems else 0.0,
                "p_value": exact_mcnemar_p(fail_to_pass, pass_to_fail),
                # How much of this cell's movement is on problems that the
                # baselines themselves disagree about, i.e. was never signal.
                "on_unstable": sum(
                    1 for q in common if grades[q] != reference[q] and q in unstable
                ),
                "hard_to_fail": sum(
                    1
                    for q in common
                    if reference[q] and not grades[q] and difficulties.get(q) == "hard"
                ),
            }
        )

    if not rows:
        print("\nno completed masked cells yet", file=sys.stderr)
        return 0

    # Majority voting sharpens toward the more likely verdict, so a single run
    # compared to a majority reference is not centred on zero: on problems the
    # model passes less than half the time the majority says "fail" while a
    # single run passes some of them. The LOO null measures that offset on cells
    # that had no mask, and every per-node net is corrected by it. With 3
    # repeats the offset was ~0; at 6 it is ~+5 problems, large enough that
    # comparing to zero instead would understate every node.
    null_offset = (
        statistics.mean(row["net_problems"] for row in null_rows) if null_rows else 0.0
    )
    for row in rows:
        row["net_corrected"] = row["net_problems"] - null_offset
        row["delta_corrected_pts"] = 100.0 * row["net_corrected"] / row["n"]

    rows.sort(key=lambda r: r["net_problems"])
    bonferroni = args.alpha / len(rows)
    print(
        f"\npaired per-node comparison ({len(rows)} cell(s)), worst first.\n"
        f"'net' is problems gained minus lost; sigma is net/sqrt(discordant) "
        f"(McNemar).\nflagging p < {args.alpha} (*) and p < {bonferroni:.4f} "
        f"(**, Bonferroni over {len(rows)} nodes)"
    )
    if null_rows:
        print(
            f"nets are corrected by the measured null offset "
            f"({null_offset:+.1f} problems); 'raw' is before correction"
        )
    print(
        f"\n{'mode':<12}{'node':>5}{'score':>8}{'raw':>6}{'net':>7}{'+':>5}{'-':>5}"
        f"{'delta':>8}{'sigma':>7}{'p':>9}{'unstbl':>8}"
    )
    for row in rows:
        stars = "**" if row["p_value"] < bonferroni else ("*" if row["p_value"] < args.alpha else "")
        print(
            f"{row['mode']:<12}{row['node']:>5}{row['score']:>8.4f}"
            f"{row['net_problems']:>6}{row['net_corrected']:>+7.1f}"
            f"{row['fail_to_pass']:>5}{row['pass_to_fail']:>5}"
            f"{row['delta_corrected_pts']:>+8.2f}{row['sigma']:>+7.1f}"
            f"{row['p_value']:>9.3f}{row['on_unstable']:>8} {stars}"
        )

    nets = [r["net_corrected"] for r in rows]
    deltas = [r["delta_corrected_pts"] for r in rows]
    pooled_net = sum(r["net_problems"] for r in rows) - null_offset * len(rows)
    pooled_discordant = sum(r["fail_to_pass"] + r["pass_to_fail"] for r in rows)
    pooled_se = math.sqrt(pooled_discordant) if pooled_discordant else 0.0
    print(
        f"\npooled over {len(rows)} cells: net {pooled_net:+.1f} problem(s) of "
        f"{pooled_discordant} discordant (null-corrected)"
    )
    if pooled_se:
        print(
            f"  = {pooled_net / pooled_se:+.2f} sigma on the McNemar SE "
            f"({pooled_se:.1f} problems)"
        )
    print(
        f"  mean per-node delta {statistics.mean(deltas):+.2f} pts, "
        f"SD {statistics.stdev(deltas) if len(deltas) > 1 else float('nan'):.2f} pts"
    )
    print(f"  nodes below the null: {sum(1 for x in nets if x < 0)}/{len(nets)}")
    counts = Counter(r["mode"] for r in rows)
    print(f"  modes: {dict(counts)}")

    # How the failures failed, per cell. Read down the columns: a mask that
    # costs accuracy through slower code looks different from one that costs it
    # through fragments that do not parse.
    bucket_names = ["pass"] + [n for n, _ in FAILURE_BUCKETS] + ["other", "no_message"]
    print("\nfailure modes per cell (counts over all problems):")
    print(f"    {'cell':<18}" + "".join(f"{name:>13}" for name in bucket_names))
    mode_rows = []
    for cell_dir in sorted(args.results_dir.glob("baseline/rep_*")) + sorted(
        args.results_dir.glob("*/node_*")
    ):
        try:
            counts = failure_modes(cell_dir)
        except FileNotFoundError:
            continue
        label = f"{cell_dir.parent.name}/{cell_dir.name}"
        mode_rows.append((label, counts))
        print(
            f"    {label:<18}"
            + "".join(f"{counts.get(name, 0):>13}" for name in bucket_names)
        )
    base = [c for label, c in mode_rows if label.startswith("baseline/")]
    masked = [c for label, c in mode_rows if not label.startswith("baseline/")]
    if base and masked:
        print(f"\n    mean baseline (n={len(base)}) vs mean masked (n={len(masked)}):")
        for name in bucket_names:
            b_mean = statistics.mean(c.get(name, 0) for c in base)
            m_mean = statistics.mean(c.get(name, 0) for c in masked)
            if b_mean or m_mean:
                print(f"      {name:<14}{b_mean:>8.1f}{m_mean:>10.1f}{m_mean - b_mean:>+9.1f}")

    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
