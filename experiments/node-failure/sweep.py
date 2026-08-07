"""Sweep EP-node failures across nodes and failure semantics on a live vLLM server.

Runs one benchmark per (mode, node) cell against a single long-lived server,
setting the dead-expert mask between cells over the ``/reap/failure`` control
plane exposed by ``reap.expert_failure_server``. That is what makes the sweep
affordable: a 32-node sweep costs one checkpoint load rather than 32.

For Qwen3-30B-A3B (128 experts / 32 nodes = 4 per node) the default matrix is
32 nodes x 2 modes (``drop_renorm``, ``reroute``) = 64 cells. At ~61 GB in bf16
the model fits on a single B200, so the cheapest layout is **one server per GPU**
and ``--shard`` to split the cell list across them. Measured at ~27.5 min per
cell, 8-way parallelism puts the full sweep at roughly 4 hours.

Typical use, single GPU::

    # terminal 1 -- boot once, leave up for the whole sweep
    CUDA_VISIBLE_DEVICES=0 python -m reap.expert_failure_server \\
        --model /home/ravira/checkpoints/Qwen3-30B-A3B --port 8000

    # terminal 2
    python experiments/node-failure/sweep.py \\
        --model /home/ravira/checkpoints/Qwen3-30B-A3B --port 8000 \\
        --results-dir artifacts/node-failure --preflight --baseline-repeats 2

Eight GPUs in parallel -- shard ``i`` takes every 8th cell, and all shards write
into the same results tree::

    for i in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$i python -m reap.expert_failure_server \\
          --model $MODEL --port $((8000+i)) &
    done
    # once healthy:
    for i in $(seq 0 7); do
      python experiments/node-failure/sweep.py --model $MODEL \\
          --port $((8000+i)) --shard $i --num-shards 8 \\
          --baseline-repeats 2 &
    done

Results land in ``<results-dir>/<mode>/node_<k>/`` with a ``cell.json`` holding
the routing counters. The driver is resumable: a cell whose ``cell.json``
exists is skipped, so an interrupted sweep restarts where it stopped. That also
makes sharding safe to redo with a different shard count.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time

import requests

# Import from the installed package rather than relative paths so this runs
# the same whether invoked as a script or a module.
from reap.args import EvalArgs, ModelArgs
from reap.eval import run_evaluate
from reap.expert_failure import DEFAULT_NUM_EXPERTS, DEFAULT_NUM_NODES, experts_for_node

logger = logging.getLogger("node-failure-sweep")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


def set_failure(base_url: str, mode: str, node_id: int | None, **kw) -> dict:
    """Broadcast a mask to every worker and return the server's confirmation.

    Raises on a non-200 or an inconsistent mask -- silently continuing here
    would mean recording an eval under the wrong (or no) failure condition,
    which is worse than stopping the sweep.
    """
    payload = {"mode": mode, "node_id": node_id, **kw}
    resp = requests.post(f"{base_url}/reap/failure", json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"failed to set failure {payload}: {resp.text}")
    state = resp.json()
    if not state.get("mask_consistent", False):
        raise RuntimeError(f"mask inconsistent across workers: {state}")
    return state


def get_failure(base_url: str) -> dict:
    resp = requests.get(f"{base_url}/reap/failure", timeout=120)
    resp.raise_for_status()
    return resp.json()


def check_server(base_url: str) -> None:
    """Fail fast if the server is missing or was launched without the control plane."""
    try:
        requests.get(f"{base_url}/health", timeout=30).raise_for_status()
    except Exception as exc:
        raise SystemExit(
            f"no healthy vLLM server at {base_url} ({exc}).\n"
            "Start one with:  python -m reap.expert_failure_server --model ... --port ..."
        ) from exc

    try:
        state = get_failure(base_url)
    except Exception as exc:
        raise SystemExit(
            f"server at {base_url} has no /reap/failure route ({exc}).\n"
            "It was probably started with plain `vllm serve`; use "
            "`python -m reap.expert_failure_server` instead."
        ) from exc

    if not state.get("patched", False):
        raise SystemExit(
            "workers report FusedMoE.select_experts is NOT patched -- the "
            "expert_failure plugin did not load, so masks would be no-ops.\n"
            "Reinstall the package (`uv pip install -e .`) so the "
            "`vllm.general_plugins` entry point is registered."
        )
    logger.info("server ready at %s; routing hook installed", base_url)


EXPECTED_MATH500_N = 500


def read_evalscope_report(cell_dir: pathlib.Path, task: str) -> dict:
    """Parse the evalscope report for ``task`` under ``cell_dir``.

    evalscope writes to ``<work_dir>/<timestamp>/reports/<model>/<task>.json``
    (see ``evalscope/utils/io_utils.py:OutputsStructure``), holding a top-level
    ``score`` plus per-subset ``num`` counts.

    This exists because ``run_evaluate`` cannot be trusted to have actually run
    anything: it catches per-benchmark exceptions and returns normally, so a
    failed dataset download looks exactly like a successful eval. Reading the
    artifact is the only way to know a real score was produced.
    """
    reports = sorted(cell_dir.glob(f"evalscope_results/**/reports/**/{task}.json"))
    if not reports:
        raise RuntimeError(
            f"no evalscope report for '{task}' under {cell_dir}. The benchmark did "
            f"not run (evalscope errors are logged, not raised, unless strict=True). "
            f"Check the run log for 'An error occurred during math evaluation'."
        )
    # Newest wins if a cell was re-run in place.
    report = json.loads(reports[-1].read_text())

    n = sum(
        subset.get("num", 0)
        for metric in report.get("metrics", [])
        for category in metric.get("categories", [])
        for subset in category.get("subsets", [])
    )
    return {"task": task, "score": report.get("score"), "num": n, "report": str(reports[-1])}


def _verify_math500(cell_dir: pathlib.Path, expect_n: int | None) -> dict:
    """Confirm MATH-500 produced a real score over the expected problem count."""
    result = read_evalscope_report(cell_dir, "math_500")
    if result["score"] is None:
        raise RuntimeError(f"math_500 report at {result['report']} has no score")
    if expect_n and result["num"] != expect_n:
        raise RuntimeError(
            f"math_500 scored {result['num']} problems, expected {expect_n} "
            f"({result['report']}). A truncated run is not comparable to the "
            f"baseline -- pass --expect-n 0 to disable this check."
        )
    return result


def _completion(base_url: str, model: str, prompt: str, max_tokens: int = 48) -> str:
    """One greedy completion, used only by the preflight check."""
    resp = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 0,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"]


def preflight(base_url: str, model: str, args: argparse.Namespace) -> None:
    """Prove the mask is real before spending hours on a sweep.

    Three questions, answered against the live server rather than by assertion:

    1. Is the hook on the hot path? -- `calls` must advance when traffic flows.
       If vLLM routes through a fused kernel that bypasses
       ``FusedMoE.select_experts``, every "masked" cell is silently a baseline.
    2. Is the right *number* of experts being masked? -- the realized dead-slot
       rate should sit near the topology's 1/32, not at zero and not at 1.
    3. Does masking actually change the model's output? -- a mask that routes
       correctly but has no effect on generation would produce a flat sweep that
       looks like "no degradation".
    """
    prompt = "Question: What is 17 * 23?\nAnswer:"

    logger.info("preflight 1/3: baseline generation with the mask off")
    set_failure(base_url, "off", None)
    before = get_failure(base_url)
    baseline_text = _completion(base_url, model, prompt)

    logger.info("preflight 2/3: hook is invoked and counts dead slots")
    # Half the experts, so the effect is unmistakable -- this is a diagnostic,
    # not a realistic failure.
    half = list(range(args.num_experts // 2))
    state = set_failure(base_url, "drop", None, expert_ids=half, num_experts=args.num_experts)
    if state.get("num_dead") != len(half):
        raise SystemExit(f"mask did not land: expected {len(half)} dead, got {state}")
    masked_text = _completion(base_url, model, prompt)
    stats = get_failure(base_url)

    if stats.get("calls", 0) <= before.get("calls", 0):
        raise SystemExit(
            "FAIL: the routing hook was never invoked despite serving traffic.\n"
            "vLLM is routing through a path that bypasses FusedMoE.select_experts "
            "(a fused FlashInfer/TRT-LLM MoE kernel, most likely). Every masked "
            "cell would silently be a baseline. The patch has to move."
        )
    rate = stats.get("dead_slot_rate", 0.0)
    if not 0.0 < rate < 1.0:
        raise SystemExit(f"FAIL: implausible dead_slot_rate {rate} with half the experts masked")
    logger.info(
        "  calls=%s tokens=%s dead_slot_rate=%.3f lost_gate_mass=%.3f",
        stats.get("calls"),
        stats.get("tokens"),
        rate,
        stats.get("lost_gate_mass_frac", 0.0),
    )

    logger.info("preflight 3/3: masking changes the output")
    if masked_text == baseline_text:
        raise SystemExit(
            "FAIL: masking half the experts produced byte-identical output.\n"
            "The counters moved but generation did not, so the masked weights are "
            "not reaching the MoE compute. Do not trust a sweep run this way."
        )

    # Realistic single-node mask, to confirm the rate matches the topology.
    set_failure(base_url, "drop", 0, num_experts=args.num_experts, num_nodes=args.num_nodes)
    _completion(base_url, model, prompt)
    node_stats = get_failure(base_url)
    expected = 1.0 / args.num_nodes
    logger.info(
        "  node 0 (%d/%d experts): dead_slot_rate=%.4f (uniform-routing expectation %.4f)",
        args.num_experts // args.num_nodes,
        args.num_experts,
        node_stats.get("dead_slot_rate", 0.0),
        expected,
    )

    set_failure(base_url, "off", None)
    logger.info("preflight PASSED -- hook is live, masks land, and output responds")


def run_cell(
    cell_dir: pathlib.Path,
    model: str,
    base_url: str,
    args: argparse.Namespace,
    mode: str,
    node_id: int | None,
) -> dict:
    """Run one (mode, node) cell and persist its result. Returns the cell record.

    ``cell.json`` is written only after the benchmark artifact has been verified,
    so a cell marked complete really did produce a score -- and a failed cell is
    retried rather than skipped on resume.
    """
    result_file = cell_dir / "cell.json"
    if result_file.exists() and not args.overwrite:
        logger.info("skipping %s (already complete)", cell_dir)
        return json.loads(result_file.read_text())

    cell_dir.mkdir(parents=True, exist_ok=True)

    state = set_failure(
        base_url,
        mode,
        node_id,
        num_experts=args.num_experts,
        num_nodes=args.num_nodes,
    )
    logger.info(
        "cell mode=%s node=%s -> %d dead expert(s) %s",
        mode,
        node_id,
        state.get("num_dead", 0),
        state.get("dead_experts", [])[:4],
    )

    # `existing_server_url` keeps run_evaluate from booting (and killing) its own
    # server, so every cell reuses the one load of the checkpoint.
    eval_args = EvalArgs(
        use_server=True,
        existing_server_url=base_url,
        run_lm_eval=False,
        run_evalplus=False,
        run_livecodebench=False,
        run_wildbench=False,
        run_math=True,
        # MATH-500 only. The default is gsm8k+math_500, which would add 1319
        # problems per cell for a benchmark this sweep does not report.
        math_tasks=["math_500"],
        # Fail the cell instead of recording an empty one.
        strict=True,
        greedy=True,
        vllm_port=args.port,
        parallel_tasks=args.parallel_tasks,
    )
    model_args = ModelArgs(model_name=model)

    start = time.time()
    run_evaluate(
        model_args=model_args,
        results_dir=cell_dir,
        eval_args=eval_args,
        seed=args.seed,
    )
    elapsed = time.time() - start

    # Read the score back off disk rather than trusting run_evaluate's return.
    math500 = _verify_math500(cell_dir, args.expect_n)

    # Routing counters are only meaningful after the eval traffic has run.
    routing = get_failure(base_url)
    if mode != "off" and routing.get("calls", 0) == 0:
        raise RuntimeError(
            f"the routing hook was never called during cell mode={mode} "
            f"node={node_id}, so the mask did nothing and this eval is really a "
            f"baseline. vLLM is routing through a path that bypasses "
            f"FusedMoE.select_experts; the patch has to move."
        )

    record = {
        "mode": mode,
        "node_id": node_id,
        "elapsed_s": round(elapsed, 1),
        "seed": args.seed,
        "math_500": {k: math500[k] for k in ("score", "num", "report")},
        "routing": routing,
    }
    result_file.write_text(json.dumps(record, indent=2))
    logger.info(
        "cell mode=%s node=%s done in %.0f min: math_500=%.4f (n=%d), "
        "dead_slot_rate=%.4f, lost_gate_mass=%.4f",
        mode,
        node_id,
        elapsed / 60,
        math500["score"],
        math500["num"],
        routing.get("dead_slot_rate", 0.0),
        routing.get("lost_gate_mass_frac", 0.0),
    )
    return record


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="model name as served (for the harness)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--results-dir", type=pathlib.Path, default=pathlib.Path("artifacts/node-failure"))
    p.add_argument(
        "--modes",
        nargs="+",
        default=["drop_renorm", "reroute"],
        choices=["drop", "drop_renorm", "reroute"],
        help="failure semantics to sweep. Default is the two deployable ones: "
        "`drop_renorm` (survivors rescaled) and `reroute` (next-best expert "
        "substituted). `drop` is available but not swept by default -- it models "
        "a runtime that neither rescales nor reroutes, which is a strict lower "
        "bound rather than a realistic serving behaviour.",
    )
    p.add_argument(
        "--nodes",
        nargs="+",
        type=int,
        default=None,
        help="node ids to sweep (default: all). Use e.g. `--nodes 0 4 8 12` to subsample.",
    )
    p.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    p.add_argument("--num-nodes", type=int, default=DEFAULT_NUM_NODES)
    p.add_argument(
        "--baseline-repeats",
        type=int,
        default=2,
        help="unmasked runs used to establish the noise floor; without these "
        "the 'within 5%%' criterion cannot be evaluated",
    )
    p.add_argument("--parallel-tasks", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="rerun cells that already have results")
    p.add_argument(
        "--shard",
        type=int,
        default=0,
        help="index of this shard when splitting the sweep across several "
        "single-GPU servers (see module docstring)",
    )
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument(
        "--expect-n",
        type=int,
        default=EXPECTED_MATH500_N,
        help="fail a cell whose MATH-500 report does not cover this many problems "
        "(0 disables). Guards against a truncated run being compared to a full baseline.",
    )
    p.add_argument(
        "--preflight",
        action="store_true",
        help="before sweeping, prove the hook is invoked, masks land, and masking "
        "actually changes generation. Takes seconds; run it once per server.",
    )
    p.add_argument(
        "--preflight-only",
        action="store_true",
        help="run the preflight check and exit without sweeping",
    )
    args = p.parse_args()
    if args.preflight_only:
        args.preflight = True

    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"--shard must be in [0, {args.num_shards}), got {args.shard}")

    base_url = f"http://{args.host}:{args.port}"
    check_server(base_url)

    if args.preflight:
        preflight(base_url, args.model, args)
        if args.preflight_only:
            return 0

    nodes = args.nodes if args.nodes is not None else list(range(args.num_nodes))
    for node in nodes:
        experts_for_node(node, args.num_experts, args.num_nodes)  # validate up front

    # Build the full cell list, then take this shard's slice. Baselines lead so
    # that a shard which owns one runs it before any masked cell, surfacing a
    # broken setup early. Every shard writes into the same results tree; the
    # resume check keeps concurrent shards from redoing each other's work.
    all_cells: list[tuple[pathlib.Path, str, int | None]] = [
        (args.results_dir / "baseline" / f"rep_{rep}", "off", None)
        for rep in range(args.baseline_repeats)
    ] + [
        (args.results_dir / mode / f"node_{node:02d}", mode, node)
        for mode in args.modes
        for node in nodes
    ]
    cells = all_cells[args.shard :: args.num_shards]

    logger.info(
        "sweep: %d baseline + %d mode(s) x %d node(s) = %d cells total; "
        "shard %d/%d runs %d of them",
        args.baseline_repeats,
        len(args.modes),
        len(nodes),
        len(all_cells),
        args.shard,
        args.num_shards,
        len(cells),
    )

    records = []
    try:
        for cell_dir, mode, node_id in cells:
            records.append(
                run_cell(cell_dir, args.model, base_url, args, mode=mode, node_id=node_id)
            )
    finally:
        # Never leave a mask applied on a shared server.
        try:
            set_failure(base_url, "off", None)
        except Exception:
            logger.warning("could not clear the failure mask on exit", exc_info=True)

        # Shard-scoped so concurrent shards don't overwrite each other; the
        # per-cell `cell.json` files remain the authoritative record either way.
        name = (
            "summary.json"
            if args.num_shards == 1
            else f"summary.shard{args.shard:02d}-of-{args.num_shards:02d}.json"
        )
        summary = args.results_dir / name
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(records, indent=2))
        logger.info("wrote %d cell record(s) to %s", len(records), summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
