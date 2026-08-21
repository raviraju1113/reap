"""Run LiveCodeBench against an existing vLLM server from inside this environment.

Unlike BFCL, ``lcb_runner`` installs cleanly here -- it is the vendored
``third-party/LiveCodeBench`` submodule, already an editable dependency -- so
there is no separate interpreter and no subprocess. This module exists for the
three things ``lcb_runner.runner.main`` does not do:

**Bound the request concurrency.** ``VLLMServerRunner.run_batch`` fires *every*
prompt at once through ``asyncio.gather``. With 454 problems against one server
that leaves ~420 requests sitting in vLLM's queue while their client-side
1800 s timeout ticks down, and ``make_auto_request`` retries forever on a
timeout -- so a slow cell degrades into a retry storm rather than a slow cell.
Masked cells are exactly the ones that generate to ``max_tokens``, so this is a
real hazard for a failure sweep and not a hypothetical one. The patch below
keeps at most ``num_threads`` requests in flight, which changes nothing about
total throughput (vLLM was going to serve them ``max_num_seqs`` at a time
anyway) and everything about per-request latency.

**Read the score back off disk.** ``lcb_main`` returns ``None``; the score lives
in a file whose name encodes the scenario, ``n`` and the temperature. A caller
that wants a number has to reconstruct that path, so it is done here once.

**Reduce to a comparable summary.** ``lcb_summary.json`` holds the pass@1, the
problem count, the per-difficulty breakdown, and the request-failure counts, in
the same shape ``reap.bfcl`` produces -- which is what lets the node-failure
sweep treat the two benchmarks identically.

The grading step executes model-generated code in subprocesses of this process
(``ProcessPoolExecutor``, ``num_process_evaluate`` workers). That is inherent to
LiveCodeBench; run it where you would be willing to run untrusted code.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
from collections import Counter

logger = logging.getLogger(__name__)

SUMMARY_FILENAME = "lcb_summary.json"

# The pinned `livecodebench/code_generation_lite` release covers 2023-09 through
# 2025-04, so an end date past that is not binding -- it is kept at the repo's
# historical value so the window is stated the same way everywhere.
DEFAULT_RELEASE_VERSION = "release_latest"
DEFAULT_START_DATE = "2024-08-01"
DEFAULT_END_DATE = "2025-07-31"

# Problem counts for the windows this repo uses. These are the `--expect-n`
# defaults: a cell that scored fewer problems than the baseline is not
# comparable to it, so a short run has to fail rather than count.
EXPECTED_N_BY_WINDOW = {
    # 2024-08 onwards: the LCB v5+v6 contest window. 454 problems.
    ("2024-08-01", "2025-07-31"): 454,
    # What src/reap/eval.py has historically run: 2025 contests only.
    ("2025-01-01", "2025-07-31"): 182,
}
DEFAULT_EXPECTED_N = EXPECTED_N_BY_WINDOW[(DEFAULT_START_DATE, DEFAULT_END_DATE)]

# Generous for non-thinking code generation (a solution is a few hundred
# tokens), and the value the repo's other LCB runs use. It matters mainly as the
# ceiling on how long a degenerate, repeating generation can run.
DEFAULT_MAX_TOKENS = 16384


def _patch_bounded_concurrency(limit: int) -> None:
    """Cap in-flight requests at ``limit`` by replacing ``run_batch``.

    A semaphore cannot simply be created once and reused: ``asyncio`` primitives
    bind to the first loop that awaits them, and ``lcb_runner`` calls
    ``asyncio.run`` afresh for every batch, which in a sweep means once per
    cell. So the semaphore is constructed inside the coroutine, per batch.

    This is upstream's ``run_batch`` minus the response cache, which this path
    never enables (``use_cache=False``) -- a cache keyed on the prompt would be
    actively wrong here, since the whole point of a sweep cell is that the same
    prompt gets a different answer under a different routing mask.
    """
    from lcb_runner.runner.vllm_server_runner import VLLMServerRunner
    from tqdm.asyncio import tqdm as atqdm

    def run_batch(self, prompts):
        async def run_all():
            semaphore = asyncio.Semaphore(limit)

            async def one(prompt):
                text = json.dumps(prompt) if isinstance(prompt, list) else prompt
                async with semaphore:
                    return await self._run_single(text)

            return await atqdm.gather(*[one(p) for p in prompts])

        return asyncio.run(run_all())

    VLLMServerRunner.run_batch = run_batch
    logger.info("LiveCodeBench: capped in-flight requests at %d", limit)


def run_livecodebench(
    hf_model_name: str,
    results_dir: str | os.PathLike,
    server_url: str,
    local_model_path: str | None = None,
    release_version: str = DEFAULT_RELEASE_VERSION,
    start_date: str | None = DEFAULT_START_DATE,
    end_date: str | None = DEFAULT_END_DATE,
    n: int = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    num_threads: int = 32,
    num_process_evaluate: int = 12,
    timeout: int = 120,
    enable_thinking: bool = False,
    output_path: str | os.PathLike | None = None,
) -> dict:
    """Run LCB code generation against ``server_url`` and return the summary.

    ``hf_model_name`` must be a key in ``lcb_runner``'s ``LanguageModelStore``
    (e.g. ``zai-org/GLM-4.5-Air``) -- it selects the prompt format and the
    runner. ``local_model_path`` is what goes on the wire as the OpenAI
    ``model`` field, so for a locally-served checkpoint it must equal the
    server's served-model-name exactly; vLLM matches served names literally.

    ``output_path`` defaults to ``results_dir``, which is where the rest of the
    repo (``scripts/report_evals.py``) expects LCB's artifacts to land.

    Raises if the run produced no score, or if any request came back empty --
    an empty generation grades as a wrong answer, which would quietly depress
    the score of whichever cell happened to hit a server hiccup.
    """
    from lcb_runner.runner.main import get_args_dict
    from lcb_runner.runner.main import main as lcb_main
    from lcb_runner.utils.path_utils import get_output_path

    results_dir = pathlib.Path(results_dir)
    out_dir = pathlib.Path(output_path) if output_path else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _patch_bounded_concurrency(num_threads)

    args = get_args_dict(
        model=hf_model_name,
        local_model_path=local_model_path,
        n=n,
        codegen_n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        release_version=release_version,
        start_date=start_date,
        end_date=end_date,
        output_path=str(out_dir),
        base_url=f"{server_url.rstrip('/')}/v1",
        enable_thinking=enable_thinking,
        evaluate=True,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )
    logger.info(
        "Running LiveCodeBench: %s %s..%s, n=%d, temperature=%s, max_tokens=%d, "
        "thinking=%s -> %s",
        release_version,
        start_date,
        end_date,
        n,
        temperature,
        max_tokens,
        enable_thinking,
        out_dir,
    )
    lcb_main(args)

    # `model.model_repr` is unused by this fork's path builder (the paths are
    # rooted at output_path), but the signature still wants it.
    generation_file = pathlib.Path(get_output_path(hf_model_name, args))
    eval_file = pathlib.Path(str(generation_file).replace(".json", "_eval.json"))
    eval_all_file = pathlib.Path(str(generation_file).replace(".json", "_eval_all.json"))

    summary = summarize_lcb_run(eval_file, eval_all_file)
    summary.update(
        {
            "release_version": release_version,
            "start_date": start_date,
            "end_date": end_date,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "enable_thinking": enable_thinking,
        }
    )

    summary_path = out_dir / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_file"] = str(summary_path)
    logger.info(
        "Finished LiveCodeBench: pass@1=%s over %s problems (%s)",
        summary["pass_at_1"],
        summary["total_count"],
        summary_path,
    )
    return summary


def summarize_lcb_run(eval_file: pathlib.Path, eval_all_file: pathlib.Path) -> dict:
    """Reduce LCB's two artifacts to the comparable summary shape.

    ``<...>_eval.json`` is the ``[metrics, results, metadatas]`` triple whose
    first element carries ``pass@1`` and a per-problem ``detail``.
    ``<...>_eval_all.json`` is the per-problem record, which is the only place
    the difficulty and the raw generations appear -- hence the second read.
    """
    if not eval_file.exists():
        raise RuntimeError(
            f"LiveCodeBench wrote no eval file at {eval_file}. Either generation "
            f"or grading did not finish; check the run log."
        )

    metrics = json.loads(eval_file.read_text())
    overall = metrics[0] if isinstance(metrics, list) else metrics
    pass_at_1 = overall.get("pass@1")
    detail = (overall.get("detail") or {}).get("pass@1") or {}

    difficulties: dict[str, dict] = {}
    failed_requests = 0
    no_code_extracted = 0
    total = len(detail)

    if eval_all_file.exists():
        graded = json.loads(eval_all_file.read_text())
        total = len(graded) or total
        passes: Counter[str] = Counter()
        counts: Counter[str] = Counter()
        for instance in graded:
            level = instance.get("difficulty") or "unknown"
            counts[level] += 1
            if any(instance.get("graded_list") or []):
                passes[level] += 1
            outputs = instance.get("output_list") or []
            if not outputs or not (outputs[0] or "").strip():
                failed_requests += 1
            elif not any((code or "").strip() for code in instance.get("code_list") or []):
                no_code_extracted += 1
        difficulties = {
            level: {"accuracy": passes[level] / counts[level], "num": counts[level]}
            for level in sorted(counts)
        }
    else:
        logger.warning("no %s; skipping the per-difficulty breakdown", eval_all_file)

    return {
        # `pass@1` is a fraction in [0, 1], matching `reap.bfcl`'s
        # `overall_accuracy`, so the sweep's reporting is benchmark-agnostic.
        "pass_at_1": pass_at_1,
        "total_count": total,
        "difficulties": difficulties,
        # An empty response is an infrastructure failure being scored as a wrong
        # answer. Counted separately from a non-empty answer that carried no
        # code block, which is a genuine model failure.
        "failed_requests": failed_requests,
        "no_code_extracted": no_code_extracted,
        "eval_file": str(eval_file),
        "eval_all_file": str(eval_all_file) if eval_all_file.exists() else None,
    }


def verify_lcb_summary(summary: dict, expect_n: int | None = None) -> dict:
    """Confirm an LCB summary describes a complete, comparable run."""
    if summary.get("pass_at_1") is None:
        raise RuntimeError(f"LiveCodeBench summary has no pass@1: {summary}")
    if expect_n and summary.get("total_count") != expect_n:
        raise RuntimeError(
            f"LiveCodeBench scored {summary.get('total_count')} problems, expected "
            f"{expect_n}. A truncated run is not comparable to the baseline -- "
            f"pass --expect-n 0 to disable this check."
        )
    if summary.get("failed_requests"):
        raise RuntimeError(
            f"{summary['failed_requests']} LiveCodeBench request(s) returned an "
            f"empty generation. Those grade as wrong answers, so this cell's "
            f"score is depressed by a server failure rather than by the mask. "
            f"Rerun the cell."
        )
    return summary


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Run LiveCodeBench code generation against a live vLLM server"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="the server's served-model-name, byte-identical (vLLM matches it "
        "literally, so a trailing slash 404s every request)",
    )
    parser.add_argument(
        "--hf-model-name",
        default=None,
        help="key in lcb_runner's LanguageModelStore, which selects the prompt "
        "format. Defaults to whatever reap.eval maps --model-path to.",
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--server-url", default="http://0.0.0.0:8000")
    parser.add_argument("--release-version", default=DEFAULT_RELEASE_VERSION)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--num-process-evaluate", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--expect-n",
        type=int,
        default=None,
        help="fail if the run did not score this many problems (0 disables). "
        "Defaults to the measured count for the selected window, if known.",
    )
    cli_args = parser.parse_args()

    hf_name = cli_args.hf_model_name
    if hf_name is None:
        # Imported here, not at module level: reap.eval pulls in torch and vLLM,
        # which this module otherwise does not need.
        from reap.eval import get_original_model_name

        hf_name, uncompressed = get_original_model_name(cli_args.model_path)
        local_path = None if uncompressed else cli_args.model_path
    else:
        local_path = cli_args.model_path

    expect_n = cli_args.expect_n
    if expect_n is None:
        expect_n = EXPECTED_N_BY_WINDOW.get((cli_args.start_date, cli_args.end_date), 0)

    result = run_livecodebench(
        hf_model_name=hf_name,
        local_model_path=local_path,
        results_dir=cli_args.results_dir,
        server_url=cli_args.server_url,
        release_version=cli_args.release_version,
        start_date=cli_args.start_date,
        end_date=cli_args.end_date,
        n=cli_args.n,
        temperature=cli_args.temperature,
        max_tokens=cli_args.max_tokens,
        num_threads=cli_args.num_threads,
        num_process_evaluate=cli_args.num_process_evaluate,
        timeout=cli_args.timeout,
        enable_thinking=cli_args.enable_thinking,
    )
    verify_lcb_summary(result, expect_n or None)
    print(json.dumps({k: v for k, v in result.items() if k != "difficulties"}, indent=2))
    sys.exit(0)
