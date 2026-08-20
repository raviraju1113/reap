"""Run the Berkeley Function-Calling Leaderboard against an existing vLLM server.

Why this is a separate process
-----------------------------
``bfcl_eval`` pins ``numpy==1.26.4`` while this repo's environment is on numpy
2.x for vLLM 0.10, and it pulls in a large vendor-SDK dependency set
(``mistralai``, ``anthropic``, ``cohere``, ``faiss-cpu``, ...) that has no
business in the serving environment. Installing it alongside vLLM would either
downgrade numpy or fail to resolve.

BFCL is *only an HTTP client* here -- the model is served by
``reap.expert_failure_server`` in the other environment -- so the clean split is
one interpreter per dependency set. This script is therefore run by
``.venv-bfcl/bin/python`` and **must not import** ``reap``, ``torch`` or
``vllm``. It lives in its own directory so that ``sys.path[0]`` (the script's
directory, which Python inserts automatically) contains nothing that could
shadow a ``bfcl_eval`` import. ``reap.bfcl`` is the in-environment wrapper that
launches it.

What it does
------------
1. Registers a prompting-mode model entry for a locally served checkpoint.
   BFCL only knows models in its own registry, and its GLM entries point at
   Zhipu's cloud API (``GLMAPIHandler``) or at the old ``glm-4-9b-chat`` prompt
   format, neither of which describes a GLM-4.5 checkpoint on our own server.
   The registration is done in-process against ``MODEL_CONFIG_MAPPING`` rather
   than by patching the vendored checkout, so ``third-party/gorilla`` stays a
   pristine upstream tree.
2. Calls BFCL's own ``generate`` then ``evaluate`` entry points -- the same
   functions the ``bfcl`` CLI calls, so scoring is upstream's, untouched.
3. Writes ``bfcl_summary.json``: per-category accuracy and counts plus a
   weighted overall, read back by the caller.

Prompting mode, not FC mode
---------------------------
The model is driven through ``/v1/completions`` with BFCL's default system
prompt, and must answer with ``[func(arg=val)]`` text that BFCL's AST checker
parses. That needs nothing from the server beyond plain completions -- no
``--tool-call-parser``, no ``--enable-auto-tool-choice`` -- which matters
because the server here is ``reap.expert_failure_server`` running with
``--enforce-eager`` and a routing mask, and every server-side option added is
another thing that has to be identical between baseline and masked cells.
Scores in this mode are lower than the leaderboard's ``glm-4.5-air-FC`` row;
that is fine and expected, because what the sweep measures is the *delta*
between an unmasked and a masked run of this exact configuration.

Thinking is disabled (``enable_thinking=False`` in the chat template, which
GLM-4.5 renders as ``/nothink`` plus a pre-filled empty ``<think></think>``),
matching what ``reap.eval`` already does for GLM elsewhere. A 1390-entry
non-live sweep with reasoning traces enabled would cost several times more per
cell for a benchmark whose answer is a single function call.

Usage
-----
::

    .venv-bfcl/bin/python src/reap/bfcl_client \\
        --model-path /sms-scratch/checkpoints/GLM-4.5-Air \\
        --host 0.0.0.0 --port 8000 \\
        --project-root artifacts/bfcl-run \\
        --test-category non_live
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from types import SimpleNamespace

# The registry name BFCL knows this model by. It must contain no underscore:
# BFCL round-trips the name through `replace("/", "_")` for directory paths and
# `replace("_", "/")` to look the config back up
# (eval_runner_helper.generate_leaderboard_csv), so an underscore anywhere else
# in the name comes back as a slash and the lookup raises KeyError.
REGISTRY_NAME = "reap-local/served-model"

SUMMARY_FILENAME = "bfcl_summary.json"


def register_model(model_path: str, display_name: str) -> None:
    """Add ``REGISTRY_NAME`` to BFCL's model registry, in this process only.

    ``model_name`` is what BFCL sends as the OpenAI ``model`` field, and it is
    overwritten with the ``--local-model-path`` value by
    ``OSSHandler.spin_up_local_server``, so the served model name must equal
    that path exactly (vLLM matches served names literally -- a trailing slash
    is a different name and yields a 404 on every request).
    """
    from bfcl_eval.constants import model_config as model_config_module
    from bfcl_eval.constants.model_config import ModelConfig

    # Sibling module; `python <this dir>` puts the directory on sys.path[0].
    # Imported here rather than at module level so this file stays importable
    # without bfcl_eval installed -- `read_scores` is unit-tested from the reap
    # environment, where it is not.
    from handler import LocalChatTemplateHandler

    config = ModelConfig(
        model_name=model_path,
        display_name=display_name,
        url="local",
        org="local",
        license="local",
        model_handler=LocalChatTemplateHandler,
        input_price=None,
        output_price=None,
        # Prompting mode: BFCL puts the function docs in the system prompt and
        # parses function calls back out of free text, so the server needs no
        # tool-call parser. See the module docstring.
        is_fc_model=False,
        underscore_to_dot=False,
    )
    model_config_module.MODEL_CONFIG_MAPPING[REGISTRY_NAME] = config

    # `local_inference_model_map` is what upstream merges into the mapping; keep
    # the two in agreement in case a code path consults it directly.
    local_map = getattr(model_config_module, "local_inference_model_map", None)
    if isinstance(local_map, dict):
        local_map[REGISTRY_NAME] = config


def read_scores(score_dir: pathlib.Path, categories: list[str]) -> dict:
    """Collect per-category accuracy from BFCL's score files.

    BFCL writes ``<score_dir>/<model>/<group>/BFCL_v4_<category>_score.json``
    as JSON *lines*, where line 1 is a header holding ``accuracy``,
    ``correct_count`` and ``total_count`` and the remaining lines are the
    individual failures (``eval_runner_helper.save_eval_results``). Only the
    header is needed here.

    A missing or unparseable file is reported as ``None`` rather than skipped,
    so the caller can tell "this category scored 0" apart from "this category
    never ran" -- the distinction the sweep's verification depends on.
    """
    model_dir = score_dir / REGISTRY_NAME.replace("/", "_")
    per_category: dict[str, dict | None] = {}
    for category in categories:
        matches = sorted(model_dir.glob(f"**/BFCL_v4_{category}_score.json"))
        if not matches:
            per_category[category] = None
            continue
        with open(matches[-1]) as handle:
            first_line = handle.readline()
        try:
            header = json.loads(first_line)
        except json.JSONDecodeError:
            per_category[category] = None
            continue
        per_category[category] = {
            "accuracy": header.get("accuracy"),
            "correct_count": header.get("correct_count"),
            "total_count": header.get("total_count"),
            "score_file": str(matches[-1]),
        }

    scored = [v for v in per_category.values() if v and v.get("total_count")]
    total_count = sum(v["total_count"] for v in scored)
    correct = sum(v.get("correct_count") or 0 for v in scored)
    return {
        "categories": per_category,
        # Entry-weighted, i.e. correct/total pooled over categories. This is
        # NOT the leaderboard's "Non-Live AST" number, which averages the four
        # AST categories unweighted; pooling is the right choice for a
        # sensitivity study because it is a single binomial proportion whose
        # noise floor is computable.
        "overall_accuracy": (correct / total_count) if total_count else None,
        "correct_count": correct,
        "total_count": total_count,
        "num_categories_scored": len(scored),
        "num_categories_requested": len(categories),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        required=True,
        help="checkpoint directory, used for the tokenizer/config AND as the "
        "OpenAI `model` name. Must match the server's served-model-name "
        "exactly, trailing slash included (i.e. excluded).",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--project-root",
        required=True,
        help="BFCL writes result/, score/ and .file_locks/ under here. One per "
        "sweep cell keeps cells from reading each other's generations.",
    )
    parser.add_argument(
        "--test-category",
        nargs="+",
        default=["non_live"],
        help="BFCL categories or collections. `non_live` is the 7-category, "
        "1390-entry V1 set (simple python/java/js, multiple, parallel, "
        "parallel_multiple, irrelevance).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.001,
        help="BFCL's own default. Not 0.0: several handlers treat 0 as unset.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=32,
        help="concurrent in-flight requests. Keep at or below the server's "
        "--max-num-seqs; BFCL's own default of 100 would queue.",
    )
    parser.add_argument(
        "--display-name",
        default="Served model (Prompt) (local)",
        help="cosmetic; appears in BFCL's leaderboard CSVs",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="leave GLM's reasoning mode on. Off by default -- see the module "
        "docstring. Whatever is chosen must be identical across every cell.",
    )
    parser.add_argument(
        "--stop-token-ids",
        nargs="*",
        type=int,
        default=[],
        help="passed to vLLM as stop_token_ids, e.g. GLM-4.5's "
        "151329 151336 151338",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="regenerate entries that already have results. Without this a "
        "re-run resumes, which is what makes an interrupted cell cheap.",
    )
    args = parser.parse_args()

    project_root = pathlib.Path(args.project_root).resolve()
    project_root.mkdir(parents=True, exist_ok=True)

    # Must be set before bfcl_eval.constants.eval_config is imported: it reads
    # BFCL_PROJECT_ROOT at import time and creates result/, score/ and
    # .file_locks/ underneath it as a side effect. Setting it afterwards would
    # silently write into the vendored checkout instead.
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)
    # How OSSHandler finds the server when --skip-server-setup is passed.
    os.environ["LOCAL_SERVER_ENDPOINT"] = args.host
    os.environ["LOCAL_SERVER_PORT"] = str(args.port)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # The handler reads these in __init__, and BFCL is what constructs it, so
    # they have to be set before generation starts.
    import handler as handler_module

    handler_module.ENABLE_THINKING = bool(args.enable_thinking)
    handler_module.STOP_TOKEN_IDS = tuple(args.stop_token_ids)

    from bfcl_eval._llm_response_generation import main as generation_main
    from bfcl_eval.eval_checker.eval_runner import main as evaluation_main
    from bfcl_eval.utils import parse_test_category_argument

    register_model(args.model_path, args.display_name)

    categories = list(parse_test_category_argument(list(args.test_category)))
    print(f"BFCL categories ({len(categories)}): {categories}", flush=True)

    generation_args = SimpleNamespace(
        model=[REGISTRY_NAME],
        test_category=list(args.test_category),
        temperature=args.temperature,
        include_input_log=False,
        exclude_state_log=False,
        num_gpus=1,
        num_threads=args.num_threads,
        gpu_memory_utilization=0.9,
        backend="vllm",
        # The whole point: the server is already up, masked, and owned by the
        # caller. Without this BFCL would launch its own unmasked vLLM.
        skip_server_setup=True,
        local_model_path=args.model_path,
        # Resolved against BFCL_PROJECT_ROOT by generation_main.
        result_dir="result",
        allow_overwrite=args.allow_overwrite,
        run_ids=False,
        enable_lora=False,
        max_lora_rank=None,
        lora_modules=None,
    )
    generation_main(generation_args)

    evaluation_main(
        model=[REGISTRY_NAME],
        test_categories=list(args.test_category),
        result_dir="result",
        score_dir="score",
    )

    summary = read_scores(project_root / "score", categories)
    summary["model_path"] = args.model_path
    summary["registry_name"] = REGISTRY_NAME
    summary["test_category"] = list(args.test_category)
    summary["resolved_categories"] = categories
    summary["enable_thinking"] = bool(args.enable_thinking)
    summary["temperature"] = args.temperature
    summary["server"] = f"http://{args.host}:{args.port}"

    summary_path = project_root / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}", flush=True)
    print(json.dumps(summary["categories"], indent=2), flush=True)
    print(
        f"overall (entry-weighted): {summary['overall_accuracy']} "
        f"over {summary['total_count']} entries",
        flush=True,
    )

    missing = [name for name, value in summary["categories"].items() if value is None]
    if missing:
        # Non-zero exit so the caller fails the cell rather than recording a
        # partial score as if it were complete.
        print(f"ERROR: no score produced for categories: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
