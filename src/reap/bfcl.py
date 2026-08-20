"""Run BFCL against an existing vLLM server from inside the reap environment.

BFCL (``bfcl_eval``) cannot be imported here: it pins ``numpy==1.26.4`` against
this environment's numpy 2.x, and its dependency set (``mistralai``,
``anthropic``, ``cohere``, ``faiss-cpu``, ``sentence-transformers``, ...) has no
place next to vLLM. Since BFCL is purely an HTTP client against the OpenAI
endpoint, it gets its own interpreter -- ``.venv-bfcl`` -- and this module is
the thin bridge: build the command line, run it, read the summary back.

Setup, once::

    uv venv --python 3.12 .venv-bfcl
    uv pip install --python .venv-bfcl -e \\
        third-party/gorilla/berkeley-function-call-leaderboard

The actual work happens in ``reap/bfcl_client/__main__.py``, which runs under
that interpreter. See its docstring for why the run is in prompting mode and
why thinking is off.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import subprocess
import sys

logger = logging.getLogger(__name__)

# Repo root, i.e. the parent of `src/`.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

DEFAULT_BFCL_PYTHON = REPO_ROOT / ".venv-bfcl" / "bin" / "python"
CLIENT_DIR = pathlib.Path(__file__).resolve().parent / "bfcl_client"

SUMMARY_FILENAME = "bfcl_summary.json"

# GLM-4.5 / GLM-4.5-Air EOS ids (config.json `eos_token_id`). Passed to vLLM as
# `stop_token_ids` so a completion ends at `<|user|>`/`<|observation|>` too,
# not only at the primary EOS.
GLM45_STOP_TOKEN_IDS = (151329, 151336, 151338)

# The 7-category, 1390-entry non-live set: the original (V1) BFCL AST benchmark
# plus irrelevance. `simple_python` 400, `simple_java` 100,
# `simple_javascript` 50, `multiple` 200, `parallel` 200,
# `parallel_multiple` 200, `irrelevance` 240.
NON_LIVE_CATEGORIES = (
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
)
EXPECTED_NON_LIVE_N = 1390


def resolve_bfcl_python(bfcl_python: str | os.PathLike | None = None) -> pathlib.Path:
    """Locate the interpreter that has ``bfcl_eval`` installed.

    Fails with the exact setup command rather than letting the subprocess die
    with ``ModuleNotFoundError`` several minutes into a sweep.
    """
    candidate = pathlib.Path(bfcl_python or DEFAULT_BFCL_PYTHON)
    if not candidate.exists():
        raise FileNotFoundError(
            f"no BFCL interpreter at {candidate}. BFCL needs its own "
            f"environment (it pins numpy 1.26 against this one's numpy 2.x):\n"
            f"  uv venv --python 3.12 {REPO_ROOT / '.venv-bfcl'}\n"
            f"  uv pip install --python {REPO_ROOT / '.venv-bfcl'} -e "
            f"{REPO_ROOT / 'third-party/gorilla/berkeley-function-call-leaderboard'}"
        )
    return candidate


def run_bfcl(
    model_path: str | os.PathLike,
    results_dir: str | os.PathLike,
    server_url: str,
    test_categories: list[str] | tuple[str, ...] = ("non_live",),
    bfcl_python: str | os.PathLike | None = None,
    num_threads: int = 32,
    temperature: float = 0.001,
    enable_thinking: bool = False,
    stop_token_ids: list[int] | tuple[int, ...] | None = None,
    allow_overwrite: bool = False,
    log_file: str | os.PathLike | None = None,
    timeout: float | None = None,
) -> dict:
    """Run BFCL against ``server_url`` and return the parsed summary.

    ``results_dir`` becomes ``BFCL_PROJECT_ROOT``: BFCL's ``result/``,
    ``score/`` and ``.file_locks/`` all land under it. Giving each sweep cell
    its own root is what keeps a masked cell from being scored against the
    baseline's cached generations -- BFCL skips test entries that already have a
    result file, so a shared root would silently reuse them.

    ``model_path`` must equal the server's served-model-name exactly: BFCL sends
    it as the OpenAI ``model`` field and vLLM matches served names literally, so
    a trailing slash turns every request into a 404.

    Raises on a non-zero exit or a missing summary; the caller records a cell
    only after this returns.
    """
    python = resolve_bfcl_python(bfcl_python)
    project_root = pathlib.Path(results_dir).resolve() / "bfcl"
    project_root.mkdir(parents=True, exist_ok=True)

    host, port = _split_server_url(server_url)

    if stop_token_ids is None:
        # Only GLM needs these; for any other model leave vLLM to its own
        # generation_config rather than inventing stop ids.
        stop_token_ids = (
            GLM45_STOP_TOKEN_IDS if "glm-4.5" in str(model_path).lower() else ()
        )

    command: list[str] = [
        str(python),
        str(CLIENT_DIR),
        "--model-path",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--project-root",
        str(project_root),
        "--test-category",
        *list(test_categories),
        "--temperature",
        str(temperature),
        "--num-threads",
        str(num_threads),
    ]
    if enable_thinking:
        command.append("--enable-thinking")
    if stop_token_ids:
        command += ["--stop-token-ids", *[str(i) for i in stop_token_ids]]
    if allow_overwrite:
        command.append("--allow-overwrite")

    log_path = pathlib.Path(log_file) if log_file else project_root / "bfcl_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Running BFCL: %s", " ".join(command))
    logger.info("BFCL log -> %s", log_path)

    with open(log_path, "w") as handle:
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            # Inherit the environment but keep the reap venv off the child's
            # path: a stray PYTHONPATH pointing at this environment's
            # site-packages would let numpy 2.x shadow BFCL's pinned 1.26.
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            check=False,
        )

    summary_path = project_root / SUMMARY_FILENAME
    if completed.returncode != 0:
        raise RuntimeError(
            f"BFCL exited {completed.returncode}. Last lines of {log_path}:\n"
            + _tail(log_path)
        )
    if not summary_path.exists():
        raise RuntimeError(
            f"BFCL exited 0 but wrote no {SUMMARY_FILENAME} under {project_root}. "
            f"Last lines of {log_path}:\n" + _tail(log_path)
        )

    summary = json.loads(summary_path.read_text())
    summary["log"] = str(log_path)
    summary["summary_file"] = str(summary_path)
    return summary


def verify_bfcl_summary(summary: dict, expect_n: int | None = None) -> dict:
    """Confirm a BFCL summary describes a complete, comparable run.

    A cell whose run covered fewer entries than the baseline's is not
    comparable to it -- the difference in score would partly be a difference in
    which problems were attempted -- so a short run is an error, not a result.
    """
    missing = [name for name, value in summary.get("categories", {}).items() if not value]
    if missing:
        raise RuntimeError(f"BFCL produced no score for categories: {missing}")
    if summary.get("overall_accuracy") is None:
        raise RuntimeError("BFCL summary has no overall accuracy")
    if expect_n and summary.get("total_count") != expect_n:
        raise RuntimeError(
            f"BFCL scored {summary.get('total_count')} entries, expected "
            f"{expect_n}. A truncated run is not comparable to the baseline -- "
            f"pass --expect-n 0 to disable this check."
        )
    return summary


def _split_server_url(server_url: str) -> tuple[str, str]:
    """``http://0.0.0.0:8000`` or ``.../v1`` -> ``("0.0.0.0", "8000")``.

    BFCL wants host and port separately (it appends ``/v1`` itself), while the
    rest of the harness passes base URLs around.
    """
    url = server_url.strip().rstrip("/")
    for prefix in ("http://", "https://"):
        if url.startswith(prefix):
            url = url[len(prefix) :]
            break
    url = url.split("/", 1)[0]
    if ":" in url:
        host, port = url.rsplit(":", 1)
        return host, port
    return url, "8000"


def _tail(path: pathlib.Path, lines: int = 40) -> str:
    try:
        return "".join(path.read_text(errors="replace").splitlines(keepends=True)[-lines:])
    except OSError as exc:  # pragma: no cover - diagnostics only
        return f"(could not read {path}: {exc})"


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run BFCL against a live vLLM server")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--server-url", default="http://0.0.0.0:8000")
    parser.add_argument("--test-category", nargs="+", default=["non_live"])
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument(
        "--expect-n",
        type=int,
        default=EXPECTED_NON_LIVE_N,
        help="fail if the run did not score this many entries (0 disables)",
    )
    cli_args = parser.parse_args()

    result = run_bfcl(
        model_path=cli_args.model_path,
        results_dir=cli_args.results_dir,
        server_url=cli_args.server_url,
        test_categories=cli_args.test_category,
        num_threads=cli_args.num_threads,
        enable_thinking=cli_args.enable_thinking,
        allow_overwrite=cli_args.allow_overwrite,
    )
    verify_bfcl_summary(result, cli_args.expect_n or None)
    print(json.dumps(result, indent=2))
    sys.exit(0)
