#!/bin/bash
# One-time setup for the BFCL (Berkeley Function-Calling Leaderboard) harness.
#
# BFCL gets its own interpreter rather than sharing the reap environment:
# `bfcl_eval` pins numpy==1.26.4 against reap's numpy 2.x (required by vLLM
# 0.10), and it pulls in a large vendor-SDK set (mistralai, anthropic, cohere,
# faiss-cpu, sentence-transformers) that has no business next to vLLM. BFCL only
# ever talks to the model over HTTP, so one interpreter per dependency set is the
# clean split. See src/reap/bfcl.py.
#
# Run from the repo root, after scripts/build.sh:
#
#   bash scripts/setup_bfcl.sh
#
# Idempotent -- safe to re-run. Override the target with BFCL_VENV=path.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BFCL_VENV="${BFCL_VENV:-$REPO_ROOT/.venv-bfcl}"
GORILLA_DIR="$REPO_ROOT/third-party/gorilla"
BFCL_PKG="$GORILLA_DIR/berkeley-function-call-leaderboard"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found. Install it first (see README Installation)." >&2
    exit 1
fi

# 1. The BFCL source. Sparse, because the gorilla monorepo holds several
#    unrelated projects and only berkeley-function-call-leaderboard is wanted.
#    Upstream is used UNMODIFIED -- the local model entry is registered
#    in-process by src/reap/bfcl_client, so this stays `git pull`-able.
#    Blobless rather than shallow: the submodule is pinned to a specific commit,
#    and a --depth 1 fetch only gets the tip of the default branch -- so it works
#    only while the pin happens to BE the tip, and fails as soon as upstream
#    lands anything. --filter=blob:none keeps it small without that trap.
echo "==> fetching the gorilla submodule (sparse)"
git submodule update --init --filter=blob:none third-party/gorilla
git -C "$GORILLA_DIR" sparse-checkout init --cone 2>/dev/null || true
git -C "$GORILLA_DIR" sparse-checkout set berkeley-function-call-leaderboard

if [ ! -f "$BFCL_PKG/pyproject.toml" ]; then
    echo "error: $BFCL_PKG/pyproject.toml missing after sparse checkout." >&2
    exit 1
fi

# 2. The isolated interpreter.
echo "==> creating $BFCL_VENV"
uv venv --python 3.12 "$BFCL_VENV"

echo "==> installing bfcl_eval (editable) -- a few minutes on a cold cache"
uv pip install --python "$BFCL_VENV" -e "$BFCL_PKG"

# `bfcl_eval.constants.model_config` imports EVERY vendor handler at module
# load, so one missing transitive dep breaks the whole model registry. qwen-agent
# imports soundfile but does not declare it; without this, any BFCL run dies with
# ModuleNotFoundError before it reaches the model.
echo "==> installing soundfile (undeclared transitive dep of qwen-agent)"
uv pip install --python "$BFCL_VENV" soundfile

# 3. Prove it. Importing the registry is the check that matters -- it is what
#    fails when a vendor handler's dependency is missing.
echo "==> verifying"
"$BFCL_VENV/bin/python" - <<'PY'
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.utils import parse_test_category_argument

categories = parse_test_category_argument(["non_live"])
print(f"  model registry loaded: {len(MODEL_CONFIG_MAPPING)} models")
print(f"  non_live resolves to {len(categories)} categories: {categories}")
assert len(categories) == 7, categories
PY

echo
echo "BFCL harness ready. Interpreter: $BFCL_VENV"
echo "Next: serve a model, then see experiments/node-failure/BFCL-RUNBOOK.md"
