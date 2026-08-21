#!/usr/bin/env bash
#
# Launch the full node-failure sweep across all GPUs on one box.
#
# One server per TP group, and the cell list is split that many ways with
# --shard. Qwen3-30B-A3B is ~61 GB in bf16 so TP=1 gives 8 shards on an 8-GPU
# box; GLM-4.5-Air is ~214 GB and needs TP=2, giving 4. Each shard preflights
# its own server before doing any work: without that a server whose routing
# hook never fires produces cells that look like clean baselines.
#
# Resumable. A cell with a cell.json is skipped, so re-running after an
# interruption picks up where it stopped, and shards never redo each other's
# work. Safe to Ctrl-C and relaunch.
#
# Usage:
#   bash experiments/node-failure/launch_sweep.sh
#   MODES="reroute" NGPU=4 bash experiments/node-failure/launch_sweep.sh
#
#   # GLM-4.5-Air on LiveCodeBench, 4 shards of TP=2:
#   MODEL=/sms-scratch/checkpoints/GLM-4.5-Air TP=2 BENCHMARK=livecodebench \
#   MODES=reroute BASELINE_REPEATS=3 RESULTS=artifacts/glm-node-failure/livecodebench \
#       bash experiments/node-failure/launch_sweep.sh
#
set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL="${MODEL:-/home/ravira/checkpoints/Qwen3-30B-A3B}"   # no trailing slash: vLLM matches served name exactly
RESULTS="${RESULTS:-artifacts/node-failure}"
MODES="${MODES:-drop_renorm reroute}"
NGPU="${NGPU:-8}"
TP="${TP:-1}"                                              # GPUs per server; NGPU/TP servers are started
BENCHMARK="${BENCHMARK:-math_500}"                         # math_500 | bfcl | livecodebench
BASE_PORT="${BASE_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"                    # must match the baselines' setting
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"                         # keep >= the benchmark's request concurrency
BASELINE_REPEATS="${BASELINE_REPEATS:-2}"                  # existing baseline cells are skipped, not rerun
SWEEP_ARGS="${SWEEP_ARGS:-}"                               # extra flags passed through to sweep.py
PY="${PY:-.venv/bin/python}"

NSERVERS=$((NGPU / TP))
if [ "$NSERVERS" -lt 1 ]; then
    echo "TP=$TP exceeds NGPU=$NGPU"
    exit 1
fi

mkdir -p "$RESULTS/logs"

echo "=== node-failure sweep ==="
echo "  model     : $MODEL"
echo "  benchmark : $BENCHMARK"
echo "  modes     : $MODES"
echo "  servers   : $NSERVERS x TP=$TP over $NGPU gpu(s) (ports $BASE_PORT-$((BASE_PORT + NSERVERS - 1)))"
echo "  results   : $RESULTS"
echo

# --- stop anything already serving on these ports -------------------------
# pkill -f with a plain pattern also matches this script's own command line;
# the bracket makes the pattern not match its own literal text.
echo "stopping existing servers..."
pkill -f "reap[.]expert_failure_server --model" 2>/dev/null
for _ in $(seq 1 30); do
    pgrep -f "reap[.]expert_failure_server --model" >/dev/null || break
    sleep 2
done
sleep 5

# --- start one server per GPU ---------------------------------------------
# --enforce-eager is also forced inside the launcher; passed here so the
# requirement is visible in `ps`. Without it vLLM captures CUDA graphs after
# the plugin patched select_experts, the hook never re-enters Python, and every
# masked cell silently scores as a baseline.
for i in $(seq 0 $((NSERVERS - 1))); do
    port=$((BASE_PORT + i))
    # Contiguous TP-sized slice of the visible devices, e.g. "2,3" for i=1,TP=2.
    devices=$(seq -s, $((i * TP)) $((i * TP + TP - 1)))
    tp_flags=""
    if [ "$TP" -gt 1 ]; then
        tp_flags="--tensor-parallel-size $TP --enable-expert-parallel"
    fi
    echo "starting server gpus=$devices port=$port"
    CUDA_VISIBLE_DEVICES=$devices nohup "$PY" -m reap.expert_failure_server \
        --model "$MODEL" --port "$port" $tp_flags \
        --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
        --trust-remote-code --enforce-eager \
        > "$RESULTS/logs/server_${port}.log" 2>&1 < /dev/null &
    disown
done

# --- wait for all of them to come up --------------------------------------
echo
echo "waiting for servers to load (~2-6 min)..."
for i in $(seq 0 $((NSERVERS - 1))); do
    port=$((BASE_PORT + i))
    for attempt in $(seq 1 240); do
        if curl -sf "http://0.0.0.0:${port}/health" >/dev/null 2>&1; then
            echo "  port $port ready"
            break
        fi
        if [ "$attempt" -eq 240 ]; then
            echo "  port $port FAILED to start -- see $RESULTS/logs/server_${port}.log"
            tail -20 "$RESULTS/logs/server_${port}.log"
            exit 1
        fi
        sleep 5
    done
done

# --- run the shards --------------------------------------------------------
# --preflight (not --preflight-only) validates this shard's server, then sweeps.
echo
echo "launching $NSERVERS sweep shards..."
pids=()
for i in $(seq 0 $((NSERVERS - 1))); do
    port=$((BASE_PORT + i))
    "$PY" experiments/node-failure/sweep.py \
        --model "$MODEL" --port "$port" \
        --results-dir "$RESULTS" \
        --benchmark "$BENCHMARK" \
        --modes $MODES \
        --baseline-repeats "$BASELINE_REPEATS" \
        --shard "$i" --num-shards "$NSERVERS" \
        --preflight $SWEEP_ARGS \
        > "$RESULTS/logs/sweep_shard${i}.log" 2>&1 &
    pids+=($!)
done

echo "shards running. follow with:"
echo "  tail -f $RESULTS/logs/sweep_shard0.log"
echo "  find $RESULTS -name cell.json | wc -l   # completed cells"
echo

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done

echo
completed=$(find "$RESULTS" -name cell.json | wc -l)
echo "=== done: $completed cell(s) complete ==="
if [ "$fail" -ne 0 ]; then
    echo "at least one shard exited non-zero -- check $RESULTS/logs/sweep_shard*.log"
    echo "re-running this script resumes; finished cells are skipped."
    exit 1
fi
