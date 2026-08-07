"""vLLM OpenAI server with a runtime control plane for EP-node-failure masking.

Why this exists
---------------
The sweep varies the dead-expert set across 32 nodes (and up to 3 failure
semantics). Restarting the server per cell would cost a full reload of a ~1 TB
checkpoint each time -- 15-25 min x 96 cells is roughly 24-40 h of pure
startup, which would dominate the experiment. Instead this launcher adds two
routes to vLLM's own FastAPI app that broadcast the mask to every TP worker via
``collective_rpc``, so an entire sweep runs on **one** server boot.

Usage
-----
Drop-in replacement for ``vllm serve``; all vLLM CLI flags are passed through::

    python -m reap.expert_failure_server \\
        --model moonshotai/Kimi-K2.6 --tensor-parallel-size 8 \\
        --enable-expert-parallel --trust-remote-code --port 8000

Then, per sweep cell::

    POST /reap/failure  {"mode": "drop", "node_id": 3}
    GET  /reap/failure                    -> current mask + routing counters

Routes are registered on ``vllm.entrypoints.openai.api_server.router`` before
``run_server`` builds the app, which is how vLLM's own endpoints are declared.
Handlers reach the engine through ``raw_request.app.state.engine_client``,
matching the pattern used by the built-in routes.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import uvloop
from fastapi import Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.api_server import router, run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser

from reap import expert_failure

logger = logging.getLogger(__name__)


def _merge_worker_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-worker replies into one response.

    Every worker runs identical routing, so the *mask* must agree across all of
    them -- a disagreement means the broadcast partially failed and any eval
    run afterwards would be silently invalid, so it is surfaced rather than
    smoothed over. The *counters* are per-worker and are summed, except for the
    derived rates which are recomputed from the summed totals.
    """
    if not results:
        return {"error": "no workers responded"}

    masks = {(r.get("mode"), tuple(r.get("dead_experts", []))) for r in results}
    merged: dict[str, Any] = {
        "num_workers": len(results),
        "mask_consistent": len(masks) == 1,
        **{k: results[0].get(k) for k in ("mode", "dead_experts", "num_dead", "patched")},
    }
    if len(masks) != 1:
        merged["per_worker"] = results
        return merged

    totals = {
        key: sum(float(r.get(key, 0.0)) for r in results)
        for key in ("calls", "tokens", "slots", "dead_slots", "gate_mass", "lost_gate_mass")
        if key in results[0]
    }
    if totals:
        merged.update(totals)
        merged["dead_slot_rate"] = (
            totals["dead_slots"] / totals["slots"] if totals.get("slots") else 0.0
        )
        merged["lost_gate_mass_frac"] = (
            totals["lost_gate_mass"] / totals["gate_mass"]
            if totals.get("gate_mass")
            else 0.0
        )
    return merged


@router.post("/reap/failure")
async def set_expert_failure(raw_request: Request) -> JSONResponse:
    """Set the failure mode and dead-expert set across all workers.

    Body: ``{"mode": "drop"|"drop_renorm"|"reroute"|"off",
             "node_id": int | null, "expert_ids": [int] | null,
             "num_experts": int, "num_nodes": int}``

    Rejects a request whose mask does not land identically on every worker: a
    partially-applied mask would produce eval numbers that look plausible but
    correspond to no real failure scenario.
    """
    body = await raw_request.json()
    kwargs: dict[str, Any] = {"mode": body.get("mode", "off")}
    for key in ("node_id", "expert_ids", "num_experts", "num_nodes"):
        if body.get(key) is not None:
            kwargs[key] = body[key]

    engine = raw_request.app.state.engine_client
    try:
        results = await engine.collective_rpc(
            expert_failure.rpc_set_failure, kwargs=kwargs
        )
    except Exception as exc:
        logger.exception("failed to broadcast expert failure mask")
        return JSONResponse(status_code=500, content={"error": str(exc)})

    merged = _merge_worker_results(list(results))
    if not merged.get("mask_consistent", False):
        return JSONResponse(status_code=500, content=merged)
    if kwargs["mode"] != "off" and not merged.get("patched", False):
        # The plugin did not install; without it the model runs unmasked and
        # the cell would be recorded as a failure result while being a baseline.
        return JSONResponse(
            status_code=500,
            content={
                "error": (
                    "FusedMoE.select_experts is not patched in the workers -- the "
                    "reap expert_failure plugin did not load. Check the "
                    "`vllm.general_plugins` entry point is installed."
                ),
                **merged,
            },
        )
    logger.info("expert failure set: %s", merged)
    return JSONResponse(content=merged)


@router.get("/reap/failure")
async def get_expert_failure(raw_request: Request) -> JSONResponse:
    """Report the active mask plus routing counters aggregated over all workers."""
    engine = raw_request.app.state.engine_client
    try:
        results = await engine.collective_rpc(expert_failure.rpc_get_stats)
    except Exception as exc:
        logger.exception("failed to collect expert failure stats")
        return JSONResponse(status_code=500, content={"error": str(exc)})
    return JSONResponse(content=_merge_worker_results(list(results)))


def _enable_callable_rpc() -> None:
    """Allow ``collective_rpc`` to ship a callable to the workers.

    vLLM v1 encodes RPC payloads with msgspec and refuses to serialize a plain
    function unless ``VLLM_ALLOW_INSECURE_SERIALIZATION=1``, which enables the
    cloudpickle fallback (``vllm/v1/serial_utils.py:125``). Without it every
    ``/reap/failure`` call fails with "Object of type <class 'function'> is not
    serializable" and no mask can ever be applied.

    The flag is read lazily through ``os.getenv`` (``vllm/envs.py:895``), and the
    engine-core process inherits ``os.environ`` when it is spawned below, so
    setting it here covers both. Set in the launcher rather than left to the
    caller because forgetting it breaks the control plane in a way that only
    shows up after the model has finished loading.

    "Insecure" here means cloudpickle instead of msgspec for RPC payloads. That
    matters when untrusted parties can reach the engine's RPC socket; for a
    local single-user experiment it is the documented way to pass a callable.
    """
    import os

    if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") not in (None, "", "0"):
        return
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    logger.info(
        "Set VLLM_ALLOW_INSECURE_SERIALIZATION=1 so collective_rpc can broadcast "
        "the expert-failure mask (cloudpickle fallback for callable payloads)."
    )


def _force_eager(args) -> None:
    """Disable CUDA graphs, without which the routing mask silently does nothing.

    vLLM compiles the model (``compilation_config.level=3``) and captures CUDA
    graphs. ``FusedMoE.select_experts`` is patched at plugin-load time, i.e.
    *before* capture, so the wrapper is traced into the graph with whatever
    ``_MODE`` held at trace time -- ``off``. Replays then never re-enter Python,
    so later ``set_failure`` calls cannot affect execution and the invocation
    counter stays at 0 no matter how much traffic is served.

    Measured on Qwen3-30B-A3B: with graphs on, ``calls`` stayed 0 across a full
    generation while the mask reported 64 dead experts. With ``enforce_eager``
    the same request gives ``calls=2304`` (48 MoE layers x 48 forwards) and a
    dead-slot rate matching the mask.

    This is forced rather than warned about because the failure is silent in the
    worst direction: every masked cell would score like an unmasked baseline and
    the sweep would conclude "no degradation". Set ``REAP_ALLOW_CUDAGRAPH=1`` to
    override -- only meaningful once the mask is made graph-safe (device-resident
    mask buffer mutated in place, no Python-side branching).
    """
    import os

    if os.environ.get("REAP_ALLOW_CUDAGRAPH") not in (None, "", "0"):
        logger.warning(
            "REAP_ALLOW_CUDAGRAPH set: leaving CUDA graphs enabled. The routing "
            "mask will NOT take effect unless it has been made graph-safe. "
            "Verify with `sweep.py --preflight-only` before trusting any result."
        )
        return
    if getattr(args, "enforce_eager", False):
        return
    args.enforce_eager = True
    logger.warning(
        "Forcing --enforce-eager: CUDA graph replay bypasses the patched "
        "FusedMoE.select_experts, which would make every masked run silently "
        "identical to the baseline. Set REAP_ALLOW_CUDAGRAPH=1 to override."
    )


def main() -> None:
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI server with the REAP expert-failure control plane"
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    _enable_callable_rpc()
    _force_eager(args)
    logger.info(
        "Starting vLLM server with REAP expert-failure routes at "
        "GET/POST /reap/failure"
    )
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
