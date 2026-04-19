"""Collect a GPU power trace by polling nvidia-smi at a fixed rate.

Writes one JSONL file per trace. Designed for building a library of
profile fingerprints that the DCGM attestation classifier can match
against — distinguish "GPU actually computed" from "cached replay"
by the shape of the power-draw curve, not just a single-point delta.

Produces:

    {
      "label": "idle_baseline",
      "gpu_index": 0,
      "started_at": "2026-04-19T20:40:00Z",
      "duration_s": 30.0,
      "interval_ms": 100,
      "samples": [
        {"t": 0.0, "power_w": 29.8, "util_pct": 0, "mem_used_mib": 347, "temp_c": 52},
        {"t": 0.1, "power_w": 29.9, "util_pct": 0, ...},
        ...
      ]
    }

Usage:

    python scripts/collect_gpu_power_trace.py \\
        --gpu 0 --label idle_baseline --duration 30 \\
        --out /archive/neurogolf/dcgm_profiles/idle_baseline.jsonl

    python scripts/collect_gpu_power_trace.py \\
        --gpu 0 --label active_cupy --duration 30 \\
        --workload cupy_matmul \\
        --out /archive/neurogolf/dcgm_profiles/active_cupy.jsonl

Workload modes:
    none            - just poll; caller runs the workload separately
    sleep           - sleeps for duration_s (models a cached-replay)
    cupy_matmul     - runs repeated cupy matmuls on the target GPU;
                      pre-allocates large tensors, drives sustained load
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("gpu_trace")


def _sample(gpu_index: int) -> dict[str, Any]:
    """One nvidia-smi sample. Returns an empty dict if the query fails so
    the outer loop never crashes."""
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.draw,utilization.gpu,memory.used,temperature.gpu",
                "--format=csv,noheader,nounits",
                f"--id={gpu_index}",
            ],
            capture_output=True,
            text=True,
            timeout=1.5,
        )
        if r.returncode != 0:
            return {}
        fields = [p.strip() for p in r.stdout.strip().split(",")]
        if len(fields) < 4:
            return {}
        return {
            "power_w": float(fields[0]),
            "util_pct": int(float(fields[1])),
            "mem_used_mib": int(float(fields[2])),
            "temp_c": int(float(fields[3])),
        }
    except Exception:
        return {}


def poll_trace(
    gpu_index: int,
    label: str,
    duration_s: float,
    interval_ms: int = 100,
) -> dict[str, Any]:
    """Block for ``duration_s`` polling nvidia-smi; return the trace dict."""
    t0 = time.time()
    samples: list[dict[str, Any]] = []
    next_tick = t0
    while time.time() - t0 < duration_s:
        now = time.time()
        if now < next_tick:
            time.sleep(max(0.0, next_tick - now))
        s = _sample(gpu_index)
        elapsed = time.time() - t0
        if s:
            s["t"] = round(elapsed, 3)
            samples.append(s)
        next_tick += interval_ms / 1000.0

    return {
        "label": label,
        "gpu_index": gpu_index,
        "started_at": datetime.fromtimestamp(t0, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "duration_s": round(time.time() - t0, 3),
        "interval_ms": interval_ms,
        "samples": samples,
    }


# ── workloads (optional — caller can also drive a workload separately) ──


def _workload_sleep(duration_s: float) -> None:
    """No-op workload — models a 'cached replay' where output is returned
    without the GPU actually computing."""
    time.sleep(duration_s)


def _workload_cupy_matmul(gpu_index: int, duration_s: float) -> None:
    """Sustained cupy matmul on the target GPU for ``duration_s``.

    Uses large tensors so SM util stays high. If cupy isn't available,
    falls back to a numpy-on-CPU workload and logs a warning — the
    resulting trace will show idle on the GPU, which is the correct
    attestation-negative outcome."""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        import cupy as cp  # type: ignore

        n = 4096
        a = cp.random.rand(n, n, dtype=cp.float32)
        b = cp.random.rand(n, n, dtype=cp.float32)
        t0 = time.time()
        while time.time() - t0 < duration_s:
            cp.matmul(a, b)
            cp.cuda.Stream.null.synchronize()
    except ImportError:
        log.warning("cupy not available; sleeping instead (workload_cupy_matmul)")
        time.sleep(duration_s)
    except Exception as e:
        log.warning("cupy_matmul failed (%s); sleeping", e)
        time.sleep(duration_s)


_WORKLOADS = {
    "none": None,
    "sleep": _workload_sleep,
    "cupy_matmul": _workload_cupy_matmul,
}


def collect(
    gpu_index: int,
    label: str,
    duration_s: float,
    workload: str = "none",
    interval_ms: int = 100,
) -> dict[str, Any]:
    """Run a workload (optional) concurrently with polling. Returns trace."""
    workload_fn = _WORKLOADS.get(workload)
    worker: threading.Thread | None = None
    if workload_fn is not None:
        if workload == "cupy_matmul":
            worker = threading.Thread(
                target=workload_fn, args=(gpu_index, duration_s), daemon=True
            )
        else:
            worker = threading.Thread(
                target=workload_fn, args=(duration_s,), daemon=True
            )
        worker.start()

    trace = poll_trace(gpu_index, label, duration_s, interval_ms)

    if worker is not None:
        worker.join(timeout=2.0)

    return trace


# ── CLI ──


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="GPU power-trace collector")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index")
    ap.add_argument("--label", required=True, help="Trace label (idle, active, ...)")
    ap.add_argument("--duration", type=float, default=30.0, help="Duration seconds")
    ap.add_argument("--interval-ms", type=int, default=100)
    ap.add_argument(
        "--workload",
        default="none",
        choices=sorted(_WORKLOADS.keys()),
        help="Optional in-process workload to drive the GPU",
    )
    ap.add_argument(
        "--out", required=True, help="Output JSONL path (trace overwrites file)"
    )
    args = ap.parse_args()

    log.info(
        "collecting: gpu=%d label=%s duration=%.1fs workload=%s → %s",
        args.gpu,
        args.label,
        args.duration,
        args.workload,
        args.out,
    )
    trace = collect(
        gpu_index=args.gpu,
        label=args.label,
        duration_s=args.duration,
        workload=args.workload,
        interval_ms=args.interval_ms,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trace, indent=2))
    # Quick summary to stdout
    n = len(trace["samples"])
    if n:
        avg = sum(s["power_w"] for s in trace["samples"]) / n
        peak = max(s["power_w"] for s in trace["samples"])
        util_avg = sum(s["util_pct"] for s in trace["samples"]) / n
        log.info(
            "done: %d samples  avg_power=%.1fW  peak=%.1fW  avg_util=%.0f%%",
            n,
            avg,
            peak,
            util_avg,
        )
    else:
        log.error("no samples collected")
        sys.exit(1)


if __name__ == "__main__":
    main()
