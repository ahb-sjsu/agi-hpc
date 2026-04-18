"""Fan out compiler-module authorship across NRP pods.

One pod per failure cluster; each pod runs compile_attempt.py which:
  1. Asks Qwen 397B to write a compile_X() for that cluster
  2. Runtime-tests the module against the cluster's tasks
  3. Emits JSON result + (if promoted) module source between markers

This module generates the Kubernetes Job manifests, submits them with
kubectl apply, and collects results by scanning pod logs.

Resource posture is strictly lightweight (cpu=1, memory=2Gi) so we stay
in swarm mode per reference_nrp_cluster_policy — no GPU needed since
compile_attempt is pure code synthesis + CPU onnxruntime verification.

Usage:
  from nrp_compiler_swarm import run_swarm
  promoted = run_swarm(clusters, namespace="ssu-atlas-ai",
                       bundle="ahb-sjsu/neurogolf-bundle",
                       max_concurrent=8)
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("nrp_swarm")

JOB_TEMPLATE = """\
apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: {namespace}
spec:
  ttlSecondsAfterFinished: 300
  backoffLimit: 0
  template:
    metadata:
      labels:
        app: erebus-compiler-author
    spec:
      restartPolicy: Never
      containers:
      - name: author
        image: gcr.io/kaggle-images/python:latest
        resources:
          requests: {{cpu: "1", memory: "2Gi"}}
          limits:   {{cpu: "1", memory: "2Gi"}}
        env:
        - name: NRP_LLM_TOKEN
          value: "{token}"
        - name: CLUSTER_JSON
          value: {cluster_json}
        - name: TASK_DIR
          value: /work/tasks
        - name: COMPILER_DIR
          value: /work/src/compiler
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -e
          mkdir -p /work && cd /work
          git clone --depth 1 https://github.com/{bundle}.git bundle
          tar -C /work -xzf bundle/tasks.tar.gz || true
          cp -r bundle/src /work/src
          pip install --quiet openai onnx onnxruntime numpy
          python3 -u /work/src/agi/autonomous/compile_attempt.py
"""


@dataclass
class SwarmResult:
    cluster_pattern: str
    tag: str
    promoted: bool
    solved_ratio: float | None
    module_source: str | None
    pod_logs: str  # last 2KB for debugging


def _safe_job_name(pattern: str, idx: int) -> str:
    # k8s job names: lowercase, alphanumeric + hyphens, <= 63 chars
    p = re.sub(r"[^a-z0-9-]", "-", pattern.lower())[:30].strip("-") or "cluster"
    ts = datetime.now(timezone.utc).strftime("%H%M%S")
    return f"ca-{p}-{idx}-{ts}"[:63]


def submit_jobs(clusters: list[dict], namespace: str, bundle: str,
                token: str, sample_codes_by_cluster: dict | None = None,
                max_concurrent: int = 4) -> list[str]:
    """Submit one Job per cluster. Returns job names.

    NRP policy: <= 4 heavy pods, OR 5+ all-lightweight. These pods are
    lightweight (no GPU) so any count is fine, but cap to `max_concurrent`
    to be polite.
    """
    if not clusters:
        return []
    job_names = []
    for idx, cluster in enumerate(clusters[:max_concurrent]):
        # Embed any sample codes from day's successes that match this
        # cluster's pattern, so the LLM sees real examples.
        cluster_copy = dict(cluster)
        if sample_codes_by_cluster:
            cluster_copy["sample_codes"] = sample_codes_by_cluster.get(
                cluster.get("pattern", ""), [])[:3]

        job_name = _safe_job_name(cluster.get("pattern", "cluster"), idx)
        job_names.append(job_name)

        manifest = JOB_TEMPLATE.format(
            name=job_name,
            namespace=namespace,
            token=token,
            # CLUSTER_JSON must be a quoted string in YAML; escape safely
            cluster_json=json.dumps(json.dumps(cluster_copy)),
            bundle=bundle,
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml",
                                          delete=False) as f:
            f.write(manifest)
            path = f.name
        try:
            r = subprocess.run(["kubectl", "apply", "-f", path],
                               capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                log.warning(f"kubectl apply failed for {job_name}: {r.stderr[:300]}")
            else:
                log.info(f"submitted {job_name}")
        finally:
            Path(path).unlink(missing_ok=True)
    return job_names


def wait_and_collect(job_names: list[str], namespace: str,
                     timeout: int = 1200, poll: int = 20) -> list[SwarmResult]:
    """Poll kubectl until all jobs Complete or Fail, then harvest pod logs."""
    if not job_names:
        return []
    done: set[str] = set()
    deadline = time.time() + timeout
    while job_names and set(job_names) - done and time.time() < deadline:
        r = subprocess.run(
            ["kubectl", "-n", namespace, "get", "jobs",
             "-o", "json", "--field-selector=metadata.name in ("
             + ",".join(job_names) + ")"],
            capture_output=True, text=True, timeout=30)
        # "in" selector isn't universal — fall back to per-job status
        for jn in job_names:
            if jn in done:
                continue
            rr = subprocess.run(["kubectl", "-n", namespace, "get", "job",
                                 jn, "-o", "jsonpath={.status.conditions[0].type}"],
                                capture_output=True, text=True, timeout=15)
            if rr.stdout.strip() in ("Complete", "Failed"):
                done.add(jn)
        if len(done) < len(job_names):
            time.sleep(poll)

    # Harvest logs
    results: list[SwarmResult] = []
    for jn in job_names:
        log_out = subprocess.run(
            ["kubectl", "-n", namespace, "logs", "job/" + jn, "--tail=500"],
            capture_output=True, text=True, timeout=30).stdout
        # Extract module source
        module = None
        if "<MODULE_BEGIN>" in log_out and "<MODULE_END>" in log_out:
            s = log_out.index("<MODULE_BEGIN>") + len("<MODULE_BEGIN>")
            e = log_out.index("<MODULE_END>")
            module = log_out[s:e].strip()
        # Extract final JSON summary
        final = None
        for line in reversed(log_out.strip().split("\n")):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    final = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        if final is None:
            results.append(SwarmResult(
                cluster_pattern=jn, tag="", promoted=False,
                solved_ratio=None, module_source=module,
                pod_logs=log_out[-2000:]))
        else:
            results.append(SwarmResult(
                cluster_pattern=final.get("cluster", jn),
                tag=final.get("tag", ""),
                promoted=final.get("promoted", False),
                solved_ratio=final.get("solved_ratio"),
                module_source=module,
                pod_logs=log_out[-500:]))
    return results


def persist_promoted(results: list[SwarmResult],
                     compiler_dir: Path) -> list[Path]:
    """Write each promoted module to compiler_dir. Returns saved paths."""
    compiler_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for r in results:
        if r.promoted and r.module_source and r.tag:
            path = compiler_dir / f"dream_{r.tag}.py"
            path.write_text(r.module_source)
            saved.append(path)
            log.info(f"persisted {path} (solved_ratio={r.solved_ratio})")
    return saved


def run_swarm(clusters: list[dict], namespace: str, bundle: str,
              compiler_dir: Path, sample_codes_by_cluster: dict | None = None,
              max_concurrent: int = 4) -> list[Path]:
    """Full pipeline: submit, wait, collect, persist promoted modules."""
    token = os.environ.get("NRP_LLM_TOKEN", "")
    if not token:
        log.warning("No NRP_LLM_TOKEN — cannot submit compiler swarm.")
        return []
    jobs = submit_jobs(clusters, namespace, bundle, token,
                       sample_codes_by_cluster, max_concurrent)
    results = wait_and_collect(jobs, namespace)
    return persist_promoted(results, compiler_dir)
