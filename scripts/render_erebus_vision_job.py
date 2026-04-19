"""Render a Kubernetes Job (not Deployment) for the GLM-4.1V vision batch.

Shape per NRP policy:
  - Max 4 heavy GPU pods (parallelism: 4)
  - GPU util must stay >40% → Jobs crunch through the queue continuously
    and exit cleanly when idle (NATS_EXIT_ON_IDLE_S).
  - Node affinity to L40 / L40S / A10 / V100 (explicitly not A100 —
    needs access form we don't have).

Dispatch flow:
  1. Atlas publishes N vision tasks to erebus.tasks.solve_task_vision
  2. kubectl apply -f <rendered>  → 4 parallel Jobs spawn
  3. Each Job loads GLM-4.1V once, drains the queue, exits
  4. K8s GCs the Jobs after ttlSecondsAfterFinished

Usage:
    python scripts/render_erebus_vision_job.py --parallelism 4 > job.yaml
    kubectl apply -f job.yaml
"""

from __future__ import annotations

import argparse
import json
import textwrap
from datetime import datetime


def _container_env() -> list[dict]:
    """Env vars for the worker container — mostly routing + model config."""
    return [
        {"name": "NATS_URL", "value": "nats://atlas-nats:4222"},
        {"name": "NATS_CONSUMER_GROUP", "value": "erebus-vision-workers"},
        {"name": "NATS_STREAM", "value": "EREBUS_TASKS"},
        {"name": "NATS_SUBJECTS", "value": "erebus.tasks.solve_task_vision"},
        {"name": "NATS_RESULT_PREFIX", "value": "erebus.results."},
        {"name": "NATS_DURABLE", "value": "0"},
        # 90s of queue-empty idle → clean exit so GPU util doesn't drift
        # below 40% for more than about a minute.
        {"name": "NATS_EXIT_ON_IDLE_S", "value": "90"},
        {
            "name": "RESULT_WEBHOOK_URL",
            "value": "https://atlas-sjsu.duckdns.org/api/erebus/result",
        },
        {"name": "TASK_DIR", "value": "/work/tasks"},
        {"name": "COMPILER_DIR", "value": "/work/bundle/src/compiler"},
        {"name": "PYTHONPATH", "value": "/work/agi-hpc/src"},
        {"name": "HF_HOME", "value": "/work/hf-cache"},
        {"name": "TRANSFORMERS_CACHE", "value": "/work/hf-cache"},
        {
            "name": "NRP_LLM_TOKEN",
            "valueFrom": {
                "secretKeyRef": {
                    "name": "erebus-worker-secrets",
                    "key": "nrp-llm-token",
                }
            },
        },
        {
            "name": "HF_TOKEN",
            "valueFrom": {
                "secretKeyRef": {"name": "erebus-worker-secrets", "key": "hf-token"}
            },
        },
    ]


def _bash_entry(bundle_repo: str, model_id: str) -> str:
    """Bootstrap script: install deps, clone repos, extract tasks, exec worker."""
    lines = [
        "set -e",
        "pip install --quiet nats-py openai numpy pillow "
        "'accelerate>=0.33' 'transformers>=4.49' "
        "'git+https://github.com/ahb-sjsu/nats-bursting.git@main#subdirectory=python'",
        "mkdir -p /work && cd /work",
        f"git clone --depth 1 https://github.com/{bundle_repo}.git bundle",
        "git clone --depth 1 https://github.com/ahb-sjsu/agi-hpc.git",
        "mkdir -p /work/tasks",
        "tar -C /work/tasks -xzf bundle/data/tasks.tar.gz",
        f"export VISION_MODEL_ID={model_id!r}",
        "exec python3 -u -m agi.autonomous.erebus_worker_main",
    ]
    return " && ".join(lines)


def render(
    name: str = "erebus-vision-batch",
    namespace: str = "ssu-atlas-ai",
    parallelism: int = 4,
    cpu: str = "4",
    memory: str = "24Gi",
    model_id: str = "THUDM/GLM-4.1V-9B-Thinking",
    bundle_repo: str = "ahb-sjsu/neurogolf-bundle",
    ttl_seconds: int = 600,
) -> str:
    # Unique job name per run so repeated kubectl apply creates fresh Jobs
    unique = f"{name}-{datetime.utcnow().strftime('%y%m%d%H%M')}"

    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": unique,
            "namespace": namespace,
            "labels": {
                "app": "erebus-vision-batch",
                "nats-bursting.role": "vision-burst",
            },
        },
        "spec": {
            # N pods run in parallel, each handles one task at a time; when the
            # queue is empty they idle-exit.  Completions = parallelism so the
            # Job completes once every pod has exited cleanly.
            "parallelism": parallelism,
            "completions": parallelism,
            "backoffLimit": 1,
            "ttlSecondsAfterFinished": ttl_seconds,
            "template": {
                "metadata": {
                    "labels": {
                        "app": "erebus-vision-batch",
                        "nats-bursting.role": "vision-burst",
                    }
                },
                "spec": {
                    "restartPolicy": "Never",
                    # Explicitly target non-A100 GPUs with enough VRAM for
                    # 9B-bf16 (~18GB). Ordered by preference.
                    "affinity": {
                        "nodeAffinity": {
                            "requiredDuringSchedulingIgnoredDuringExecution": {
                                "nodeSelectorTerms": [
                                    {
                                        "matchExpressions": [
                                            {
                                                "key": "nvidia.com/gpu.product",
                                                "operator": "In",
                                                "values": [
                                                    "NVIDIA-L40",
                                                    "NVIDIA-L40S",
                                                    "NVIDIA-A10",
                                                    "Tesla-V100-SXM2-32GB",
                                                ],
                                            }
                                        ]
                                    }
                                ]
                            },
                            "preferredDuringSchedulingIgnoredDuringExecution": [
                                {
                                    "weight": 100,
                                    "preference": {
                                        "matchExpressions": [
                                            {
                                                "key": "nvidia.com/gpu.product",
                                                "operator": "In",
                                                "values": ["NVIDIA-L40", "NVIDIA-L40S"],
                                            }
                                        ]
                                    },
                                }
                            ],
                        }
                    },
                    "containers": [
                        {
                            "name": "vision-worker",
                            "image": "gcr.io/kaggle-images/python:latest",
                            "resources": {
                                "requests": {
                                    "cpu": cpu,
                                    "memory": memory,
                                    "nvidia.com/gpu": "1",
                                },
                                "limits": {
                                    "cpu": cpu,
                                    "memory": memory,
                                    "nvidia.com/gpu": "1",
                                },
                            },
                            "env": _container_env(),
                            "command": ["/bin/bash", "-c"],
                            "args": [_bash_entry(bundle_repo, model_id)],
                        }
                    ],
                },
            },
        },
    }

    # Emit as YAML-shaped via json (k8s accepts JSON manifests too).
    return json.dumps(manifest, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="erebus-vision-batch")
    ap.add_argument("--namespace", default="ssu-atlas-ai")
    ap.add_argument(
        "--parallelism", type=int, default=4, help="Max 4 per NRP heavy-mode cap."
    )
    ap.add_argument("--cpu", default="4")
    ap.add_argument("--memory", default="24Gi")
    ap.add_argument("--model-id", default="THUDM/GLM-4.1V-9B-Thinking")
    ap.add_argument("--bundle-repo", default="ahb-sjsu/neurogolf-bundle")
    args = ap.parse_args()

    # Clamp to NRP heavy-mode limit
    parallelism = min(max(1, args.parallelism), 4)
    if parallelism != args.parallelism:
        print(
            f"# WARN: clamped parallelism {args.parallelism} → {parallelism} "
            "per NRP heavy-pod cap",
            file=__import__("sys").stderr,
        )

    print(
        render(
            name=args.name,
            namespace=args.namespace,
            parallelism=parallelism,
            cpu=args.cpu,
            memory=args.memory,
            model_id=args.model_id,
            bundle_repo=args.bundle_repo,
        )
    )


if __name__ == "__main__":
    main()
