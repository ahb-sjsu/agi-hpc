"""Render the Kubernetes Deployment manifest for the Erebus worker pool.

Usage:
    python scripts/render_erebus_pool.py [--replicas N] > erebus-pool.yaml
    kubectl apply -f erebus-pool.yaml
"""

from __future__ import annotations

import argparse

from nats_bursting import PoolDescriptor, pool_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="erebus-workers")
    ap.add_argument("--namespace", default="ssu-atlas-ai")
    ap.add_argument("--replicas", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=0,
                    help="Request N GPUs per pod (0 = CPU only, stay in swarm mode)")
    ap.add_argument("--bundle-repo", default="ahb-sjsu/neurogolf-bundle")
    args = ap.parse_args()

    desc = PoolDescriptor(
        name=args.name,
        namespace=args.namespace,
        replicas=args.replicas,
        cpu="1" if args.gpu == 0 else "4",
        memory="2Gi" if args.gpu == 0 else "16Gi",
        gpu=args.gpu,
        consumer_group="erebus-workers",
        stream="EREBUS_TASKS",
        subjects=["erebus.tasks.>"],
        env={
            "TASK_DIR": "/work/tasks",
            "COMPILER_DIR": "/work/src/compiler",
            "BUNDLE_REPO": args.bundle_repo,
            "PYTHONPATH": "/work/src",
            "NATS_RESULT_PREFIX": "erebus.results.",
        },
        env_from_secrets={
            "NRP_LLM_TOKEN": ("erebus-worker-secrets", "nrp-llm-token"),
        },
        # Install deps + clone bundle each pod start, then exec the
        # Erebus handler-registered worker loop.
        pre_install=[
            "pip install --quiet nats-py openai onnx onnxruntime numpy "
            "'git+https://github.com/ahb-sjsu/nats-bursting.git@main#subdirectory=python'",
            "mkdir -p /work && cd /work && "
            f"git clone --depth 1 https://github.com/{args.bundle_repo}.git bundle && "
            "mkdir -p /work/tasks && "
            "tar -C /work/tasks -xzf bundle/data/tasks.tar.gz && "
            "cp -r bundle/src /work/src && "
            # Also clone agi-hpc src so handlers import
            "git clone --depth 1 https://github.com/ahb-sjsu/neurogolf-bundle.git /tmp/ignore 2>/dev/null || true",
        ],
        entry=[
            "python3", "-u", "-m", "agi.autonomous.erebus_worker_main",
        ],
    )
    print(pool_manifest(desc))


if __name__ == "__main__":
    main()
