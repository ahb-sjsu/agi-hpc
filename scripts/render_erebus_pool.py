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
            # Tasks come from the bundle (public repo); compiler modules
            # come from the bundle too. Handler source comes from agi-hpc.
            "TASK_DIR": "/work/tasks",
            "COMPILER_DIR": "/work/bundle/src/compiler",
            "PYTHONPATH": "/work/agi-hpc/src",
            "NATS_RESULT_PREFIX": "erebus.results.",
        },
        env_from_secrets={
            "NRP_LLM_TOKEN": ("erebus-worker-secrets", "nrp-llm-token"),
        },
        # Pod bootstrap: install deps, clone bundle (tasks + compiler
        # modules), clone agi-hpc (Erebus handler source), then exec
        # the pool worker with handlers registered.
        pre_install=[
            "pip install --quiet nats-py openai onnx onnxruntime numpy "
            "'git+https://github.com/ahb-sjsu/nats-bursting.git@main#subdirectory=python'",
            "mkdir -p /work && cd /work && "
            f"git clone --depth 1 https://github.com/{args.bundle_repo}.git bundle && "
            "git clone --depth 1 https://github.com/ahb-sjsu/agi-hpc.git && "
            "mkdir -p /work/tasks && "
            "tar -C /work/tasks -xzf bundle/data/tasks.tar.gz",
        ],
        entry=[
            "python3", "-u", "-m", "agi.autonomous.erebus_worker_main",
        ],
    )
    print(pool_manifest(desc))


if __name__ == "__main__":
    main()
