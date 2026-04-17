"""Submit a Nemotron training job to NRP via nats-bursting.

Usage:
  python3 nemotron/submit_nrp.py --image ttl.sh/nemotron-nrp-XXXX:24h --lr 1.5e-4

The job:
  - Pulls the pre-built nemotron-nrp image (torch + mamba_ssm baked in)
  - Downloads competition data from HuggingFace or uses baked-in /data
  - Runs nemotron_nrp.py with the specified hyperparameters
  - Saves adapter weights to /output/submission.zip
  - Stdout captures training logs for monitoring
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid

try:
    from nats_bursting import Client, JobDescriptor, Resources
except ImportError:
    sys.stderr.write("nats_bursting not installed\n")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Docker image URI")
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--memory", default="80Gi")
    ap.add_argument("--cpu", default="8")
    ap.add_argument("--nats-url", default="nats://192.168.0.7:4222")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    job_name = f"nemotron-train-{uuid.uuid4().hex[:6]}"

    # Competition data: download from Kaggle API inside the pod
    # (requires KAGGLE_USERNAME + KAGGLE_KEY env vars)
    # OR: bake data into image / mount from Ceph
    desc = JobDescriptor(
        name=job_name,
        image=args.image,
        command=[
            "python3", "/app/nemotron_nrp.py",
            "--data-dir", "/data",
            "--output-dir", "/output",
            "--lr", str(args.lr),
            "--epochs", str(args.epochs),
            "--lora-r", str(args.lora_r),
            "--batch-size", str(args.batch_size),
            "--grad-accum", str(args.grad_accum),
        ],
        env={},
        resources=Resources(
            cpu=args.cpu,
            memory=args.memory,
            gpu=args.gpu,
        ),
        labels={
            "app.kubernetes.io/managed-by": "nats-bursting",
            "nemotron.io/type": "training",
        },
        backoff_limit=0,
    )

    if args.dry_run:
        print(json.dumps(desc.to_dict(), indent=2))
        return

    client = Client(nats_url=args.nats_url)
    result = client.submit(desc)
    print(f"Submitted: {job_name}")
    print(f"  accepted: {result.accepted}")
    print(f"  k8s_job: {result.k8s_job_name}")


if __name__ == "__main__":
    main()
