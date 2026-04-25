#!/usr/bin/env python3
# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""SDXL image renderer for the Halyard wiki.

Generates atmospheric concept art for the Beyond the Heliopause
campaign wiki entries. Designed to run on Atlas (Quadro GV100,
32GB VRAM) using the local diffusers stack; outputs PNG into
``wiki/halyard/art/`` keyed by entry id.

Usage::

    # Single prompt to a single output
    python art_render.py --prompt "..." --out art/scene.png

    # Batch from a YAML/JSON manifest
    python art_render.py --manifest scripts/halyard/art_manifest.yaml

The manifest format is a list of jobs::

    - id:     earth/cities/lagos
      prompt: "Marina Stack arcology cluster at dusk, ..."
      out:    wiki/halyard/art/earth-cities-lagos.png
      seed:   42                  # optional; for reproducibility
      width:  1024                # optional; default 1024
      height: 1024                # optional; default 1024
      steps:  30                  # optional; default 30
      negative: "..."             # optional; default below

Style anchor: a consistent prompt suffix is appended to every job
unless the manifest entry sets ``style: false``. The anchor
favors painterly, Stalenhag-adjacent, muted-palette atmospheric
work — matches the campaign's tone better than photographic
rendering.

Caches the model load between jobs in batch mode; one pipeline
instance services the whole manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Defer heavy imports until we know we're rendering.
# The CLI argument parsing should be cheap.

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_FALLBACK = "segmind/SSD-1B"  # ungated, smaller, ~2GB

DEFAULT_NEGATIVE = (
    "low quality, blurry, watermark, signature, text, logo, "
    "deformed, low resolution, jpeg artifacts, oversaturated, "
    "cartoon, anime"
)

STYLE_ANCHOR = (
    " digital painting, atmospheric, cinematic, muted color palette, "
    "Simon Stalenhag adjacent, Beeple adjacent, hard science fiction, "
    "lived-in, photorealistic lighting, detailed environment"
)


@dataclass
class Job:
    id: str
    prompt: str
    out: str
    seed: int | None = None
    width: int = 1024
    height: int = 1024
    steps: int = 30
    negative: str = DEFAULT_NEGATIVE
    style: bool = True


def _load_pipeline(model_id: str):
    """Load an SDXL pipeline. Returns None if loading fails."""
    import torch
    from diffusers import StableDiffusionXLPipeline

    print(f"loading {model_id} ...", flush=True)
    t0 = time.time()
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
    except Exception as e:
        print(f"  fp16 variant load failed: {e}", file=sys.stderr)
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
        except Exception as e2:
            print(f"  default load also failed: {e2}", file=sys.stderr)
            return None
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
    return pipe


def _render(pipe, job: Job) -> bool:
    """Render a single job. Returns True on success."""
    import torch

    prompt = job.prompt + (STYLE_ANCHOR if job.style else "")
    out_path = Path(job.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  [skip] {out_path}", flush=True)
        return True

    generator = None
    if job.seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(job.seed)

    t0 = time.time()
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=job.negative,
            width=job.width,
            height=job.height,
            num_inference_steps=job.steps,
            generator=generator,
        )
        result.images[0].save(out_path)
        print(
            f"  [ok]   {out_path}  ({time.time() - t0:.1f}s)",
            flush=True,
        )
        return True
    except Exception as e:
        print(f"  [fail] {out_path}: {e}", file=sys.stderr)
        return False


def _load_manifest(path: Path) -> list[Job]:
    """Load a manifest in JSON or YAML."""
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError:
            print("PyYAML not installed; install or use JSON.", file=sys.stderr)
            sys.exit(2)
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, list):
        print("Manifest must be a list of jobs.", file=sys.stderr)
        sys.exit(2)
    jobs = []
    for raw in data:
        jobs.append(
            Job(
                id=raw["id"],
                prompt=raw["prompt"],
                out=raw["out"],
                seed=raw.get("seed"),
                width=raw.get("width", 1024),
                height=raw.get("height", 1024),
                steps=raw.get("steps", 30),
                negative=raw.get("negative", DEFAULT_NEGATIVE),
                style=raw.get("style", True),
            )
        )
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "HuggingFace model id. Default: stabilityai SDXL base. "
            "If gated and download fails, falls back to "
            f"{DEFAULT_FALLBACK}."
        ),
    )
    parser.add_argument("--prompt", help="Single-render prompt.")
    parser.add_argument("--out", help="Single-render output path.")
    parser.add_argument(
        "--manifest", help="JSON or YAML manifest for batch render."
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument(
        "--no-style",
        action="store_true",
        help="Disable the style anchor suffix.",
    )
    args = parser.parse_args()

    if not args.prompt and not args.manifest:
        parser.error("Need --prompt + --out, or --manifest.")
    if args.prompt and not args.out:
        parser.error("--prompt requires --out.")

    pipe = _load_pipeline(args.model)
    if pipe is None and args.model != DEFAULT_FALLBACK:
        print(
            f"falling back to {DEFAULT_FALLBACK} (ungated)",
            file=sys.stderr,
        )
        pipe = _load_pipeline(DEFAULT_FALLBACK)
    if pipe is None:
        print("no pipeline available; bailing", file=sys.stderr)
        return 1

    if args.prompt:
        jobs = [
            Job(
                id="ad-hoc",
                prompt=args.prompt,
                out=args.out,
                seed=args.seed,
                width=args.width,
                height=args.height,
                steps=args.steps,
                style=not args.no_style,
            )
        ]
    else:
        jobs = _load_manifest(Path(args.manifest))

    print(f"rendering {len(jobs)} jobs ...", flush=True)
    ok = 0
    for j in jobs:
        if _render(pipe, j):
            ok += 1
    print(f"done — {ok}/{len(jobs)} succeeded", flush=True)
    return 0 if ok == len(jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
