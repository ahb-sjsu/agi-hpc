"""The Primer — always-on Claude-style tutor for Erebus.

Inspired by Stephenson's Young Lady's Illustrated Primer: an adaptive
teaching layer that meets the learner at their current confusion and
walks them through to understanding.

Core components:
- ``vmoe``: virtual Mixture-of-Experts router/ensemble over multiple
  NRP-hosted frontier models. Picks the right expert per query, or
  fires N in parallel and returns the first verified answer.
- ``service``: the Primer daemon — watches Erebus's help queue, reads
  wiki + episodic memory for context, runs the vMOE, verifies code
  against task.train, writes verified sensei notes to the wiki.
"""

from .vmoe import Expert, Response, vMOE, default_experts

__all__ = ["Expert", "Response", "vMOE", "default_experts"]
