# From Scalars to Tensors: Operationalizing D4 Gauge Symmetry in AGI Safety

*When your safety system computes the wrong thing correctly, you have a more dangerous problem than computing nothing at all*

---

We just completed a significant upgrade to the ethical reasoning layer in **AGI-HPC**, our cognitive architecture for safe embodied AGI. The Bond Index -- our core metric for ethical consistency -- has been rebuilt from the ground up using the D4 dihedral group structure and multi-rank moral tensors.

This article explains what changed, why it matters, and what the mathematics of symmetry has to do with keeping robots from making morally incoherent decisions.

## The Bug We Fixed

Our previous Bond Index implementation had a fundamental flaw: **it was computing the wrong thing**.

The Bond Index is supposed to measure *correlative symmetry* -- whether an ethical judgment remains consistent when you swap perspectives between parties. If I have an obligation to you, then you have a claim against me. If I have liberty to act, you have no claim to stop me. These are Hohfeld's correlative pairs, and the Bond Index measures how faithfully a system preserves them.

Our old implementation? It computed a *veto rate*: the fraction of plan steps that got blocked. That's a useful number, but it has nothing to do with Hohfeldian symmetry. A plan where every step is vetoed would score 1.0, and a plan where no steps are vetoed would score 0.0 -- but neither number tells you whether the system's ethical reasoning is *internally consistent*.

This is the kind of bug that passes every test while undermining the theoretical foundation of the entire safety subsystem.

## What Hohfeldian Symmetry Actually Is

Wesley Hohfeld identified four fundamental normative positions in 1917:

```
    Obligation (O) -------- Claim (C)
         |                       |
         |                       |
    Liberty (L) ---------- No-claim (N)
```

These form two correlative pairs (perspective swaps):
- **O <-> C**: My obligation is your claim
- **L <-> N**: My liberty is your lack of claim

And two negation pairs (logical opposites):
- **O <-> L**: Obligation is the absence of liberty
- **C <-> N**: Claim is the absence of no-claim

These four states sit at the vertices of a square. The symmetries of a square form a mathematical group called **D4** -- the dihedral group of order 8. It has eight elements: four rotations and four reflections.

| Element | Action | Ethical Meaning |
|---------|--------|-----------------|
| e | Identity | No change |
| r | 90-degree rotation: O->C->L->N->O | Quarter-turn through all positions |
| r^2 | 180-degree rotation: O<->L, C<->N | Negation |
| r^3 | 270-degree rotation: O->N->L->C->O | Reverse quarter-turn |
| s | Reflection: O<->C, L<->N | **Correlative** (perspective swap) |
| sr | s composed with r | Combined transformation |
| sr^2 | s composed with r^2 | Combined transformation |
| sr^3 | s composed with r^3 | Combined transformation |

The correlative operation -- the one central to the Bond Index -- is the reflection **s**. The negation operation is the 180-degree rotation **r^2**. Together, they generate the Klein four-group V4 = {e, r^2, s, sr^2}, an abelian subgroup of D4.

But D4 itself is **non-abelian**: r * s != s * r. This matters. If empirical moral reasoning only exhibits correlative (s) and negation (r^2) operations, we're observing the abelian subgroup. Demonstrating that human ethical reasoning uses the full non-abelian D4 structure requires finding evidence of the quarter-turn elements {r, r^3, sr, sr^3} -- operations that don't commute with the correlative.

## The New Implementation

### D4 Group Algebra (`hohfeld.py`)

We now have a complete, formally verified D4 implementation:

- **Full 8x8 multiplication table** -- all 64 products computed from the group presentation
- **Group axioms verified in tests**: closure, identity, inverse, associativity
- **Group action on states**: `d4_apply_to_state(element, state)` computes how any of the 8 symmetry transformations maps any of the 4 Hohfeldian positions
- **Bond Index**: `compute_bond_index(verdicts_a, verdicts_b, tau)` measures correlative symmetry by counting defects -- positions where B != correlative(A) -- normalized by total comparisons and a temperature parameter
- **Wilson observable**: `compute_wilson_observable(path, initial, observed)` computes the holonomy around a closed path of D4 transformations, detecting gauge anomalies in ethical reasoning
- **Subgroup analysis**: Functions to determine whether observed transformations require the full non-abelian D4 structure or fall within the abelian Klein four subgroup

The Bond Index formula is now:

```
Bond Index = (defects / total) / tau

where defects = |{i : verdict_b[i] != correlative(verdict_a[i])}|
```

- **0** means perfect correlative symmetry
- **0.155** is the empirical baseline from the Dear Abby corpus
- **0.30** is the block threshold

### Multi-Rank Moral Tensors (`moral_tensor.py`)

The previous system represented ethical assessments as a flat 8+1 dimensional `MoralVector` -- a rank-0 (well, rank-1) object that collapses all contextual structure into nine floats.

The new `MoralTensor` class supports ranks 1 through 6:

| Rank | Shape | What It Represents |
|------|-------|--------------------|
| 1 | (9,) | Single assessment (backward-compatible with MoralVector) |
| 2 | (9, n) | Per-party assessment (n parties/stakeholders) |
| 3 | (9, n, tau) | Temporal evolution of per-party assessments |
| 4 | (9, n, a, c) | Coalition actions and consequences |
| 5 | (9, n, tau, a, s) | Temporal with actions and uncertainty samples |
| 6 | (9, n, tau, a, c, s) | Full multi-agent spatiotemporal context |

The 9 ethical dimensions forming the first axis are:

1. **Physical harm** (0 = safe, 1 = dangerous)
2. **Rights respect** (0 = violates, 1 = respects)
3. **Fairness & equity** (0 = unfair, 1 = fair)
4. **Autonomy respect** (0 = coercive, 1 = respects)
5. **Privacy protection** (0 = violates, 1 = protects)
6. **Societal/environmental** (0 = harmful, 1 = beneficial)
7. **Virtue & care** (0 = vicious, 1 = virtuous)
8. **Legitimacy & trust** (0 = illegitimate, 1 = legitimate)
9. **Epistemic quality** (0 = uncertain, 1 = certain)

These nine dimensions arise from a 3x3 matrix crossing three *domains of concern* (individual, relational, collective) with three *epistemic modes* (what matters, who decides, what we know).

### Why Higher-Rank Tensors Matter

Consider a plan with 5 steps affecting 3 stakeholders. The old MoralVector approach evaluated each step independently and produced 5 separate 9-dimensional vectors. To get a plan-level assessment, it averaged them -- losing all information about which stakeholders were affected by which steps and how the ethical profile evolved across the plan.

The new approach builds a rank-2 tensor of shape (9, 5) for the plan, or a rank-3 tensor of shape (9, 3, 5) incorporating stakeholder perspectives. From this tensor, we can:

- **Slice by party**: Extract one stakeholder's ethical profile across all steps
- **Slice by dimension**: Track how physical harm risk evolves across the plan
- **Reduce**: Compute mean, max, or min across any axis
- **Contract**: Apply stakeholder importance weights to get a weighted assessment
- **Promote rank**: Expand a single assessment to a multi-party tensor via broadcasting
- **Compute Pareto dominance**: Check if one plan dominates another across all dimensions
- **Track veto locations**: Know exactly which stakeholder at which step triggered a safety veto

All arithmetic operations (addition, subtraction, multiplication, division) are clamped to [0, 1] to maintain valid ethical assessment values. Sparse COO storage is available for memory-efficient high-rank tensors.

### Service Integration

The `ErisMLServicer` now:

1. **Derives Hohfeldian verdicts from moral vectors**: Each moral dimension implies normative positions. High physical harm (>0.5) implies the agent has an **Obligation** to avoid harm, and the affected party has a **Claim** to safety. High autonomy respect (>0.7) implies the agent has **Liberty** and the affected party has **No-claim**.

2. **Computes proper Bond Index**: Instead of counting vetoes, we now count *correlative symmetry defects* -- cases where the affected party's derived Hohfeldian position does not match the correlative of the agent's position.

3. **Builds plan-level tensors**: Plan evaluation produces a rank-2 MoralTensor alongside the per-step results, enabling downstream systems to perform their own tensor analysis.

## The Bond Invariance Principle

This implementation is grounded in a broader theoretical framework: the **Bond Invariance Principle** (BIP).

The BIP states: *An ethical judgment is valid only if it is invariant under all transformations that preserve the bonds* -- where "bonds" are morally relevant relationships (obligations, claims, dependencies, vulnerabilities, consent).

Formally: for any bond-preserving transformation g in the symmetry group G, and any ethical judgment function J:

```
J(T) = J(g * T)
```

If the bonds haven't changed, the judgment shouldn't change. The D4 group structure provides the mathematical machinery to test this: the correlative transformation s swaps agent and patient perspectives, and the Bond Index measures whether judgments survive this swap.

The BIP has been validated empirically through cross-lingual transfer learning across English, Classical Chinese, Hebrew, Arabic, and Sanskrit ethical texts, achieving 80% ethical classification with near-zero language leakage and an 11x structural-to-surface sensitivity ratio.

## What This Enables

### Tensor-Based Safety Monitoring

Instead of a binary safe/unsafe decision, the system now produces a rich tensor that downstream components can analyze:

```python
from agi.safety.erisml import MoralTensor, compute_bond_index

# Plan evaluation produces a (9, n_steps) tensor
plan_tensor = MoralTensor.from_dense(plan_data)

# Track physical harm across steps
harm_profile = plan_tensor.slice_dimension("physical_harm")

# Weight by stakeholder importance
weighted = plan_tensor.contract("n", weights=stakeholder_weights)

# Check Pareto dominance between alternative plans
if plan_a_tensor.dominates(plan_b_tensor):
    prefer_plan_a()
```

### Auditable Symmetry Verification

Every plan evaluation now reports not just whether the plan was approved, but the specific correlative symmetry violations found:

```
Step 3: agent(O) <-> affected(L): expected C
Step 7: agent(L) <-> affected(O): expected N
Bond Index: 0.18 (within threshold, baseline 0.155)
```

This gives governance teams specific, mathematically grounded evidence for why ethical assessments may be inconsistent.

### Klein Four Subgroup Detection

The system can now detect whether its ethical reasoning is using the full D4 structure or only the abelian Klein four subgroup. If all observed transformations commute, we may be missing non-obvious ethical asymmetries that only the quarter-turn elements reveal.

## By the Numbers

| Metric | Value |
|--------|-------|
| New source files | 2 (hohfeld.py, moral_tensor.py) |
| Modified source files | 3 (service.py, erisml.proto, __init__.py) |
| New tests | 110 (43 D4 group, 67 tensor) |
| Total tests passing | 896 |
| Regressions | 0 |
| D4 multiplication table entries | 64 |
| Tensor ranks supported | 1-6 |
| Ethical dimensions | 9 |
| Bond Index: perfect symmetry | 0.000 |
| Bond Index: empirical baseline | 0.155 |
| Bond Index: block threshold | 0.300 |

## The Bigger Picture

This update moves us from a system that *talks about* Hohfeldian symmetry to one that *computes with* it. The D4 group is not a metaphor -- it is the actual mathematical structure governing how normative positions transform under perspective changes. The moral tensor is not an analogy to physics -- it is the natural representation for ethical assessments that have directional structure, multi-party scope, and temporal evolution.

```
sqnd-probe          ->  Empirical D4 structure in human moral reasoning
       |
erisml-lib          ->  Tensor algebra + D4 gauge theory for ethics
       |
agi-hpc             ->  Operationalized in cognitive architecture (this update)
```

The empirical research validates the mathematical framework. The mathematical framework gets operationalized in the cognitive architecture. The cognitive architecture generates decisions whose ethical consistency can be measured and audited. Each layer informs the others.

## What's Next

- **Rank-4+ tensor evaluation**: Coalition action assessment for multi-robot coordination
- **Wilson loop monitoring**: Detecting gauge anomalies in real-time ethical reasoning
- **Temporal Bond Index tracking**: How ethical consistency evolves over extended operation
- **Non-abelian structure detection**: Empirical evidence for quarter-turn transformations in AGI decision-making
- **Sparse tensor optimization**: Memory-efficient representation for high-rank assessment in HPC deployments

---

*The question was never "is this action good?" -- a scalar question with a scalar answer. The question is "does this judgment preserve the bonds?" -- a geometric question requiring geometric tools. With the D4 group and multi-rank tensors, we now have the right tools for the right question.*

---

**Andrew H. Bond, Senior Member IEEE**
Department of Computer Engineering
San Jose State University

#AISafety #AGI #EthicalAI #TensorialEthics #GroupTheory #Robotics #HPC #GaugeSymmetry #MachineLearning #SJSU

---

## Technical Details

**Repository**: github.com/ahb-sjsu/agi-hpc

**New Components**:
- `src/agi/safety/erisml/hohfeld.py` -- D4 dihedral group structure (~330 lines)
- `src/agi/safety/erisml/moral_tensor.py` -- Multi-rank tensor framework (~750 lines)
- `tests/unit/test_hohfeld.py` -- 43 tests covering group axioms, Bond Index, Wilson observable
- `tests/unit/test_moral_tensor.py` -- 67 tests covering ranks 1-6, arithmetic, serialization

**Modified Components**:
- `src/agi/safety/erisml/service.py` -- Proper D4-based Bond Index, Hohfeldian verdict derivation
- `proto/erisml.proto` -- MoralTensorProto message, enhanced BondIndexResultProto
- `src/agi/safety/erisml/__init__.py` -- Complete public API exports

**Test Coverage**: 896 tests passing (110 new, 0 regressions)

**Key Dependencies**: Python 3.10+, numpy (optional for tensor operations), gRPC, Protocol Buffers

**References**:
- Hohfeld, W.N. (1917). "Fundamental Legal Conceptions as Applied in Judicial Reasoning." *Yale Law Journal*, 26(8), 710-770.
- Bond, A.H. (2025). "The Bond Invariance Principle." In *Tensorial Ethics: A Geometric Framework for Moral Philosophy*.
- Bond, A.H. & Claude (2026). "SQND-Probe: A Gamified Instrument for Measuring Dihedral Gauge Structure in Human Moral Reasoning."
