# Security Radar: Systematic Vulnerability Discovery via GPU Beam Search and LLM-Guided Evaluation

## Research Proposal

**Principal Investigator:** Andrew H. Bond, Senior Member, IEEE
**Affiliation:** Department of Computer Engineering, San Jose State University
**Contact:** andrew.bond@sjsu.edu | ORCID: 0009-0003-2599-6158

---

## 1. Problem Statement

Current approaches to automated vulnerability discovery fall into two camps: **random fuzzing** (AFL, LibFuzzer) which achieves high throughput but poor coverage of structured attack spaces, and **LLM-guided analysis** (Project Glasswing, Claude Mythos) which achieves deep reasoning but lacks systematic coverage guarantees. Neither approach tells a defender: "we have exhaustively searched this bounded attack surface and can characterize exactly where the vulnerabilities are and aren't."

We propose **Security Radar** — a system that combines exhaustive GPU beam search over formally bounded attack grammars with LLM-guided evaluation of candidate exploits. The key insight comes from our work on Theory Radar (Bond, IEEE TAI 2026), where we showed that GPU beam search over a depth-bounded formula space, combined with meta-learned pruning rules, can exhaustively cover a combinatorial search space while eliminating 88-99% of dead branches with zero false negatives. We apply the same architecture to vulnerability discovery.

## 2. Technical Approach

### 2.1 Attack Grammar

We define a formal grammar G over parameterized attack patterns:

```
Attack := InjectionType × Encoding × Evasion × Surface × Payload

InjectionType := {SQLi, XSS, CommandInj, PathTraversal, SSRF, 
                  Deserialization, TemplateInj, LDAPInj, ...}

Encoding := {None, URL, Base64, Unicode, DoubleEncode, 
             HTMLEntity, Hex, Octal, ...}

Evasion := {None, CaseMutation, WhitespaceVariant, CommentInsertion,
            NullByte, Truncation, CharsetMix, ...}

Surface := {QueryParam, Header, Cookie, Body, Path, 
            Filename, Metadata, ...}

Payload := Parameterized templates per InjectionType
```

At depth 3 (injection × encoding × evasion), with 8 injection types, 8 encodings, and 8 evasion techniques, the search space is 512 attack patterns per surface. At depth 4 (adding payload variants), it grows to ~10,000 — well within GPU beam search capacity.

### 2.2 GPU Beam Search

Adapted from Theory Radar's beam search engine:

1. **Enumeration**: Generate all attack candidates from grammar G up to depth d
2. **Parallel evaluation**: Launch N candidates against a sandboxed target simultaneously
3. **Scoring**: Each candidate receives a score based on:
   - HTTP response code deviation from baseline
   - Response body anomalies (error messages, stack traces, data leakage)
   - Timing anomalies (blind injection detection)
   - WAF bypass success
4. **Beam pruning**: Keep top-k candidates by score, expand to depth d+1
5. **Meta-learned pruning**: Eliminate attack subtrees that are provably dead (see 2.3)

**Hardware**: GPU beam search runs on consumer hardware (demonstrated on 2x Quadro GV100 32GB). The search is embarrassingly parallel — each candidate evaluation is independent.

### 2.3 Meta-Learned Pruning (from Theory Radar)

The key innovation from Theory Radar that transfers directly:

1. **Exhaustive micro-search**: On a small, well-characterized target (e.g., DVWA, WebGoat), exhaustively enumerate ALL depth-3 attacks and record which succeed
2. **Rule discovery**: From the exhaustive results, learn pruning rules of the form "if the target uses Framework X and InjectionType Y fails at depth 1, then ALL depth-2+ extensions of Y also fail"
3. **Transfer**: Apply these pruning rules to larger targets, eliminating 88-99% of the search space without false negatives (within the enumerated grammar)
4. **Zero false negative guarantee**: Within the bounded grammar, pruning rules are validated exhaustively. Extrapolation to deeper search is empirical (same caveat as Theory Radar)

### 2.4 LLM-Guided Evaluation (Glasswing Integration)

Where Theory Radar uses F1 score as the evaluation metric, Security Radar uses an LLM (Claude Mythos or equivalent) as a more sophisticated evaluator:

1. **Candidate triage**: For each candidate that scores above threshold in automated evaluation, the LLM assesses:
   - Is this a true positive or a false alarm?
   - What is the severity (CVSS estimate)?
   - Can this be chained with other findings?
   - What is the root cause?

2. **Attack chain synthesis**: The LLM composes individual findings into multi-step attack chains — the same capability that makes Glasswing powerful for zero-day discovery

3. **Exploit generation**: For confirmed vulnerabilities, the LLM generates proof-of-concept code for responsible disclosure

4. **Coverage analysis**: The LLM interprets the pruning rules to generate natural-language explanations of which attack surfaces have been exhaustively cleared

### 2.5 Architecture

```
                    Target Application (sandboxed)
                              |
                    [Attack Grammar G]
                              |
                    [GPU Beam Search]           ← Theory Radar engine
                     /        |        \
              [depth-1]  [depth-2]  [depth-3]   ← parallel evaluation
                     \        |        /
                    [Meta-Learned Pruning]       ← 88-99% elimination
                              |
                    [Candidate Exploits]
                              |
                    [LLM Evaluation]             ← Claude Mythos / Glasswing
                     /        |        \
              [Triage]  [Chaining]  [PoC Gen]
                     \        |        /
                    [Security Report]
                              |
              [Responsible Disclosure Pipeline]
                              |
              [Glasswing Partner Network]
                    (CrowdStrike, Palo Alto, ...)
```

## 3. Differentiation from Existing Work

| Approach | Coverage | Intelligence | Systematic? | Pruning? |
|----------|----------|-------------|------------|----------|
| AFL/LibFuzzer | Random mutation | None | No | No |
| Burp Suite | Template-based | Rule-based | Partial | Manual |
| Project Glasswing | LLM-guided | Deep reasoning | No | No |
| **Security Radar** | **Grammar-bounded** | **LLM-evaluated** | **Yes** | **Meta-learned** |

**Key differentiator**: Security Radar is the only approach that provides a *coverage guarantee* within a bounded attack grammar. After a Security Radar sweep, a defender can say: "all 512 depth-3 attack patterns in grammar G have been tested, 488 were pruned as provably ineffective, 24 were evaluated, and 3 are confirmed vulnerabilities." No existing tool makes this claim.

## 4. Evaluation Plan

### Phase 1: Proof of Concept (3 months)
- Implement attack grammar G with 5 injection types, 4 encodings, 4 evasion techniques
- Adapt Theory Radar's GPU beam search for HTTP request generation
- Evaluate against OWASP WebGoat, DVWA, and Juice Shop (standard vulnerable applications)
- Measure: detection rate vs Burp Suite, ZAP, and manual testing
- Measure: false positive rate, coverage percentage, time to complete

### Phase 2: Meta-Learning (3 months)
- Exhaustive micro-search on 10 vulnerable applications (known ground truth)
- Learn pruning rules; measure prune rate and false negative rate
- Transfer pruning rules to 10 unseen applications
- Measure: generalization accuracy, prune rate on transfer targets

### Phase 3: LLM Integration (3 months)
- Integrate Claude (or Glasswing-authorized model) as evaluator
- Implement attack chain synthesis
- Evaluate on HackerOne disclosed vulnerabilities (retrospective: can Security Radar rediscover known bugs?)
- Measure: precision, recall, CVSS accuracy, chain complexity

### Phase 4: Real-World Pilot (3 months)
- Partner with Glasswing network member for authorized testing
- Run Security Radar against a real production application (with authorization)
- Responsible disclosure of any findings
- Publish results with partner approval

## 5. Ethical Considerations

### 5.1 Responsible Use
- All testing conducted only against authorized targets (sandboxed or with written permission)
- Findings disclosed through Glasswing's responsible disclosure pipeline
- No public release of exploit code without vendor patch availability
- Research subject to SJSU IRB review (exempt category — no human subjects)

### 5.2 Dual-Use Risk
- The attack grammar and beam search engine could be misused for offensive purposes
- Mitigation: the grammar definitions and pruning rules are published; the evaluation tooling requires authorized LLM access (Glasswing-gated)
- The meta-learned pruning rules are *defensive* artifacts — they tell defenders which attack surfaces are clear, which is strictly beneficial

### 5.3 Alignment with Glasswing Mission
- Glasswing's stated goal: "securing the world's most critical software for the AI era"
- Security Radar provides *systematic coverage* that complements Glasswing's *deep reasoning*
- The combination gives defenders both "we found these specific bugs" AND "we've cleared these attack surfaces"

## 6. Team and Resources

### Personnel
- **Andrew H. Bond** (PI): GPU beam search (Theory Radar), distributed AI systems (Atlas AI), embedding compression (TurboQuant Pro). IEEE Senior Member since 1989. 37 years of systems engineering experience at AT&T.

### Computing Resources
- **Atlas Workstation**: 2x Quadro GV100 32GB, 252GB RAM, 48 CPU threads. Demonstrated capability for GPU beam search (Theory Radar), LLM inference (Atlas AI with 7 concurrent models), and large-scale vector operations (TurboQuant Pro on 3.3M vectors).
- **SJSU HPC**: 2x P100 16GB for larger-scale experiments.
- **Kaggle/Cloud**: For Blackwell GPU access if needed for larger models.

### Software
- **Theory Radar**: Existing GPU beam search engine (Python/PyTorch, ~8,000 LOC). Directly adaptable for attack pattern enumeration.
- **Atlas AI**: Existing multi-model cognitive architecture with safety pipeline. Can serve as the evaluation framework (safety gateway evaluates attack patterns for real-world risk).
- **TurboQuant Pro**: Existing embedding compression (relevant for embedding-based similarity search over known vulnerability databases).

## 7. Timeline and Budget

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| 1: Proof of Concept | Months 1-3 | Working prototype on OWASP targets |
| 2: Meta-Learning | Months 4-6 | Pruning rules with transfer evaluation |
| 3: LLM Integration | Months 7-9 | Glasswing-integrated evaluation pipeline |
| 4: Real-World Pilot | Months 10-12 | Authorized production testing + paper |

**Budget**: $0 direct costs (existing hardware + open-source tooling). Glasswing API credits requested via partnership application. Travel for conference presentation (estimated $3,000 if accepted to IEEE S&P, USENIX Security, or CCS).

## 8. Expected Outcomes

1. **Open-source Security Radar tool** with attack grammar definition language, GPU beam search engine, and meta-learned pruning framework
2. **Empirical benchmark** comparing Security Radar coverage and detection rate against Burp Suite, ZAP, AFL, and manual testing on standard vulnerable applications
3. **Meta-pruning rule library** — transferable pruning rules for common web application frameworks (Django, Rails, Express, Spring)
4. **IEEE publication** (target: IEEE S&P, USENIX Security, or ACM CCS) documenting the approach, coverage guarantees, and real-world evaluation
5. **Glasswing partnership** — integration of systematic search with LLM-guided evaluation

## 9. References

1. Bond, A.H. "Theory Radar: Charting the Boundary Between Interpretable Formulas and Black-Box Ensembles." IEEE Trans. Artificial Intelligence, 2026. (Under review)
2. Anthropic. "Project Glasswing: Securing Critical Software for the AI Era." April 2026.
3. Zandieh, A. et al. "Sub-linear Memory Inference via PolarQuant and QJL." ICLR, 2026.
4. Böhme, M. et al. "Coverage-based greybox fuzzing as Markov chain." IEEE Trans. Software Engineering, 2019.
5. Devvrit et al. "MatFormer: Nested Transformer for Elastic Inference." arXiv:2310.07707, 2023.

---

*Prepared April 2026. Contact: andrew.bond@sjsu.edu*
