# Cover Email — Security Radar / Project Glasswing

## Subject: Research Proposal: Systematic Vulnerability Discovery via GPU Beam Search + Glasswing

---

To the Project Glasswing Team,

I'm Andrew Bond, a Senior IEEE Member (since 1989) and independent AI researcher at San Jose State University. I'd like to propose a research collaboration that combines my GPU beam search engine with Glasswing's LLM-guided vulnerability analysis.

**The idea in one sentence:** Apply the exhaustive search methodology from my symbolic AI research (Theory Radar — under review at IEEE TAI) to systematically enumerate attack patterns, with Claude Mythos as the evaluator, giving defenders something no existing tool provides: a coverage guarantee over a bounded attack surface.

**Why this is different from fuzzing or scanning:**
- Fuzzers are random — they can't tell you what they *haven't* tested
- Scanners are template-based — they miss novel combinations
- Security Radar exhaustively enumerates a formally bounded attack grammar (injection type × encoding × evasion × surface), prunes 88-99% of dead branches via meta-learned rules, and uses an LLM to evaluate the remainder
- After a sweep, a defender can say: "all depth-3 attack patterns in this grammar have been tested"

**What I bring:**
- Theory Radar: working GPU beam search engine with meta-learned pruning (88-99% elimination, zero false negatives within the enumerated space). 31-dataset benchmark, IEEE TAI submission.
- Atlas AI: multi-model cognitive architecture with 3-layer safety pipeline, running 7 LLM instances on local hardware. Live demo at atlas-sjsu.duckdns.org.
- TurboQuant Pro: open-source compression toolkit on PyPI (v0.7.0), demonstrating production-quality systems engineering.
- 37 years of systems engineering at AT&T.

**What I'm requesting:**
- Glasswing API access for the evaluation pipeline (Claude Mythos or equivalent)
- Guidance on responsible disclosure integration with the Glasswing partner network
- Potential co-authorship on the resulting publication (target: IEEE S&P or USENIX Security)

**What I'm NOT requesting:**
- Funding (I have existing hardware and open-source tooling)
- Unrestricted model access (sandboxed evaluation is sufficient)

I've attached a detailed research proposal covering the technical approach, evaluation plan, ethical considerations, and timeline. The proof of concept (Phase 1) can be completed in 3 months using standard vulnerable applications (OWASP WebGoat, DVWA).

I'd welcome the opportunity to discuss this further. I'm available for a call at your convenience.

Best regards,

Andrew H. Bond
Senior Member, IEEE (#01054659)
Department of Computer Engineering
San Jose State University
andrew.bond@sjsu.edu
ORCID: 0009-0003-2599-6158

GitHub: github.com/ahb-sjsu
Live demo: atlas-sjsu.duckdns.org
PyPI: pypi.org/project/turboquant-pro

---

## Where to Send

1. **Primary — Glasswing application portal:**
   - https://www.anthropic.com/glasswing (look for "Claude for Open Source" apply link)
   - https://red.anthropic.com/2026/mythos-preview/ (security research access)

2. **Linux Foundation / OpenSSF:**
   - They administer Glasswing credits for open-source security researchers
   - https://openssf.org — contact via their membership/research channels

3. **Direct outreach (LinkedIn):**
   - Jason Clinton — Anthropic CISO, leads Glasswing
   - Chris Olah — Anthropic co-founder, interpretability/security research
   - Dario Amodei — Anthropic CEO (for executive sponsorship)

4. **NVIDIA connection:**
   - You're already in the Nemotron competition — NVIDIA is a Glasswing launch partner
   - Reach out via your Kaggle/NVIDIA contacts about the security application

5. **Conference networking:**
   - IEEE S&P (May), USENIX Security (August) — Glasswing team likely presenting
   - Mention Theory Radar + Security Radar concept in person
