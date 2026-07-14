# Rebuild SecureAI_Bond_2026_slides.tex as a native, editable PPTX.
# Brand palette from erisml-lib/docs/brand (atlasblue/erisgold/ink).
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
import copy

ASSETS = r"C:\source\agi-hpc\benchmarks\ieee_bds_2026\revision\assets"
OUT = r"C:\source\agi-hpc\benchmarks\ieee_bds_2026\SecureAI_Bond_2026_slides.pptx"

ATLASBLUE = RGBColor(0x2B, 0x6C, 0xB0)
ERISGOLD = RGBColor(0xA6, 0x75, 0x00)  # erisgold!85!black
INK = RGBColor(0x1A, 0x23, 0x40)
BLOCKBG = RGBColor(0xE9, 0xF0, 0xF9)   # atlasblue!12
GRAY = RGBColor(0x66, 0x66, 0x66)
BLACK = RGBColor(0x20, 0x20, 0x20)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Calibri"
SW, SH = Inches(13.333), Inches(7.5)

prs = Presentation()
prs.slide_width = SW
prs.slide_height = SH
BLANK = prs.slide_layouts[6]

# ---- helpers ---------------------------------------------------------------
# a "segment" is (text, style); style: None, 'hi' (blue bold), 'gold' (gold bold),
# 'em' (italic), 'bold', 'mono' (Consolas)


def add_runs(p, segments, size, base_color=BLACK):
    for text, style in segments:
        r = p.add_run()
        r.text = text
        f = r.font
        f.name = FONT
        f.size = Pt(size)
        f.color.rgb = base_color
        if style == "hi":
            f.bold = True
            f.color.rgb = ATLASBLUE
        elif style == "gold":
            f.bold = True
            f.color.rgb = ERISGOLD
        elif style == "em":
            f.italic = True
        elif style == "bold":
            f.bold = True
        elif style == "mono":
            f.name = "Consolas"


def textbox(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tb.text_frame.word_wrap = True
    return tb.text_frame


def para(tf, segments, size=16, bullet=False, indent=0, space_after=6,
         align=PP_ALIGN.LEFT, color=BLACK, first=False):
    p = tf.paragraphs[0] if first and not tf.paragraphs[0].runs else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    if bullet:
        segments = [("•  ", None)] + segments
        p.level = indent
    add_runs(p, segments, size, base_color=color)
    return p


def title_bar(slide, text, num=None):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, Inches(0.72))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ATLASBLUE
    bar.line.fill.background()
    tf = bar.text_frame
    tf.margin_left = Inches(0.45)
    tf.margin_top = Inches(0.08)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.name = FONT
    r.font.size = Pt(22)
    r.font.bold = True
    r.font.color.rgb = WHITE
    if num:
        ftf = textbox(slide, SW - Inches(0.9), SH - Inches(0.42), Inches(0.7), Inches(0.3))
        para(ftf, [(str(num), None)], size=10, align=PP_ALIGN.RIGHT, color=GRAY, first=True)


def block(slide, x, y, w, h, title, body_lines, body_size=15):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shp.adjustments[0] = 0.06
    shp.fill.solid()
    shp.fill.fore_color.rgb = BLOCKBG
    shp.line.color.rgb = ATLASBLUE
    shp.line.width = Pt(0.75)
    tf = shp.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.18)
    tf.margin_right = Inches(0.18)
    tf.margin_top = Inches(0.1)
    p = tf.paragraphs[0]
    add_runs(p, [(title, "hi")], body_size + 1)
    p.space_after = Pt(6)
    for seg in body_lines:
        para(tf, seg["segments"], size=body_size, bullet=seg.get("bullet", False),
             space_after=seg.get("space_after", 4))
    return shp


def table(slide, x, y, w, rows_data, col_widths, header=True, size=14,
          row_h=0.32):
    nrows, ncols = len(rows_data), len(rows_data[0])
    shp = slide.shapes.add_table(nrows, ncols, x, y, w, Inches(row_h * nrows))
    tbl = shp.table
    tbl.first_row = header
    tbl.horz_banding = False
    for i, cw in enumerate(col_widths):
        tbl.columns[i].width = Inches(cw)
    for ri, row in enumerate(rows_data):
        for ci, cell_segs in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.fill.solid()
            cell.fill.fore_color.rgb = ATLASBLUE if (header and ri == 0) else WHITE
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            base = WHITE if (header and ri == 0) else BLACK
            segs = cell_segs if isinstance(cell_segs, list) else [(cell_segs, "bold" if header and ri == 0 else None)]
            add_runs(p, segs, size, base_color=base)
    return shp


def footnote(slide, segments, y=None, size=11):
    y = y if y is not None else SH - Inches(0.62)
    tf = textbox(slide, Inches(0.55), y, SW - Inches(1.1), Inches(0.5))
    para(tf, segments, size=size, color=GRAY, first=True)


def new_slide(title=None, num=None):
    s = prs.slides.add_slide(BLANK)
    if title:
        title_bar(s, title, num)
    return s


# ---- 1. title slide --------------------------------------------------------
s = prs.slides.add_slide(BLANK)
s.shapes.add_picture(ASSETS + r"\atlas_mark_light.png", Inches(5.92), Inches(0.55), height=Inches(1.5))
tf = textbox(s, Inches(1.2), Inches(2.3), Inches(10.93), Inches(3.9))
para(tf, [("Selective Invariance Violations in\nLarge Language Model Moral Judgment", "bold")],
     size=30, align=PP_ALIGN.CENTER, color=INK, first=True, space_after=10)
para(tf, [("A Geometric Framework for Behavioral Manipulation Detection", None)],
     size=19, align=PP_ALIGN.CENTER, color=ATLASBLUE, space_after=22)
para(tf, [("Andrew H. Bond", "bold")], size=17, align=PP_ALIGN.CENTER, color=INK, space_after=2)
para(tf, [("Dept. of Computer Engineering, San Jose State University", None)],
     size=14, align=PP_ALIGN.CENTER, color=GRAY, space_after=16)
para(tf, [("IEEE BigDataService 2026 — Special Track on Secure AI", "em")],
     size=14, align=PP_ALIGN.CENTER, color=INK)

# ---- 2. the problem ---------------------------------------------------------
s = new_slide("The problem: LLMs as decision services", 2)
tf = textbox(s, Inches(0.55), Inches(1.0), Inches(12.2), Inches(2.9))
para(tf, [("LLMs increasingly gate ", None), ("safety-critical decisions", "hi"),
          (" — content moderation, clinical triage, legal analysis.", None)],
     size=17, bullet=True, first=True, space_after=10)
para(tf, [("A secure evaluator should judge the ", None), ("facts", "hi"),
          (", not their ", None), ("presentation", "hi"), (".", None)],
     size=17, bullet=True, space_after=10)
para(tf, [("Behavioral manipulation vulnerability", "gold"),
          (": an attacker shifts the decision by changing morally ", None), ("irrelevant", "em"),
          (" surface features — framing, tone, sensory detail — without changing the underlying situation.", None)],
     size=17, bullet=True)
block(s, Inches(0.55), Inches(4.35), Inches(12.2), Inches(2.35), "The measurement gap", [
    {"segments": [("Prior work probes one bias at a time and reports a ", None),
                  ("single scalar robustness score", "hi"),
                  (". Across n independent failure directions, a scalar destroys n−1 of them. ", None),
                  ("No post-hoc procedure recovers the lost structure.", "gold")]},
], body_size=16)

# ---- 3. our approach --------------------------------------------------------
s = new_slide("Our approach: a geometric vulnerability profile", 3)
tf = textbox(s, Inches(0.55), Inches(1.05), Inches(7.9), Inches(5.6))
para(tf, [("Map each judgment to a point in a ", None), ("7-D harm space", "hi", )],
     size=17, bullet=True, first=True, space_after=2)
para(tf, [("(physical, emotional, financial, autonomy, trust, social, identity; each 0–10).", None)],
     size=13, color=GRAY, space_after=10)
para(tf, [("Apply qualitatively distinct ", None), ("perturbations", "hi"),
          ("; measure ", None), ("displacement", "hi"), (" of the judgment vector.", None)],
     size=17, bullet=True, space_after=10)
para(tf, [("Output a ", None), ("per-model vulnerability profile", "hi"), (", not one number.", None)],
     size=17, bullet=True, space_after=10)
para(tf, [("Runs under a realistic ", None), ("$50/day", "hi"), (" compute budget.", None)],
     size=17, bullet=True)
s.shapes.add_picture(ASSETS + r"\erisml_apple_light.png", Inches(9.7), Inches(1.5), height=Inches(2.4))
tf = textbox(s, Inches(8.9), Inches(4.1), Inches(4.0), Inches(1.4))
para(tf, [("The 7-D space is a measurement-reliable projection of the ", None),
          ("DEME v3", "hi"), (" 9-D moral vector.", None)],
     size=12, align=PP_ALIGN.CENTER, color=GRAY, first=True)

# ---- 4. grounding table -------------------------------------------------------
s = new_slide("Grounding: 7-D harm space → DEME v3 9-D vector", 4)
rows = [
    ["Harm dim.", "DEME v3 axis", "Mapping"],
    ["physical", "physical harm", "direct"],
    ["autonomy", "autonomy / consent", "direct"],
    ["trust", "legitimacy / trust", "direct"],
    ["social", "societal / environmental", "direct"],
    ["financial", "fairness / equity", "partial"],
    ["emotional", "virtue / care", "partial"],
    ["identity", "privacy / standing", "partial"],
    ["—", "rights, epistemic quality", "not scored"],
]
table(s, Inches(2.87), Inches(1.05), Inches(7.6), rows, [2.6, 3.4, 1.6], size=14, row_h=0.34)
tf = textbox(s, Inches(0.9), Inches(4.55), Inches(11.5), Inches(1.0))
para(tf, [("The evaluation pipeline is the ", None), ("measurement front-end", "hi"),
          ("; the ErisML compiler's ", None), ("DEME bridge", "hi"),
          (" is the downstream decision kernel (", None), ("DEMEVerdict", "mono"), (").", None)],
     size=14, align=PP_ALIGN.CENTER, first=True)

# ---- 5. finding 1 -----------------------------------------------------------
s = new_slide("Finding 1 — Vulnerabilities are selective", 5)
tf = textbox(s, Inches(0.55), Inches(1.0), Inches(5.9), Inches(4.6))
para(tf, [("Displace judgments (real attack surface):", "bold")], size=16, first=True, space_after=6)
para(tf, [("Linguistic framing", "gold")], size=16, bullet=True, space_after=4)
para(tf, [("Emotional anchoring", "gold"), ("  (d = 0.6–1.1)", None)], size=16, bullet=True, space_after=4)
para(tf, [("Irrelevant sensory detail", "gold")], size=16, bullet=True, space_after=12)
para(tf, [("Do ", "bold"), ("not", "em"), (" displace:", "bold")], size=16, space_after=6)
para(tf, [("Gender swap", None)], size=16, bullet=True, space_after=4)
para(tf, [("Evaluation order", None)], size=16, bullet=True)
block(s, Inches(6.8), Inches(1.1), Inches(6.0), Inches(3.3), "The common thread", [
    {"segments": [("The three live surfaces all make morally irrelevant features ", None),
                  ("perceptually salient", "hi"), (".", None)], "space_after": 8},
    {"segments": [("⇒ The attack mechanism is ", None), ("salience manipulation", "gold"),
                  (" — and the framework discriminates it from noise.", None)]},
], body_size=15)
footnote(s, [("We lead with ", None), ("effect sizes", "hi"),
             (" (Cohen's d, displacement in harm-points) over σ; empirical control arms rule out stochastic drift.", None)], size=12)

# ---- 6. finding 2 -----------------------------------------------------------
s = new_slide("Finding 2 — Robustness profiles are dissociable", 6)
tf = textbox(s, Inches(0.55), Inches(1.05), Inches(12.2), Inches(2.6))
para(tf, [("No model dominates", "hi"), (" all attack surfaces.", None)],
     size=17, bullet=True, first=True, space_after=10)
para(tf, [("Claude: ", None), ("zero sycophancy", "gold"), (" — but ", None), ("worst", "gold"),
          (" emotional-anchoring recovery (20%) and worst divided attention.", None)],
     size=17, bullet=True, space_after=10)
para(tf, [("Flash 2.0: ", None), ("best", "gold"), (" anchoring recovery (73%) — but ", None),
          ("worst", "gold"), (" working memory.", None)],
     size=17, bullet=True)
block(s, Inches(0.55), Inches(4.3), Inches(12.2), Inches(2.2), "Consequence for Secure AI", [
    {"segments": [("Averaging partially-independent dimensions yields a number that ", None),
                  ("describes no model accurately.", "gold")], "space_after": 6},
    {"segments": [("Single-test certification provides false assurance.", "gold")]},
], body_size=16)

# ---- 7. validation ------------------------------------------------------------
s = new_slide("NEW — Harm-space validation (reviewer-driven)", 7)
tf = textbox(s, Inches(0.55), Inches(0.95), Inches(12.2), Inches(1.7))
para(tf, [("Is the 7-D space reliably measurable, or asserted?", "bold")], size=17, first=True, space_after=6)
para(tf, [("Inter-model agreement over an ", None), ("open, multi-family panel", "hi"),
          (" (6 models, 5 families: Qwen3, Qwen3-small, GLM-5, GPT-OSS, MiniMax-M2, Gemma) via the ", None),
          ("NRP managed LLM API", "hi"),
          ("; 31 scenarios, disjoint from systems under test, fully reproducible.", None)], size=15)
rows = [
    ["", "ICC(2,k)", "Krippendorff α"],
    ["Financial (highest)", "0.983", "0.902"],
    ["Trust", "0.958", "0.783"],
    ["Physical / Autonomy (lowest)", "0.893", "≈0.57"],
    [[("Overall", "bold")], [("0.969", "gold")], [("0.836", "gold")]],
]
table(s, Inches(3.42), Inches(3.05), Inches(6.5), rows, [3.3, 1.6, 1.6], size=14, row_h=0.36)
footnote(s, [("Test–retest median r = 0.96. Dimensions index a shared, model-independent structure.", None)],
         y=Inches(5.35), size=12)

# ---- 8. worked exploit ---------------------------------------------------------
s = new_slide("NEW — Worked exploit: threshold evasion", 8)
tf = textbox(s, Inches(0.55), Inches(1.1), Inches(12.2), Inches(5.2))
para(tf, [("Content-moderation filter: flag if total harm > T.", "bold")], size=18, first=True, space_after=14)
para(tf, [("Euphemistic rewriting (morally invariant) lowers panel-averaged harm by ", None),
          ("−14.0 points", "gold"),
          (" on the 0–70 scale (4/6 gold > 10 pts); dramatic only +7.3.", None)],
     size=17, bullet=True, space_after=12)
para(tf, [("At a median-calibrated threshold, ", None), ("3 of 6", "gold"),
          (" gold items silently reclassify FLAG → PASS.", None)],
     size=17, bullet=True, space_after=12)
para(tf, [("Asymmetry", "gold"), (": far easier to ", None), ("evade", "em"),
          (" (hide harm) than to ", None), ("trip", "em"),
          (" a false positive — the dangerous direction.", None)],
     size=17, bullet=True)

# ---- 9. kernel ---------------------------------------------------------------
s = new_slide("NEW — The exploit survives a principled kernel", 9)
tf = textbox(s, Inches(0.55), Inches(0.95), Inches(12.2), Inches(1.3))
para(tf, [("Route each scenario through the real ", None), ("ErisML DEME v3", "hi"), (" decision kernel:", None)],
     size=16, first=True, space_after=6)
para(tf, [("text → ", None), ("LLM extracts ethical facts", "hi"), (" → ", None),
          ("GenevaEMV3", "mono"), (" → typed verdict   ", None),
          ("(forbid → avoid → neutral → prefer → strongly_prefer)", None)],
     size=14, align=PP_ALIGN.CENTER)
block(s, Inches(0.55), Inches(2.55), Inches(12.2), Inches(2.5), "Across 161 scenario–model cases", [
    {"segments": [("Euphemistic rewriting flips forbid/avoid → permissive in ", None),
                  ("13.7%", "gold"), (" of cases; mean shift ", None), ("+0.65", "gold"),
                  (" ordinal steps.", None)], "bullet": True, "space_after": 8},
    {"segments": [("forbid verdicts fall ", None), ("44 → 27", "gold"),
                  (" (−39%); strongly_prefer rises 46 → 78.", None)], "bullet": True},
], body_size=16)
tf = textbox(s, Inches(0.55), Inches(5.45), Inches(12.2), Inches(1.2))
para(tf, [("A rule engine inherits, rather than cures, the front-end's salience vulnerability.", "gold"),
          ("  Harden ", None), ("fact extraction", "em"), (", not just the rule.", None)],
     size=16, first=True)

# ---- 10. since submission: defense ---------------------------------------------
s = new_slide("Since submission — a measured defense", 10)
tf = textbox(s, Inches(0.55), Inches(0.95), Inches(12.2), Inches(5.3))
para(tf, [("Equivalence-class averaging", "bold"),
          (": generate the input's paraphrase class (m paraphrases), ", None),
          ("average per-dimension perception over the class", "hi"), (", then decide.", None)],
     size=16, first=True, space_after=12)
para(tf, [("Pre-registered bar: decision-layer drift θₔ ≤ 0.5. At scale (60 held-out items, m = 6): raw drift ", None),
          ("0.407 → 0.219", "gold"), (" — the mechanism ", None), ("halves", "hi"),
          (" salience drift and meets the bar.", None)],
     size=16, bullet=True, space_after=10)
para(tf, [("LLM paraphrasers ", None), ("refuse 24%", "gold"),
          (" of harmful inputs. A non-refusing red-team paraphraser (NLLB back-translation, 6 pivots): ", None),
          ("θₔ = 0.301", "gold"), (" on harmful content — the hole is the ", None),
          ("generator's", "hi"), (", not the mechanism's.", None)],
     size=16, bullet=True, space_after=10)
para(tf, [("Trust boundary: refusal → singleton class → ", None), ("escalate by default", "hi"),
          ("; audit proof records class members.", None)],
     size=16, bullet=True)
footnote(s, [("Natural (not adversarial) paraphrases at scale. Post-camera-ready results, July 2026 — extended version in preparation.  ", None),
             ("pip install moral-spectrum-analyzer", "mono")], size=11)

# ---- 11. since submission: discovery --------------------------------------------
s = new_slide("Since submission — the harm space is extensible and falsifiable", 11)
tf = textbox(s, Inches(0.55), Inches(0.95), Inches(12.2), Inches(5.3))
para(tf, [("Discovery loop", "bold"),
          (": residual analysis flags candidate missing dimensions → pre-registered admission gate.", None)],
     size=16, first=True, space_after=12)
para(tf, [("Scorecard: 3 flagged → ", None), ("1 validated", "gold"), (" (identity_attack), ", None),
          ("1 retracted", "gold"), (" (threat), ", None), ("1 declined", "gold"),
          (" (sexual content → policy channel, not a moral axis). ", None),
          ("The gate has teeth.", "hi")],
     size=16, bullet=True, space_after=10)
para(tf, [("identity_attack", "mono"), (": cross-dataset held-out AUROC ", None), ("0.80", "gold"),
          (" CI [0.78, 0.83] (+0.25 over null, 2 corpora, n = 6400) — wired live as a ", None),
          ("10th channel", "hi"),
          ("; dominant feature of a learned moderation contraction (OOF AUROC ", None),
          ("0.863", "gold"), (", leakage-controlled).", None)],
     size=16, bullet=True, space_after=10)
para(tf, [("Cross-lingual at scale: invariance index ", None), ("0.72–0.80", "gold"),
          (" across es/ar/zh/hi/sw, ", None), ("harmful ≈ benign", "hi"), (".", None)],
     size=16, bullet=True)
footnote(s, [("Validated on the learned-encoder perception layer (", None), ("xbse", "mono"),
             ("); native DEME re-run of the five tracks is the extended version. Post-camera-ready results, July 2026.", None)], size=11)

# ---- 12. implications ------------------------------------------------------------
s = new_slide("Implications for Secure AI deployment", 12)
tf = textbox(s, Inches(0.55), Inches(1.1), Inches(12.2), Inches(5.2))
para(tf, [("Where to look", "hi"), (": attacks that repackage the same facts with different ", None),
          ("salience", "gold"), (" (euphemistic minimization).", None)],
     size=17, bullet=True, first=True, space_after=12)
para(tf, [("Prompt-level defenses are bounded", "hi"),
          (": explicit warnings recover only ~38% — a ceiling co-occurring with universal overconfidence (ECE 0.19–0.42).", None)],
     size=17, bullet=True, space_after=12)
para(tf, [("Ask the right question", "hi"), (": “which vulnerabilities matter for our use case?” — not “what's the robustness score?”", None)],
     size=17, bullet=True, space_after=12)
para(tf, [("Hardening target", "hi"), (": the fact-extraction front-end, since downstream kernels inherit its failures.", None)],
     size=17, bullet=True)

# ---- 13. repro + takeaways --------------------------------------------------------
s = new_slide("Reproducibility & takeaways", 13)
tf = textbox(s, Inches(0.55), Inches(0.95), Inches(12.2), Inches(1.7))
para(tf, [("Five tracks, multi-model, ~8,000 calls — under ", None), ("$50/day", "hi"),
          (" (Kaggle); validation panel on the open ", None), ("NRP managed LLM API", "hi"), (".", None)],
     size=16, bullet=True, first=True, space_after=8)
para(tf, [("Code + data: ", None), ("pip install agi-hpc", "mono"), (", tag ", None),
          ("bds2026-v1", "mono"), ("; decision kernel: ", None), ("ErisML / DEME v3", "hi"), (".", None)],
     size=16, bullet=True)
block(s, Inches(0.55), Inches(3.0), Inches(12.2), Inches(3.4), "Four takeaways", [
    {"segments": [("1.  Vulnerabilities are ", None), ("selective", "gold"),
                  (" — salience manipulation is the surface.", None)], "space_after": 8},
    {"segments": [("2.  Robustness profiles are ", None), ("dissociable", "gold"),
                  (" — no single score is safe.", None)], "space_after": 8},
    {"segments": [("3.  The exploit is ", None), ("end-to-end", "gold"),
                  (" — it flips scalar scores ", None), ("and", "em"), (" typed verdicts.", None)], "space_after": 8},
    {"segments": [("4.  It is ", None), ("defensible", "gold"),
                  (" — averaging over the paraphrase class halves drift (since submission).", None)]},
], body_size=16)

# ---- 14. thanks --------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
s.shapes.add_picture(ASSETS + r"\atlas_mark_light.png", Inches(5.75), Inches(0.9), height=Inches(1.85))
tf = textbox(s, Inches(1.2), Inches(3.1), Inches(10.93), Inches(3.4))
para(tf, [("Thank you", "hi")], size=30, align=PP_ALIGN.CENTER, first=True, space_after=10)
para(tf, [("Andrew H. Bond    ·    ", None), ("andrew.bond@sjsu.edu", "mono")],
     size=16, align=PP_ALIGN.CENTER, color=INK, space_after=6)
para(tf, [("Selective Invariance Violations in LLM Moral Judgment", None)],
     size=13, align=PP_ALIGN.CENTER, color=GRAY, space_after=2)
para(tf, [("IEEE BigDataService 2026 — Secure AI", None)],
     size=13, align=PP_ALIGN.CENTER, color=GRAY, space_after=12)
para(tf, [("Questions?", "em")], size=13, align=PP_ALIGN.CENTER, color=INK)

# ---- speaker notes (written for a presenter who did not do the research) ----
NOTES = [
# 1 — title
"""TIMING: ~12-15 min talk + Q&A. This paper was ACCEPTED at IEEE BigDataService 2026, Special Track on Secure AI (both reviews: weak accept).

If you are not Andrew: "I'm presenting on behalf of Andrew Bond, San Jose State University."

ONE-SENTENCE SUMMARY of the whole talk: LLMs that make moral/safety decisions can be manipulated just by REWORDING the input — we built a framework that measures exactly WHICH rewordings move WHICH models, showed the attack works end-to-end against a real moderation pipeline, and (new since the paper) built and measured a defense.

The word "geometric" just means: a judgment is a POINT in a multi-dimensional harm space, and we measure how far perturbations MOVE that point (displacement). No deep math needed on stage.""",

# 2 — problem
"""KEY TERM — "behavioral manipulation vulnerability": the attacker never changes the FACTS of a scenario, only the PRESENTATION (wording, tone, irrelevant detail). A secure evaluator should give the same verdict either way. These models don't.

Why it matters: LLMs already gate content moderation, triage, legal-intake style decisions. If wording moves the verdict, wording is an attack surface.

BOTTOM BLOCK (the measurement gap): prior work tests one bias at a time and reports ONE robustness number. Analogy that lands well: a single credit score vs. the full credit report — if failures have n independent directions, one scalar throws away n−1 of them, and you cannot reconstruct them afterwards.

Don't over-dwell here; the punchline slides are 8-11.""",

# 3 — approach
"""HOW THE FRAMEWORK WORKS (3 steps): (1) every model judgment is scored as a 7-dimensional harm vector — physical, emotional, financial, autonomy, trust, social, identity — each 0-10, so total harm is 0-70. (2) apply perturbations that should NOT change the moral evaluation. (3) measure how far the vector moved = displacement. Output is a per-model VULNERABILITY PROFILE, not one number.

$50/day: the whole 5-model, ~8,000-call evaluation fits a Kaggle budget — any team can afford to reproduce this. Worth saying out loud.

RIGHT SIDE — DEME v3: the 9-dimension moral vector of the ErisML compiler (Andrew's open-source ethics decision kernel). The 7-D space is the measurement-reliable projection of those 9 dimensions — i.e., the coordinates aren't invented for this paper; two axes were dropped because models couldn't score them reliably.""",

# 4 — grounding table
"""This table exists because reviewers asked "where do the 7 dimensions come from?" Answer: they are a PROJECTION of the DEME v3 9-axis moral vector. Four map directly, three partially, and two (rights, epistemic quality) are NOT scored — they produced unreliable judgments in preliminary testing, so the paper says so honestly rather than pretending.

Key phrase: "a reliability-driven projection, not a bijection."

Bottom line of the slide: this evaluation pipeline is the measurement FRONT-END; the ErisML compiler's DEME bridge is the downstream decision kernel that turns harm vectors into a typed verdict (DEMEVerdict). That wiring becomes important on slide 9, where we attack the whole chain.

If pressed on "identity" mapping weakly to privacy/standing: note it gets properly fixed by the new identity_attack channel on slide 11.""",

# 5 — finding 1
"""THE HEADLINE FINDING of the accepted paper. Vulnerabilities are SELECTIVE, not uniform:

MOVE judgments (all 5 models): linguistic framing (euphemistic vs dramatic rewording), emotional anchoring (Cohen's d 0.6-1.1 = medium-to-large), irrelevant sensory detail.
DO NOT move judgments: gender swap, evaluation order.

The common thread (right block): the three live surfaces all make morally irrelevant features PERCEPTUALLY SALIENT. So the attack mechanism is salience manipulation — attackers must change what's perceptually prominent, not just any surface token. That selectivity also proves the framework discriminates real vulnerabilities from noise.

FOOTNOTE (methods honesty, reviewers asked for this): headline stats are effect sizes (Cohen's d, harm-point displacement), not sigmas; every test is compared against replication control arms (same text re-judged) — apparent 6-sigma violations in early analyses VANISHED under those controls. Say this if any statistician looks skeptical.""",

# 6 — finding 2
"""SECOND FINDING: robustness profiles are DISSOCIABLE — no model dominates.

Concrete contrast to say out loud: Claude has ZERO sycophancy (never accepts a fabricated correction) — but the WORST emotional-anchoring recovery (20%) and worst divided attention. Gemini Flash 2.0 recovers from emotional manipulation the BEST (73%) — but has the worst working memory.

Consequence (bottom block): averaging partially-independent dimensions gives a number that describes NO model accurately. So single-test certification — "this model scored X on robustness" — is FALSE ASSURANCE. This is the main deployment message of the accepted paper.

Anticipate: "which model should I use then?" Answer: depends which attack surfaces matter for YOUR use case — that's exactly why the output is a profile, not a ranking.""",

# 7 — validation
"""Both reviewers' #1 concern: "is the 7-D harm space validated or asserted?" This slide is the camera-ready answer.

Design: 6 OPEN models from 5 different families (Qwen3 x2, GLM-5, GPT-OSS, MiniMax-M2, Gemma) each score the full 7-D vector on 31 scenarios, 3 repetitions. Panel is served by the NRP (National Research Platform) managed LLM API — fully open, no proprietary endpoints, so anyone can re-run it. The panel is DISJOINT from the 5 models under test.

TERMS: ICC(2,k) = intraclass correlation, two-way random, average-measures — the standard inter-rater reliability statistic; 0.969 overall is excellent. Krippendorff's alpha = chance-corrected agreement; 0.836 overall is strong.

Honest caveat if asked why physical/autonomy are lowest (~0.57): near-floor artifact — interpersonal scenarios rarely involve physical harm, so tiny disagreements at 0-1 crush the coefficient. Test-retest r=0.96 sets the stochastic floor the perturbation effects are measured against.""",

# 8 — worked exploit
"""Reviewer ask #3: "show an end-to-end exploit, not just displacement." Walk it as a story:

You run a moderation filter: flag anything with total harm > T. The attacker controls ONLY the wording. They rewrite the same facts euphemistically — corporate-liability language, "involuntary separation from the platform" style. Panel-averaged harm drops 14.0 points on the 0-70 scale. At a median-calibrated threshold, 3 of the 6 audited gold scenarios silently flip from FLAG to PASS. Nothing about the underlying event changed.

THE ASYMMETRY IS THE SCARY PART: euphemism moves scores twice as much as dramatization (-14 vs +7.3). So the filter is far easier to EVADE (hide real harm) than to TRIP (fake a false positive) — it fails in the dangerous direction.

Honest scope if asked: n=6 gold items with hand-audited rewrites = an EXISTENCE PROOF of the end-to-end exploit; the displacement magnitude itself is established on 31 scenarios.""",

# 9 — kernel
"""Natural objection: "so don't use a scalar threshold — use a principled rule engine." This slide kills that hope.

Setup: route each scenario through the real ErisML DEME v3 kernel: an LLM extracts structured ethical FACTS (rights violation? valid consent? harm severity?), then GenevaEMV3 — a fixed, deterministic rule module — maps facts to a typed verdict: forbid / avoid / neutral / prefer / strongly_prefer.

Result across 161 scenario-model cases: euphemistic rewriting flips a restrictive verdict (forbid/avoid) to a permissive one in 13.7% of cases; forbid verdicts drop from 44 to 27 (-39%).

THE QUOTABLE LINE: "a rule engine INHERITS, rather than cures, the front-end's salience vulnerability." The rules are fine — the FACTS feeding them were corrupted by the rewording. So hardening must target fact extraction / perception. That sets up the next slide (the defense).""",

# 10 — defense (since submission)
"""IMPORTANT FRAMING: slides 10-11 are work done AFTER the camera-ready. Say explicitly: "these results are post-publication, from the extended version in preparation." Do not present them as claims of the accepted paper.

THE IDEA: a single wording is one arbitrary member of an equivalence class of morally identical texts. So don't judge the wording — judge the CLASS: generate m paraphrases of the input, average the per-dimension perception scores over the class, then decide. (For ML folks: semantic randomized smoothing / SmoothLLM, but with meaning-preserving paraphrases.)

NUMBERS: theta_d = decision-layer drift under re-description; lower is better; bar of 0.5 was PRE-REGISTERED. At scale (60 held-out items, m=6): raw drift 0.407 -> 0.219. The mechanism HALVES drift and meets the bar.

THE HOLE AND ITS LOCATION: LLM paraphrasers refuse 24% of harmful inputs — exactly the content that matters. Red-team fix: NLLB back-translation through 6 languages refuses NOTHING; theta_d = 0.301 on harmful content. Conclusion: the hole belongs to the GENERATOR, not the mechanism.

Deployment rule: if the generator refuses -> singleton class -> ESCALATE to a human by default; the audit proof records the class member hashes.

Caveat if asked: at-scale paraphrases are natural, not adversarial; adversarial-at-scale is open work.""",

# 11 — discovery (since submission)
"""Also post-camera-ready. This answers "is the harm space validated?" at a deeper level: the dimension set is EXTENSIBLE and FALSIFIABLE.

THE LOOP: residual analysis asks "is there moderation signal the current axes don't carry?" -> flags candidates -> each faces a PRE-REGISTERED admission gate (cross-dataset transfer, margin over null, fuzz test).

SCORECARD — say all three outcomes, the rejections are the credibility: 3 flagged. 1 VALIDATED: identity_attack, held-out AUROC 0.80, CI [0.78-0.83], on two independent corpora, n=6400 — now live as a 10th channel. 1 RETRACTED: threat (failed a balanced resample). 1 DECLINED: sexual content — analysis showed it's a PLATFORM POLICY signal, not a moral axis (consensual explicitness isn't a moral violation; the moral weight was harassment, already covered). "The gate has teeth."

Also: identity_attack is the dominant feature of the learned moderation contraction (AUROC 0.863, leakage-controlled), and invariance holds cross-lingually (0.72-0.80 across Spanish/Arabic/Chinese/Hindi/Swahili, harmful ≈ benign).

HONESTY LINE if asked: these validations are on the learned-encoder perception layer (xbse), not on the LLM judges of the paper — re-running the paper's five tracks natively through this instrument IS the extended version.

If asked "isn't identity_attack just a Perspective API label?": yes, the LABEL exists; the contribution is the INSTRUMENT — discovery from residuals, pre-registered admission, provenance-carrying deployment.""",

# 12 — implications
"""Deployment guidance — four points, one line each:

1. WHERE TO LOOK: the live attack surface is salience repackaging — especially euphemistic minimization. Watch for measured, liability-flavored language hiding severity.
2. PROMPT DEFENSES ARE BOUNDED: telling the model "you are being manipulated" recovers only ~38% — and this ceiling co-occurs with universal overconfidence (ECE 0.19-0.42). System-prompt guardrails fail two times out of three.
3. ASK THE RIGHT QUESTION: not "what's the robustness score?" but "WHICH vulnerabilities matter for OUR use case?" — profiles, not rankings.
4. HARDENING TARGET: the fact-extraction / perception front-end — downstream kernels inherit its failures (slide 9). And slide 10 showed perception CAN be hardened measurably.

ECE = expected calibration error: gap between stated confidence and actual accuracy.""",

# 13 — takeaways
"""Reproducibility first (quick): everything is pip-installable — the paper artifact (agi-hpc, tag bds2026-v1), and the new instrument: moral-spectrum-analyzer + xbse, both on PyPI as of July 2026. Decision kernel: ErisML / DEME v3, open source. Whole evaluation ran under $50/day on Kaggle; the validation panel runs on the open NRP API.

THE FOUR TAKEAWAYS — read them slowly, this is the summary the audience leaves with:
1. Vulnerabilities are SELECTIVE — salience manipulation is the attack surface.
2. Robustness profiles are DISSOCIABLE — no single score is safe; certify per-surface.
3. The exploit is END-TO-END — it flips scalar thresholds AND typed rule-engine verdicts.
4. (since submission) It is DEFENSIBLE — deciding on the paraphrase CLASS instead of the wording halves drift, with the residual hole located and escalated.

Attack -> measurement -> defense. That's the arc.""",

# 14 — thanks / Q&A prep
"""Q&A PREP — likely questions and short answers:

Q: Why 7 dimensions and not 9? A: Two DEME axes (rights, epistemic quality) produced unreliable model scores; dropped for measurement reliability and disclosed. The extended version scores natively in 9+1-D.
Q: Isn't this just prompt sensitivity? A: Prompt sensitivity is undirected noise; we show DIRECTED, selective, reproducible displacement under meaning-preserving transforms, with control arms — plus an end-to-end decision consequence.
Q: How is the defense different from SmoothLLM / randomized smoothing? A: Same aggregation idea, but the perturbation family is meaning-preserving PARAPHRASE (semantic, not character-level), the certified quantity is a multi-dimensional moral decision, and the paraphrase generator explicitly joins the trust boundary (refusal -> escalate).
Q: Do the new results change the accepted paper? A: No — they're clearly labeled post-camera-ready; extended/journal version in preparation.
Q: Model coverage is small (5 models / 2 families)? A: Acknowledged in the paper; the harm-space validation panel adds 6 open models / 5 families; trend claims are flagged as suggestive.
Q: Human agreement rather than inter-model agreement? A: Open panel was chosen for reproducibility (no proprietary endpoints); human-rater validation is future work.

Contact: andrew.bond@sjsu.edu""",
]

for _slide, _note in zip(prs.slides, NOTES):
    _slide.notes_slide.notes_text_frame.text = _note

prs.save(OUT)
print(f"saved {OUT} with {len(prs.slides.slides if hasattr(prs.slides,'slides') else prs.slides._sldIdLst)} slides")
