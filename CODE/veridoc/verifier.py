"""Stage 4: the grounding verifier.

A candidate fact is loaded only if no check fails:
  agree     two independent OCR readings give the same value (scanned pages)
  span      the evidence span exists on the page and contains the value
  nli       the evidence entails the verbalized fact (cross-encoder; advisory by default, see NLI_MODE)
  shacl     datatype + SHACL shapes on the per-document instance graph
  crossdoc  agreement and arithmetic across fields and documents
  template  a free-text value on the application form must not be text printed on the blank form (a label)

A fact that fails is not deleted: it is quarantined with a reason code and
can be restored by a second opinion or a human reviewer (see apply_review).
"""
from __future__ import annotations
import json, re
from datetime import date
from collections import defaultdict
from rapidfuzz import fuzz
from rdflib import Graph, Namespace, Literal, RDF, URIRef
from rdflib.namespace import XSD
from pyshacl import validate

from . import config
from .schema import Fact, PASS, FAIL, SKIP
from .fields import FIELD_TYPES, PERIODS_PER_YEAR, norm_value

EX = Namespace("http://veridoc.example/onto#")
ENTITY_OF = {}
for _e, _fs in {
    "Group": "group_name group_number group_street group_state situs_state eligible_employees proposed_effective_date "
             "signature_date presented_by proposal_date",
    "Certificate": "face_amount certificate_number certificate_date maturity_date risk_class plan min_paidup_amount "
                   "premium_frequency loan_interest_rate annual_premium payroll_frequency payroll_deduction_amount",
    "Insured": "insured_name date_of_birth issue_age ee_city ee_state ee_zip date_of_hire annual_salary job_title",
    "Rider": "rider_date rider_benefit_amount rider_annual_premium rider_termination_date",
}.items():
    for _f in _fs.split():
        ENTITY_OF[_f] = _e
XSD_OF = {"date": XSD.date, "money": XSD.decimal, "int": XSD.integer, "percent": XSD.decimal}


def _nw(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _set(f: Fact, check: str, result: str, detail: str = "") -> None:
    f.checks[check] = {"result": result, "detail": detail}


# ------------------------------------------------------------------ agree
AGREE_POLICY = "strict"   # strict | corroborate | conflict_only; compared in E8 (EVALUATIONS/eval_agree_policy.py)


def _in_text(value: str, text: str) -> bool:
    """value appears in text as a whole token sequence (whitespace-flexible, case-sensitive)."""
    toks = value.split()
    if not toks:
        return False
    rx = r"(?<![A-Za-z0-9])" + r"\s*".join(re.escape(t) for t in toks) + r"(?![A-Za-z0-9])"
    return re.search(rx, text) is not None


def merge_readings(primary: list, alt: list, ocr_pages: set, texts: dict | None = None, policy: str | None = None) -> list:
    """Combine candidates from two OCR readings of the same pages.

    Both readings extract the same normalized value -> agree=pass.
    One reading extracts a value the other does not:
      strict         -> fail (the reading that missed it may have missed it for a reason)
      corroborate    -> pass if the value's text appears verbatim in the other reading's page text
                        (the other reading saw it, only the extraction pattern did not fire); else fail
      conflict_only  -> pass unless the other reading extracted a different value
    texts: (doc_id, page) -> [reading 1 text, reading 2 text]; needed for corroborate."""
    policy = policy or AGREE_POLICY
    def key(f): return (f.doc_id, f.page, f.field)
    by_p, by_a = defaultdict(list), defaultdict(list)
    for f in primary: by_p[key(f)].append(f)
    for f in alt: by_a[key(f)].append(f)
    out = []

    def single(f, other_vals, other_text, me, other):
        detail = f"only in reading {me}" + (f"; reading {other} has {sorted(other_vals)}" if other_vals else "")
        if policy == "conflict_only" and not other_vals:
            _set(f, "agree", PASS, f"reading {me} only; reading {other} extracted nothing (lenient)")
        elif policy == "corroborate" and other_text is not None and _in_text(f.value, other_text):
            _set(f, "agree", PASS, f"reading {me} extracted it; text also in reading {other}")
        else:
            _set(f, "agree", FAIL, detail)

    for k in sorted(set(by_p) | set(by_a)):
        if (k[0], k[1]) not in ocr_pages:              # native page: one exact reading
            for f in by_p[k]:
                _set(f, "agree", SKIP, "native text layer"); out.append(f)
            continue
        t = (texts or {}).get((k[0], k[1]), [])
        t1, t2 = (t[0] if len(t) > 0 else None), (t[1] if len(t) > 1 else None)
        vp = {f.normalized: f for f in by_p[k]}
        va = {f.normalized: f for f in by_a[k]}
        for v, f in vp.items():
            if v in va:
                _set(f, "agree", PASS, "both OCR readings")
            else:
                single(f, set(va), t2, 1, 2)
            out.append(f)
        for v, f in va.items():
            if v not in vp:
                f.fact_id += ":alt"
                single(f, set(vp), t1, 2, 1)
                out.append(f)
    return out


# ------------------------------------------------------------------ span
def check_span(f: Fact, page_texts: list, threshold: int = 90, value_exact: bool = True) -> None:
    """Evidence must be on the page (any available reading) and contain the value.
    The span may match fuzzily (OCR noise), but with value_exact the value itself
    must appear verbatim on the page: a one-character fabrication then fails even
    though its long span still matches fuzzily (see EVALUATIONS, E5)."""
    span, value = _nw(f.span), _nw(f.value)
    if not span or value not in span:
        return _set(f, "span", FAIL, "value_not_in_span")
    texts = [_nw(t) for t in page_texts]
    if value_exact and not any(value in t for t in texts):
        return _set(f, "span", FAIL, "value_not_on_page")
    if any(span in t for t in texts):
        return _set(f, "span", PASS, "exact")
    best = max((fuzz.partial_ratio(span, t) for t in texts), default=0)
    if best >= threshold:
        return _set(f, "span", PASS, f"fuzzy:{best:.0f}")
    _set(f, "span", FAIL, f"not_found:{best:.0f}")


# ------------------------------------------------------------------ nli
_NLI = None
# How the NLI check is used. Measured in EVALUATIONS/eval_nli.py before it is allowed to quarantine anything:
#   off       not run
#   advisory  run when the model is installed; the result is recorded and shown, but does not quarantine
#   hard      a fact below the entailment threshold is quarantined
NLI_MODE = "advisory"
NLI_HYPOTHESIS = "label"     # chosen in E5b (3 of 34 true facts rejected, against 8 for "field"). field: "The rider termination date is X."   label: wording printed on the form
NLI_PREMISE = "line"         # line: the evidence line only               context: document title + 2 lines above + line
NLI_THRESHOLD = 0.5

# Hypothesis wording that follows the label printed on the page (no entity qualifier the line does not state).
LABEL_TEMPLATE = {
    "presented_by": "It was presented by {v}.", "proposal_date": "The date is {v}.", "signature_date": "The date is {v}.",
    "group_street": "The street is {v}.", "group_state": "The state is {v}.",
    "eligible_employees": "The number of eligible employees is {v}.", "insured_name": "The insured is {v}.",
    "min_paidup_amount": "The minimum paid-up life insurance amount is {v}.",
    "premium_frequency": "The initial premium payment frequency is {v}.",
    "rider_benefit_amount": "The benefit amount is {v}.", "rider_annual_premium": "The annual premium is {v}.",
    "rider_termination_date": "The termination date is {v}.", "annual_premium": "The annual certificate premium is {v}.",
    "ee_city": "The city is {v}.", "ee_state": "The state is {v}.", "ee_zip": "The zip code is {v}.",
    "payroll_frequency": "The premium is {v}.", "payroll_deduction_amount": "The premium is {v}.",
}


def verbalize(f: Fact, style: str | None = None) -> str:
    style = style or NLI_HYPOTHESIS
    if style == "label" and f.field in LABEL_TEMPLATE:
        return LABEL_TEMPLATE[f.field].format(v=f.value)
    return f"The {f.field.replace('_', ' ')} is {f.value}."


def premise(f: Fact, texts: dict | None = None, style: str | None = None, lines_above: int = 2) -> str:
    """Text the hypothesis is tested against. 'context' adds the document title and the lines above the
    evidence line, so a qualifier such as "rider" or "proposal" is available to the model."""
    style = style or NLI_PREMISE
    if style == "line" or not texts:
        return f.span
    first = f.span.split("\n")[0]
    for t in texts.get((f.doc_id, f.page), []):
        i = t.find(first)
        if i >= 0:
            above = t[:i].rstrip("\n").split("\n")[-lines_above:]
            title = config.DOC_TITLES.get(config.doc_type(f.doc_id), "document")
            return f"This page is from the {title.lower()}.\n" + "\n".join(above + [f.span])
    return f.span


def nli_probs(pairs: list, model: str = "cross-encoder/nli-deberta-v3-base"):
    """Entailment probability for (premise, hypothesis) pairs, or None when the model is unavailable."""
    global _NLI
    try:
        if _NLI is None:
            from sentence_transformers import CrossEncoder
            _NLI = CrossEncoder(model)
    except Exception:                                      # model or library unavailable
        return None
    import numpy as np
    if not pairs:
        return np.zeros(0)
    logits = np.asarray(_NLI.predict(pairs))
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True))[:, 1]        # label order: contradiction, entailment, neutral


def check_nli(facts: list, texts: dict | None = None, threshold: float | None = None) -> None:
    """Entailment of the verbalized fact by its evidence. Skipped (not failed) when the model is not
    installed, so the pipeline runs without a download. Whether a fail quarantines depends on NLI_MODE."""
    threshold = NLI_THRESHOLD if threshold is None else threshold
    probs = nli_probs([(premise(f, texts), verbalize(f)) for f in facts])
    if probs is None:
        for f in facts:
            _set(f, "nli", SKIP, "model_unavailable")
        return
    for f, p in zip(facts, probs):
        _set(f, "nli", PASS if p >= threshold else FAIL, f"p_entail:{p:.2f}")


# ------------------------------------------------------------------ shacl
def instance_graph(facts: list) -> Graph:
    g = Graph(); g.bind("ex", EX)
    for f in facts:
        ent = ENTITY_OF.get(f.field)
        if not ent or not f.normalized:
            continue
        node = URIRef(f"{EX}{f.doc_id}/{ent}")
        g.add((node, RDF.type, EX[ent]))
        ftype = FIELD_TYPES.get(f.field, "text")
        g.add((node, EX[f.field], Literal(f.normalized, datatype=XSD_OF[ftype]) if ftype in XSD_OF else Literal(f.normalized)))
    return g


def check_shacl(facts: list) -> None:
    for f in facts:
        if not f.normalized:
            _set(f, "shacl", FAIL, "datatype")
    ok = [f for f in facts if f.normalized]
    shapes = Graph().parse(config.ONTOLOGY / "shapes.ttl")
    _, report, _ = validate(instance_graph(ok), shacl_graph=shapes, inference="none")
    SH = Namespace("http://www.w3.org/ns/shacl#")
    bad = defaultdict(set)
    for r in report.subjects(RDF.type, SH.ValidationResult):
        node, path, comp = report.value(r, SH.focusNode), report.value(r, SH.resultPath), report.value(r, SH.sourceConstraintComponent)
        if node is None or path is None:
            continue
        doc_id = str(node).split("#")[-1].split("/")[0]
        bad[(doc_id, str(path).split("#")[-1])].add(str(comp).split("#")[-1].replace("ConstraintComponent", ""))
    for f in ok:
        codes = bad.get((f.doc_id, f.field))
        _set(f, "shacl", FAIL if codes else PASS, ",".join(sorted(codes)) if codes else "")


# ------------------------------------------------------------------ crossdoc
CROSSDOC_MATCH = "canonical"   # canonical | fuzzy | surface; compared in E12 (EVALUATIONS/eval_canonical.py)
# Documents that describe the same individual of each class in this case. Only these are compared field by
# field: the policy specimen is a sample for another group, and the proposal's rate illustrations are examples,
# not this employee's certificate, so their values are a different entity, not a conflict.
CASE_DOCS = {"Group": ("proposal", "master_app"), "Certificate": ("certificate", "employee_app_blank"),
             "Insured": ("certificate", "employee_app_blank"), "Rider": ("certificate",)}


def same_value(field: str, raws: list, method: str | None = None) -> tuple:
    """(agree, detail) for values of one field read from different documents."""
    from .keys import canon
    method = method or CROSSDOC_MATCH
    if method == "surface":                                   # the text as printed
        return len({_nw(r) for r in raws}) == 1, ""
    if method == "fuzzy":                                     # the earlier rule: similarity of normalized text
        vals = sorted({norm_value(FIELD_TYPES.get(field, "text"), r) or _nw(r) for r in raws}, key=len)
        return all(fuzz.token_set_ratio(vals[0], v) >= 90 for v in vals[1:]), ""
    vals = {canon(field, r) for r in raws}
    if None in vals:
        return False, ":unparsed"
    if len(vals) == 1:
        return True, ""
    v = sorted(vals, key=len)
    return False, ":truncated" if all(x.startswith(v[0]) for x in v[1:]) else ":differ"


def check_crossdoc(facts: list) -> list:
    """Agreement and arithmetic across fields/documents. Returns soft warnings
    (date-order observations that do not quarantine)."""
    warnings = []
    for f in facts:
        _set(f, "crossdoc", SKIP, "no rule")
    live = [f for f in facts if f.normalized and
            all(f.checks.get(c, {}).get("result") != FAIL for c in ("agree", "span", "shacl", "template"))]
    by = defaultdict(list)
    for f in live:
        by[f.field].append(f)

    def one(field):
        vals = {f.normalized for f in by[field]}
        return next(iter(vals)) if len(vals) == 1 else None

    def mark(fields, result, detail):
        for fld in fields:
            for f in by[fld]:
                _set(f, "crossdoc", result, detail)

    # 1. a single-valued field read more than once in the case's documents (any page, any document) has
    #    one canonical value (veridoc/keys.py: dates, money, states, frequencies and names in one form).
    #    Exact comparison of canonical forms, not similarity: a truncated or redacted name ("Quest" vs
    #    "Quest Benefits") is a conflict for review, not a match. On a conflict every reading of the field
    #    goes to review, because the check cannot tell which one is right.
    for field, fs in by.items():
        fs = [f for f in fs if f.doc_id in CASE_DOCS.get(ENTITY_OF.get(field), ())]
        if len(fs) > 1:
            ok, detail = same_value(field, [f.value for f in fs])
            for f in fs:
                _set(f, "crossdoc", PASS if ok else FAIL, f"{field}_one_value_in_case{detail}")
    # 2. situs state (proposal) = group state (master application)
    a, b = one("situs_state"), one("group_state")
    if a and b:
        mark(["situs_state", "group_state"], PASS if a == b else FAIL, "state_proposal_vs_master_app")
    # 3. payroll arithmetic: deduction x periods = annual certificate premium + rider premium
    amt, freq, prem, rider = one("payroll_deduction_amount"), one("payroll_frequency"), one("annual_premium"), one("rider_annual_premium")
    if amt and freq and prem:
        total = float(prem) + float(rider or 0)
        n = PERIODS_PER_YEAR[freq]
        ok = abs(float(amt) * n - total) <= 0.005 * n + 0.01
        mark(["payroll_deduction_amount", "payroll_frequency", "annual_premium", "rider_annual_premium"],
             PASS if ok else FAIL, f"premium_arithmetic:{amt}x{n}={float(amt)*n:.2f} vs {total:.2f}")
    # 4. issue age matches date of birth at the certificate date
    dob, age, cd = one("date_of_birth"), one("issue_age"), one("certificate_date")
    if dob and age and cd:
        d0, d1 = date.fromisoformat(dob), date.fromisoformat(cd)
        real = d1.year - d0.year - ((d1.month, d1.day) < (d0.month, d0.day))
        mark(["date_of_birth", "issue_age"], PASS if real == int(age) else FAIL, f"age_arithmetic:{real}")
    # 5. soft: a certificate should not predate the group's proposed effective date
    ped = one("proposed_effective_date") or next((f.normalized for f in facts if f.field == "proposed_effective_date" and f.normalized), None)
    if ped and cd and cd < ped:
        warnings.append({"rule": "certificate_before_group_effective_date", "certificate_date": cd, "proposed_effective_date": ped,
                         "note": "The sample certificate predates the master application: the documents are examples from different cases."})
    return warnings


# ------------------------------------------------------------------ driver
# ------------------------------------------------------------------ template
FORM_MARK = "MassMutual@WORK"        # printed on every page of the employee application
_TEMPLATE = None


def _phrase(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def template_text() -> str:
    """All text printed on the blank employee application, as one normalized string."""
    global _TEMPLATE
    if _TEMPLATE is None:
        from . import extract
        _TEMPLATE = " " + _phrase(" ".join(p.text for p in extract.load_pages("employee_app_blank"))) + " "
    return _TEMPLATE


def check_template(facts: list, texts: dict) -> None:
    """On the application form, a free-text "value" that is itself printed on the blank form is a label or
    an instruction, not something a person filled in ("Job Title/Occupation", "State Zip Code").
    Typed fields are left to the datatype check. A real value that happens to be printed on the form
    (a city named in the insurer's address) would be quarantined too and go to review."""
    for f in facts:
        on_form = config.doc_type(f.doc_id) == "employee_app_blank" or any(FORM_MARK in t for t in texts.get((f.doc_id, f.page), []))
        if not on_form or FIELD_TYPES.get(f.field, "text") not in ("text", "name"):
            _set(f, "template", SKIP, "not a free-text field on the application form")
            continue
        v = _phrase(f.value)
        hit = len(v) >= 3 and f" {v} " in template_text()
        _set(f, "template", FAIL if hit else PASS, "printed_on_blank_form" if hit else "")


CHECKS = ("agree", "span", "nli", "shacl", "crossdoc", "template")


def hard_checks() -> tuple:
    """The checks whose failure quarantines a fact (nli only in hard mode)."""
    return tuple(c for c in CHECKS if c != "nli" or NLI_MODE == "hard")


HARD = hard_checks()


def decide(f: Fact, use: tuple | None = None) -> None:
    use = hard_checks() if use is None else use
    fails = [f"{c}:{f.checks[c]['detail']}" for c in use if f.checks.get(c, {}).get("result") == FAIL]
    f.status, f.reason = ("quarantined", fails[0]) if fails else ("verified", "")


def verify(facts: list, texts: dict, use: tuple | None = None, nli: bool = True) -> list:
    """texts: (doc_id, page) -> list of page texts (one per reading)."""
    for f in facts:
        f.checks.setdefault("agree", {"result": SKIP, "detail": ""})
        check_span(f, texts.get((f.doc_id, f.page), []))
    if nli and NLI_MODE != "off":
        check_nli(facts, texts)
    check_shacl(facts)
    check_template(facts, texts or {})
    warnings = check_crossdoc(facts)       # last: compares only facts that passed the other hard checks
    for f in facts:
        decide(f, use)
    return warnings


def apply_review(facts: list, review_path=None) -> int:
    """Human review of the quarantine queue: {"fact_id": "accept" | "reject"}."""
    path = review_path or (config.PROCESSED / "review.json")
    if not path.exists():
        return 0
    decisions, n = json.loads(path.read_text()), 0
    for f in facts:
        d = decisions.get(f.fact_id)
        if d == "accept" and f.status == "quarantined":
            f.status = "human_verified"; n += 1
        elif d == "reject":
            f.status = "rejected"; n += 1
    return n
