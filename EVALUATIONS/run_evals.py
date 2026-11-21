"""Preliminary evaluations E1, E3, E4, E5 (E2, text fidelity, is eval_text_fidelity.py).

Run from the repo root:  python EVALUATIONS/run_evals.py
Writes JSON files to EVALUATIONS/results/ and prints a short summary.
"""
import json, random, re, sys, copy
from collections import defaultdict, Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from veridoc import config, extract, fields, verifier, pipeline, deduction
from veridoc.kg import LOADED

OUT = config.RESULTS
OUT.mkdir(parents=True, exist_ok=True)


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=1))
    return obj


# ---------------------------------------------------------------- E1 routing
def e1_routing():
    rows = []
    for doc_id in config.DOC_FILES:
        ps = extract.load_pages(doc_id)
        nat = [p for p in ps if p.route == "native"]; ocr = [p for p in ps if p.route == "ocr"]
        rows.append({"doc": doc_id, "pages": len(ps), "native": len(nat), "ocr": len(ocr),
                     "sec_per_native_page": round(sum(p.seconds for p in nat) / len(nat), 2) if nat else None,
                     "sec_per_ocr_page": round(sum(p.seconds for p in ocr) / len(ocr), 2) if ocr else None,
                     "mean_ocr_conf": round(sum(w.conf for p in ocr for w in p.words) / max(1, sum(len(p.words) for p in ocr)), 1) if ocr else None,
                     "watermark_glyphs_removed": sum(p.features["n_rotated_chars"] for p in ps),
                     "covered_chars_dropped": sum(p.features.get("n_hidden_chars", 0) for p in ps)})
    tot = {"doc": "TOTAL", "pages": sum(r["pages"] for r in rows), "native": sum(r["native"] for r in rows), "ocr": sum(r["ocr"] for r in rows)}
    return dump("e1_routing.json", {"rows": rows, "total": tot})


# ---------------------------------------------------------------- E3 fields
def load_gold():
    g = json.loads((config.GOLD / "fields_gold_draft.json").read_text())["items"]
    return {(d, f): v for d, f, v in g}


def score(facts, gold, docs=None):
    """Distinct (doc, field, value) level. Only gold (doc, field) keys are scored."""
    emitted = {(f.doc_id, f.field, f.normalized or f.value.lower()) for f in facts if (f.doc_id, f.field) in gold}
    if docs:
        emitted = {e for e in emitted if e[0] in docs}
        gold = {k: v for k, v in gold.items() if k[0] in docs}
    correct = {e for e in emitted if gold[(e[0], e[1])] == e[2]}
    pos = {k for k, v in gold.items() if v is not None}
    found = {(e[0], e[1]) for e in correct}
    on_null = [e for e in emitted if gold[(e[0], e[1])] is None]
    p = len(correct) / len(emitted) if emitted else 1.0
    r = len(found) / len(pos) if pos else 1.0
    return {"emitted": len(emitted), "correct": len(correct), "wrong": len(emitted) - len(correct), "gold_values": len(pos),
            "precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "values_on_null_fields": len(on_null)}


def systems():
    """Candidate/verified fact lists for every configuration compared."""
    primary, alt, texts = pipeline.load_readings()
    ocr = {(p.doc_id, p.page) for p in primary if p.route == "ocr"}
    out = {}
    r1 = [f for p in primary for f in fields.extract_rules(p)]
    r2 = [f for k, p in alt.items() for f in fields.extract_rules(p)] + [f for p in primary if p.route == "native" for f in fields.extract_rules(p)]
    out["rules, OCR reading 1 only"] = r1
    out["rules, OCR reading 2 only"] = r2
    both = pipeline.candidates(primary, alt)
    out["rules, union of readings (no verifier)"] = copy.deepcopy(both)
    verifier.verify(both, texts)
    out["rules + verifier (pipeline)"] = [f for f in both if f.status in LOADED]
    prox = pipeline.candidates(primary, alt, extractor=fields.extract_proximity)
    out["proximity baseline (no verifier)"] = copy.deepcopy(prox)
    verifier.verify(prox, texts)
    out["proximity + verifier"] = [f for f in prox if f.status in LOADED]
    # optional extractors: included when their model / output exists
    try:
        from veridoc import crf
        if crf.MODEL.exists():
            c = pipeline.candidates(primary, alt, extractor=crf.extract_crf)
            out["CRF, trained on synthetic dev forms (no verifier)"] = copy.deepcopy(c)
            verifier.verify(c, texts)
            out["CRF + verifier"] = [f for f in c if f.status in LOADED]
    except ImportError:
        pass
    llm_path = config.FACTS / "facts_llm.json"
    if llm_path.exists():
        from veridoc.schema import load_facts
        lf = load_facts(llm_path)
        model = lf[0].extractor if lf else "llm"
        out[f"LLM {model} (no verifier)"] = copy.deepcopy(lf)
        verifier.verify(lf, texts)
        out[f"LLM {model} + verifier"] = [f for f in lf if f.status in LOADED]
    return out, both, prox, texts, primary, alt


def e3_fields(sys_facts):
    gold = load_gold()
    scanned = {"master_app", "certificate"}
    res = {name: {"all": score(fs, gold), "scanned_docs": score(fs, gold, scanned)} for name, fs in sys_facts.items()}
    return dump("e3_fields.json", {"gold": "DATA/gold/fields_gold_draft.json (draft, single annotator)",
                                   "gold_items": len(gold), "gold_non_null": sum(v is not None for v in gold.values()), "systems": res})


# ---------------------------------------------------------------- E4 blank form
def e4_blank(sys_facts, both, prox):
    def blank(fs): return [f for f in fs if f.doc_id == "employee_app_blank"]
    rows = {}
    for name, cand, ver in (("rules (typed)", sys_facts["rules, union of readings (no verifier)"], sys_facts["rules + verifier (pipeline)"]),
                            ("proximity (naive)", sys_facts["proximity baseline (no verifier)"], sys_facts["proximity + verifier"])):
        rows[name] = {"values_emitted_on_blank_form": len(blank(cand)), "loaded_after_verifier": len(blank(ver))}
    for k in [k for k in sys_facts if k.endswith("(no verifier)") and (k.startswith("CRF") or k.startswith("LLM"))]:
        base = k.replace(" (no verifier)", "").split(",")[0]
        ver = sys_facts[k.replace(", trained on synthetic dev forms", "").replace(" (no verifier)", " + verifier")]
        rows[base] = {"values_emitted_on_blank_form": len(blank(sys_facts[k])), "loaded_after_verifier": len(blank(ver)),
                      "examples": [f"{f.field}={f.value!r}" for f in blank(sys_facts[k])][:6]}
    rej = Counter(f.reason.split(":")[0] + ":" + f.reason.split(":")[1].split(",")[0] for f in blank(prox) if f.status == "quarantined")
    examples = [{"field": f.field, "value": f.value, "status": f.status, "reason": f.reason} for f in blank(prox)][:8]
    return dump("e4_blank_form.json", {"note": "Every value extracted from the blank application is false by construction.",
                                       "rows": rows, "proximity_rejection_reasons": dict(rej), "examples": examples,
                                       "llm": "included" if any(k.startswith("LLM") for k in rows) else "pending: run python -m veridoc.fields_llm"})


# ---------------------------------------------------------------- E5 verifier stress
def perturb(value, rng):
    digits = [i for i, c in enumerate(value) if c.isdigit()]
    if digits:
        i = rng.choice(digits)
        return value[:i] + rng.choice([d for d in "0123456789" if d != value[i]]) + value[i + 1:]
    letters = [i for i, c in enumerate(value) if c.isalpha()]
    i = rng.choice(letters)
    return value[:i] + rng.choice([c for c in "aeiourstn" if c != value[i].lower()]) + value[i + 1:]


def e5_stress(both, texts, primary, alt, thresholds=(80, 85, 90, 95, 98, 100), seed=573):
    rng = random.Random(seed)
    true = [f for f in both if f.status in LOADED]
    by_page = defaultdict(list)
    for f in true:
        by_page[(f.doc_id, f.page)].append(f)
    fab, mis = [], []
    for f in true:
        page_text = " ".join(texts[(f.doc_id, f.page)])
        for _ in range(20):                                  # near-miss fabrication: one character changed, not on the page
            v = perturb(f.value, rng)
            if v not in page_text:
                g = copy.deepcopy(f); g.value = v; g.span = f.span.replace(f.value, v); fab.append(g); break
        others = [o for o in by_page[(f.doc_id, f.page)] if o.field != f.field and o.value != f.value
                  and fields.FIELD_TYPES[o.field] == fields.FIELD_TYPES[f.field]]
        if others:                                           # misattribution: a real value from the same page, wrong field
            o = rng.choice(others)
            g = copy.deepcopy(f); g.value = o.value; g.span = f.span.replace(f.value, o.value); mis.append(g)
    # true facts under real OCR variation: span from reading 1 checked against reading 2 only
    ocr_true = [f for p in primary if p.route == "ocr" for f in fields.extract_rules(p)]
    rows = []
    for mode in ("span fuzzy", "span fuzzy + value exact on page"):
        for t in thresholds:
            def rejected(f, page_texts):
                g = copy.deepcopy(f)
                verifier.check_span(g, page_texts, threshold=t, value_exact=mode.endswith("page"))
                return g.checks["span"]["result"] == "fail"
            d_fab = sum(rejected(f, texts[(f.doc_id, f.page)]) for f in fab)
            d_mis = sum(rejected(f, texts[(f.doc_id, f.page)]) for f in mis)
            fr = sum(rejected(f, [alt[(f.doc_id, f.page)].text]) for f in ocr_true)
            rows.append({"mode": mode, "threshold": t,
                         "fabrication_detected": round(d_fab / len(fab), 3), "misattribution_detected": round(d_mis / len(mis), 3) if mis else None,
                         "false_rejection_under_ocr_variation": round(fr / len(ocr_true), 3)})
    return dump("e5_verifier_stress.json", {"n_true_facts": len(true), "n_fabricated": len(fab), "n_misattributed": len(mis),
                                            "n_ocr_facts_for_false_rejection": len(ocr_true), "seed": seed, "rows": rows,
                                            "note": "Synthetic corruptions of verified facts; span check only. NLI not included (not yet run)."})


# ---------------------------------------------------------------- E6 pipeline outcome
def e6_outcome(both):
    gold = load_gold()
    q = [f for f in both if f.status == "quarantined"]
    def truth(f):
        g = gold.get((f.doc_id, f.field), "?")
        return "correct value (false rejection)" if g == f.normalized else "wrong value (true rejection)"
    cols, row, src = deduction.fill(both)
    expect = {"Group Name": "Quest", "EE Effective Date": "August 1, 2022", "Payroll Frequency": "Semi-Monthly",
              "Payroll Deduction Amount": "$5.70", "EE City": "SALEM", "EE State": "OR", "EE Zip Code": "97305"}
    ok = sum(row[c] == v for c, v in expect.items()); wrong = sum(bool(row[c]) and row[c] != expect.get(c) for c in cols)
    return dump("e6_pipeline_outcome.json", {
        "candidates": len(both), "verified": sum(f.status == "verified" for f in both), "quarantined": len(q),
        "quarantine": [{"doc": f.doc_id, "page": f.page, "field": f.field, "value": f.value, "reason": f.reason, "truth": truth(f)} for f in q],
        "deduction_file": {"columns": len(cols), "filled_correct": ok, "filled_wrong": wrong, "left_empty": sum(not row[c] for c in cols),
                           "fillable_from_this_document_set": len(expect),
                           "note": "The other 15 columns are redacted or absent in the de-identified set, so empty is the correct output."}})


if __name__ == "__main__":
    e1 = e1_routing()
    sys_facts, both, prox, texts, primary, alt = systems()
    e3 = e3_fields(sys_facts); e4 = e4_blank(sys_facts, both, prox); e5 = e5_stress(both, texts, primary, alt); e6 = e6_outcome(both)
    print("E1", e1["total"])
    for k, v in e3["systems"].items():
        print("E3", f"{k:42s}", v["all"])
    print("E4", e4["rows"], e4["proximity_rejection_reasons"])
    for r in e5["rows"]:
        print("E5", r)
    print("E6", {k: v for k, v in e6.items() if k != "quarantine"}); [print("   ", x) for x in e6["quarantine"]]
