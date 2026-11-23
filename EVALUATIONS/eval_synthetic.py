"""E7: held-out test on synthetic filled forms (gold set G-C).

Run from the repo root (after generating the forms, see veridoc/synthetic.py):
    python EVALUATIONS/eval_synthetic.py --set dev     # forms used to fix rules and train the CRF
    python EVALUATIONS/eval_synthetic.py --set test    # held-out forms, generated after the rules were frozen
Pages are extracted once and cached. Writes EVALUATIONS/results/e7_synthetic_<set>.json.

Every extractor runs on every form in three conditions (native text layer, 300 dpi scan,
150 dpi degraded scan); each form is verified on its own. Scoring is the same as E3:
distinct (form, field, value), with values on empty fields counted separately.
"""
import argparse, json, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from veridoc import config, extract, fields, verifier, pipeline
from veridoc.kg import LOADED
from veridoc import synthetic
from veridoc.synthetic import paths, CONDITIONS
from veridoc.schema import load_facts

SET = "dev"
# The generator prints the payroll frequency over "Modal" with a white box. In the native
# PDF the old word stays in the text layer under the box, so both texts come out interleaved
# ("YouYr oMuor nMtohdlya"). That is an artifact of the generator, so these two fields are
# not scored on native forms (they are scored on the scans, where only the visible text exists).
NATIVE_UNSCORED = {"payroll_frequency", "payroll_deduction_amount"}


def P():
    root, forms, gold = paths(SET)
    return {"root": root, "forms": forms, "gold": gold, "pages": root / "pages", "images": root / "images",
            "result": config.RESULTS / f"e7_synthetic_{SET}.json"}


def _process(args):
    pdf, doc_id, pages_dir, images_dir = args
    for variant in ("primary", "alt"):
        d = pages_dir / variant
        if not list(d.glob(f"{doc_id}_p*.json")):
            pages = extract.process_pdf(pdf, doc_id, variant=variant, image_dir=images_dir)
            extract.save_pages(pages, d)
    return doc_id


def extract_all(workers=4):
    p = P()
    jobs = sorted((f, f.stem, p["pages"], p["images"]) for f in p["forms"].glob("syn_*.pdf"))
    with Pool(workers) as pool:
        for i, d in enumerate(pool.imap_unordered(_process, jobs), 1):
            print(f"\r  pages extracted for {i}/{len(jobs)} forms", end="", flush=True)
    print()


def load():
    p = P()
    gold_doc = json.loads(p["gold"].read_text())
    gold = {}
    for g in gold_doc["forms"]:
        for cond in CONDITIONS:
            for field, v in g["values"].items():
                if cond == "native" and field in NATIVE_UNSCORED:
                    continue
                gold[(f"{g['form']}_{cond}", field)] = None if v is None else fields.norm_value(fields.FIELD_TYPES[field], v)
    primary = extract.load_pages(in_dir=p["pages"] / "primary")
    alt = {(x.doc_id, x.page): x for x in extract.load_pages(in_dir=p["pages"] / "alt")}
    return gold, primary, alt


def extractors():
    ex = {"proximity (naive)": fields.extract_proximity, "rules (typed)": fields.extract_rules}
    try:
        from veridoc import crf
        if crf.MODEL.exists():
            ex["CRF"] = crf.extract_crf
    except ImportError:
        pass
    return ex


def llm_facts():
    """LLM facts on the synthetic forms, if fields_llm was run on them (see README)."""
    f = P()["root"] / "facts_llm.json"
    return load_facts(f) if f.exists() else None


def verify_per_doc(facts, primary, alt):
    texts = {(p.doc_id, p.page): [p.text] + ([alt[(p.doc_id, p.page)].text] if (p.doc_id, p.page) in alt and p.route == "ocr" else [])
             for p in primary}
    by = defaultdict(list)
    for f in facts:
        by[f.doc_id].append(f)
    for fs in by.values():
        verifier.verify(fs, texts, nli=False)
    return facts


def score(facts, gold):
    emitted = {(f.doc_id, f.field, f.normalized or f.value.lower()) for f in facts if (f.doc_id, f.field) in gold}
    correct = {e for e in emitted if gold[(e[0], e[1])] == e[2]}
    pos = {k for k, v in gold.items() if v is not None}
    found = {(e[0], e[1]) for e in correct}
    on_null = [e for e in emitted if gold[(e[0], e[1])] is None]
    p = len(correct) / len(emitted) if emitted else 1.0
    r = len(found) / len(pos) if pos else 1.0
    return {"emitted": len(emitted), "correct": len(correct), "wrong": len(emitted) - len(correct), "gold_values": len(pos),
            "empty_fields": len(gold) - len(pos), "precision": round(p, 3), "recall": round(r, 3),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0, "values_on_empty_fields": len(on_null)}


def main():
    synthetic.ensure(SET)
    extract_all()
    gold, primary, alt = load()
    if not primary:
        sys.exit(f"No pages for the synthetic {SET} set: generation or extraction failed.")
    systems = {}
    for name, fn in extractors().items():
        cands = pipeline.candidates(primary, alt, fn)
        verified = verify_per_doc([f for f in cands], primary, alt)
        systems[f"{name} + verifier"] = [f for f in verified if f.status in LOADED]
        systems[name] = cands                       # every candidate, before the verifier
    lf = llm_facts()
    if lf is not None:
        systems["LLM"] = lf
        systems["LLM + verifier"] = [f for f in verify_per_doc(lf, primary, alt) if f.status in LOADED]
    rows, per_field = {}, {}
    for name, fs in systems.items():
        rows[name] = {cond: score([f for f in fs if f.doc_id.endswith(cond)], {k: v for k, v in gold.items() if k[0].endswith(cond)})
                      for cond in CONDITIONS}
        rows[name]["all"] = score(fs, gold)
        per_field[name] = {fld: score([f for f in fs if f.field == fld], {k: v for k, v in gold.items() if k[1] == fld})["f1"]
                           for fld in sorted({k[1] for k in gold})}
    out = {"set": SET, "note": "Synthetic forms with known answers; see veridoc/synthetic.py for which set is held out.",
           "forms": len({k[0].rsplit('_', 1)[0] for k in gold}), "conditions": list(CONDITIONS),
           "systems": rows, "f1_by_field": per_field}
    P()["result"].write_text(json.dumps(out, indent=1))
    for name, r in rows.items():
        print(f"{name:28s}", {c: (r[c]['precision'], r[c]['recall'], r[c]['values_on_empty_fields']) for c in (*CONDITIONS, 'all')})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["dev", "test"], default="dev")
    SET = ap.parse_args().set
    main()
