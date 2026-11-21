"""E8: the agreement check's policy (the false-negative trade-off the TA asked about).

When only one of the two OCR readings yields a value:
  strict         quarantine it (default)
  corroborate    accept it if the value's text also appears in the other reading's page text
  conflict_only  accept it unless the other reading yielded a different value
Measured on the real documents (draft gold) and on the synthetic forms (dev and test, if extracted).
Run from the repo root:  python EVALUATIONS/eval_agree_policy.py  -> EVALUATIONS/results/e8_agree_policy.json
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from veridoc import config, fields, pipeline, verifier
from veridoc.kg import LOADED
from run_evals import load_gold, score
import eval_synthetic as syn

POLICIES = ("strict", "corroborate", "conflict_only")


def row(cands, gold, texts, per_doc=False, primary=None, alt=None):
    if per_doc:
        syn.verify_per_doc(cands, primary, alt)
    else:
        verifier.verify(cands, texts, nli=False)
    loaded = [f for f in cands if f.status in LOADED]
    s = score(loaded, gold)
    q = [f for f in cands if f.status == "quarantined" and (f.doc_id, f.field) in gold]
    s["quarantined"] = len(q)
    s["quarantined_but_correct"] = sum(gold[(f.doc_id, f.field)] == f.normalized for f in q)
    return s


def main():
    out = {}
    primary, alt, texts = pipeline.load_readings()
    gold = load_gold()
    out["real documents"] = {p: row(pipeline.candidates(primary, alt, policy=p), gold, texts) for p in POLICIES}
    for which in ("dev", "test"):
        syn.SET = which
        if not syn.P()["pages"].exists():
            continue
        g, pr, al = syn.load()
        for ex_name, ex in (("rules", fields.extract_rules),):
            out[f"synthetic {which}, {ex_name}"] = {p: row(pipeline.candidates(pr, al, ex, policy=p), g, None, True, pr, al) for p in POLICIES}
    (config.RESULTS / "e8_agree_policy.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k)
        for p, r in v.items():
            print(f"   {p:14s} P={r['precision']:.3f} R={r['recall']:.3f} wrong_loaded={r['wrong']} quarantined={r['quarantined']} (correct among them: {r['quarantined_but_correct']})")


if __name__ == "__main__":
    main()
