"""E5b: the NLI check measured on its own, before it is allowed to quarantine anything.

Question (from the proposal review): how many true facts does the NLI cross-encoder reject, and
does it catch the misattributed values that the span check misses?

Needs sentence-transformers (downloads cross-encoder/nli-deberta-v3-base, about 0.7 GB, once):
    pip install sentence-transformers
    python EVALUATIONS/eval_nli.py            -> EVALUATIONS/results/e5b_nli.json

Four ways of posing the test are compared:
  hypothesis  field    "The rider termination date is August 1, 2061."   (ontology field name)
              label    "The termination date is August 1, 2061."         (wording printed on the form)
  premise     line     the evidence line only
              context  document title + the two lines above + the evidence line
Hypothesis sets, built from the facts the other checks verified:
  true            the fact as extracted
  fabricated      one character of the value changed
  misattributed   the value of another field of the same type from the same page
"""
import copy, json, random, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from veridoc import config, fields, pipeline, verifier
from veridoc.kg import LOADED
from run_evals import perturb

THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)
COMBOS = [(h, p) for h in ("field", "label") for p in ("line", "context")]


def build_sets(true, seed=573):
    rng = random.Random(seed)
    by_page = defaultdict(list)
    for f in true:
        by_page[(f.doc_id, f.page)].append(f)
    sets = {"true": [], "fabricated": [], "misattributed": []}
    for f in true:
        sets["true"].append((f, f))                                   # (fact that supplies the premise, fact that is claimed)
        g = copy.deepcopy(f); g.value = perturb(f.value, rng)
        sets["fabricated"].append((f, g))
        others = [o for o in by_page[(f.doc_id, f.page)] if o.field != f.field and o.value != f.value
                  and fields.FIELD_TYPES[o.field] == fields.FIELD_TYPES[f.field]]
        if others:
            g = copy.deepcopy(f); g.value = rng.choice(others).value
            sets["misattributed"].append((f, g))
    return sets


def main():
    if verifier.nli_probs([("a", "a")]) is None:
        sys.exit("NLI model unavailable: pip install sentence-transformers (and allow the download).")
    primary, alt, texts = pipeline.load_readings()
    facts = pipeline.candidates(primary, alt)
    verifier.verify(facts, texts, nli=False)                           # every check except NLI
    true = [f for f in facts if f.status in LOADED]
    sets = build_sets(true)
    out = {"model": "cross-encoder/nli-deberta-v3-base", "n": {k: len(v) for k, v in sets.items()}, "combos": {}}
    for hyp, prem in COMBOS:
        probs = {name: verifier.nli_probs([(verifier.premise(src, texts, prem), verifier.verbalize(claim, hyp)) for src, claim in rows])
                 for name, rows in sets.items()}
        rows = [{"threshold": t,
                 "true_facts_rejected": round(float((probs["true"] < t).mean()), 3),
                 "fabricated_caught": round(float((probs["fabricated"] < t).mean()), 3),
                 "misattributed_caught": round(float((probs["misattributed"] < t).mean()), 3) if len(probs["misattributed"]) else None}
                for t in THRESHOLDS]
        rejected = [{"fact": f.fact_id, "hypothesis": verifier.verbalize(f, hyp), "premise": verifier.premise(f, texts, prem),
                     "p_entail": round(float(p), 3)} for (f, _), p in zip(sets["true"], probs["true"]) if p < 0.5]
        missed = [{"hypothesis": verifier.verbalize(c, hyp), "premise": verifier.premise(s, texts, prem), "p_entail": round(float(p), 3)}
                  for (s, c), p in zip(sets["misattributed"], probs["misattributed"]) if p >= 0.5]
        out["combos"][f"hypothesis={hyp}, premise={prem}"] = {"rows": rows, "true_facts_rejected_at_0.5": rejected,
                                                             "misattributed_accepted_at_0.5": missed}
    (config.RESULTS / "e5b_nli.json").write_text(json.dumps(out, indent=1))
    print("n:", out["n"])
    for name, c in out["combos"].items():
        print(name)
        for r in c["rows"]:
            print(f"   t={r['threshold']:.1f}  true rejected {r['true_facts_rejected']:.2f}   fabricated caught {r['fabricated_caught']:.2f}"
                  f"   misattributed caught {r['misattributed_caught']}")
        for r in c["true_facts_rejected_at_0.5"]:
            print(f"      rejected true: {r['hypothesis']}  (p={r['p_entail']})")


if __name__ == "__main__":
    main()
