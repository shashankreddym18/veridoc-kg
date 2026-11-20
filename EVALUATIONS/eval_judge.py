"""E5c: an LLM second opinion on what the NLI check rejects.

Needs sentence-transformers and Ollama (see run_laptop.sh).
    python EVALUATIONS/eval_judge.py [--model llama3.1:8b]   -> EVALUATIONS/results/e5c_judge.json

Part 1, same three sets as E5b (true facts, one-character fabrications, same-page misattributions):
  NLI alone        reject when entailment < 0.5
  LLM judge alone  reject when the LLM says the evidence does not support the claim
  cascade          reject only when NLI rejects AND the LLM agrees (the LLM is asked only about NLI rejections)
Part 2, the values the LLM extractor produced on the real documents that passed every hard check:
  how many of the wrong ones each option would stop, and how many correct ones it would lose.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from veridoc import config, pipeline, verifier, judge
from veridoc.kg import LOADED
from veridoc.schema import load_facts
from run_evals import load_gold
from eval_nli import build_sets


def decisions(rows, texts, model):
    """rows: (source fact, claimed fact). Returns per row: nli_reject, judge_reject, cascade_reject."""
    p = verifier.nli_probs([(verifier.premise(s, texts), verifier.verbalize(c)) for s, c in rows])
    nli_rej = [bool(x < verifier.NLI_THRESHOLD) for x in p]
    jv = judge.judge([judge.pair(s, c, texts) for s, c in rows], model)
    judge_rej = [v is False for v in jv]                       # an unusable reply does not reject
    return nli_rej, judge_rej, [a and b for a, b in zip(nli_rej, judge_rej)], sum(v is None for v in jv)


def rate(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def main(model):
    if verifier.nli_probs([("a", "a")]) is None:
        sys.exit("NLI model unavailable: pip install sentence-transformers")
    primary, alt, texts = pipeline.load_readings()
    facts = pipeline.candidates(primary, alt)
    verifier.verify(facts, texts, nli=False)
    sets = build_sets([f for f in facts if f.status in LOADED])
    out = {"model": model, "nli": f"hypothesis={verifier.NLI_HYPOTHESIS}, premise={verifier.NLI_PREMISE}, threshold={verifier.NLI_THRESHOLD}",
           "n": {k: len(v) for k, v in sets.items()}, "part1": {}, "unusable_judge_replies": 0}
    dec = {}
    for name, rows in sets.items():
        *d, bad = decisions(rows, texts, model)
        dec[name] = d
        out["unusable_judge_replies"] += bad
    for i, option in enumerate(("NLI alone", "LLM judge alone", "cascade (NLI, then LLM on rejections)")):
        out["part1"][option] = {"true_facts_rejected": rate(dec["true"][i]), "fabricated_caught": rate(dec["fabricated"][i]),
                                "misattributed_caught": rate(dec["misattributed"][i]),
                                "llm_calls": 0 if i == 0 else (sum(map(len, sets.values())) if i == 1 else int(sum(sum(dec[k][0]) for k in dec)))}
    out["true_facts_still_rejected_by_cascade"] = [verifier.verbalize(c, "field") for (s, c), r in zip(sets["true"], dec["true"][2]) if r]

    llm_path = config.FACTS / "facts_llm.json"
    if llm_path.exists():
        gold = load_gold()
        lf = load_facts(llm_path)
        verifier.verify(lf, texts, nli=False)
        passed = [f for f in lf if f.status in LOADED and (f.doc_id, f.field) in gold]
        correct = [f for f in passed if gold[(f.doc_id, f.field)] == (f.normalized or f.value.lower())]
        wrong = [f for f in passed if f not in correct]
        part2 = {"llm_values_passing_hard_checks": {"correct": len(correct), "wrong": len(wrong)}}
        dc, dw = decisions([(f, f) for f in correct], texts, model)[:3], decisions([(f, f) for f in wrong], texts, model)[:3]
        for i, option in enumerate(("NLI alone", "LLM judge alone", "cascade (NLI, then LLM on rejections)")):
            part2[option] = {"wrong_values_stopped": sum(dw[i]), "correct_values_lost": sum(dc[i])}
        part2["wrong_values"] = [{"field": f.field, "value": f.value, "evidence": f.span[:80], "nli_rejects": bool(a), "judge_rejects": bool(b)}
                                 for f, a, b in zip(wrong, dw[0], dw[1])]
        out["part2"] = part2
    (config.RESULTS / "e5c_judge.json").write_text(json.dumps(out, indent=1))
    print("part 1 (n:", out["n"], ")")
    for k, v in out["part1"].items():
        print(f"   {k:40s} true rejected {v['true_facts_rejected']}  fabricated caught {v['fabricated_caught']}  misattributed caught {v['misattributed_caught']}  LLM calls {v['llm_calls']}")
    print("   still rejected by cascade:", out["true_facts_still_rejected_by_cascade"])
    if "part2" in out:
        print("part 2:", out["part2"]["llm_values_passing_hard_checks"])
        for k in list(out["part1"]):
            print(f"   {k:40s}", out["part2"][k])
        for w in out["part2"]["wrong_values"]:
            print("     ", w)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    main(ap.parse_args().model)
