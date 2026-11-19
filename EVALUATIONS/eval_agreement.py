"""E11: agreement between the two annotation passes over the field labels.

Run from the repo root after the second pass (python CODE/app/annotate.py):
    python EVALUATIONS/eval_agreement.py      -> EVALUATIONS/results/e11_label_agreement.json

Reported: agreement on whether the field has a value at all (with Cohen's kappa), exact agreement on the
normalized value where both passes gave one, and every disagreement for adjudication.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from veridoc import config

FIRST = config.GOLD / "fields_gold_draft.json"
SECOND = config.GOLD / "fields_gold_second.json"


def kappa(pairs):
    """Cohen's kappa for two raters over binary labels."""
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def main():
    if not SECOND.exists():
        sys.exit("No second pass yet: run  python CODE/app/annotate.py")
    first = {(d, f): v for d, f, v in json.loads(FIRST.read_text())["items"]}
    second = {(v["doc"], v["field"]): v["normalized"] for v in json.loads(SECOND.read_text()).values()}
    both = [k for k in first if k in second]
    present = [(first[k] is not None, second[k] is not None) for k in both]
    valued = [k for k in both if first[k] is not None and second[k] is not None]
    same = [k for k in valued if str(first[k]).lower() == str(second[k]).lower()]
    disagreements = [{"doc": k[0], "field": k[1], "first": first[k], "second": second[k]} for k in both
                     if (first[k] is None) != (second[k] is None) or (first[k] is not None and str(first[k]).lower() != str(second[k]).lower())]
    out = {"items_in_first_pass": len(first), "items_labeled_by_both": len(both),
           "has_value_agreement": round(sum(a == b for a, b in present) / len(present), 3), "has_value_kappa": kappa(present),
           "both_gave_a_value": len(valued), "same_value": len(same),
           "value_agreement": round(len(same) / len(valued), 3) if valued else None,
           "overall_agreement": round(1 - len(disagreements) / len(both), 3), "disagreements": disagreements}
    (config.RESULTS / "e11_label_agreement.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "disagreements"}, indent=1))
    for d in disagreements:
        print("   disagree:", d)


if __name__ == "__main__":
    main()
