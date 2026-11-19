"""End-to-end run: pages -> candidate facts -> verifier -> graph + deduction file."""
from __future__ import annotations
import json
from collections import Counter
from . import config, extract, fields, verifier, kg, deduction
from .schema import save_facts, load_facts, FAIL

ALT_PAGES = config.PROCESSED / "pages_alt"


def load_readings():
    primary = extract.load_pages()
    alt = {(p.doc_id, p.page): p for p in extract.load_pages(in_dir=ALT_PAGES)}
    texts = {(p.doc_id, p.page): [p.text] + ([alt[(p.doc_id, p.page)].text] if (p.doc_id, p.page) in alt else []) for p in primary}
    return primary, alt, texts


def candidates(primary, alt, extractor=fields.extract_rules, policy=None) -> list:
    """Candidates from both OCR readings, merged with the agreement check."""
    ocr_pages = {(p.doc_id, p.page) for p in primary if p.route == "ocr"}
    c1 = [f for p in primary for f in extractor(p)]
    c2 = [f for k, p in alt.items() if k in ocr_pages for f in extractor(p)]
    texts = {(p.doc_id, p.page): [p.text, alt[(p.doc_id, p.page)].text] for p in primary if (p.doc_id, p.page) in alt}
    return verifier.merge_readings(c1, c2, ocr_pages, texts, policy)


def nli_note(facts) -> str:
    ran = [f for f in facts if f.checks.get("nli", {}).get("result") in ("pass", "fail")]
    if not ran:
        return "skipped (model not installed)" if verifier.NLI_MODE != "off" else "off"
    low = sum(f.checks["nli"]["result"] == "fail" for f in ran)
    return (f"{verifier.NLI_MODE}: {low} of {len(ran)} facts below the entailment threshold"
            + ("" if verifier.NLI_MODE == "hard" else " (recorded, not used to quarantine)"))


def run(use=None, save=True, nli=True) -> dict:
    primary, alt, texts = load_readings()
    facts = candidates(primary, alt)
    warnings = verifier.verify(facts, texts, use=use, nli=nli)
    reviewed = verifier.apply_review(facts)
    summary = {
        "pages": len(primary), "ocr_pages": sum(p.route == "ocr" for p in primary),
        "candidates": len(facts), "status": dict(Counter(f.status for f in facts)),
        "quarantine_reasons": dict(Counter(f.reason.split(":")[0] for f in facts if f.reason)),
        "check_failures": {c: sum(f.checks.get(c, {}).get("result") == FAIL for f in facts) for c in verifier.CHECKS},
        "nli": nli_note(facts),
        "reviewed": reviewed, "warnings": warnings,
    }
    if save:
        config.FACTS.mkdir(parents=True, exist_ok=True)
        save_facts(config.FACTS / "facts.json", facts)
        summary["kg"] = kg.save(facts)
        summary["deduction"] = deduction.save(facts)
        (config.PROCESSED / "run_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
