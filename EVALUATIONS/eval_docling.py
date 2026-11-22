"""E2b: Tesseract (two readings) against Docling on the scanned pages.

Needs DATA/processed/pages_docling/ (python -m veridoc.docling_extract).
Run from the repo root:  python EVALUATIONS/eval_docling.py   -> EVALUATIONS/results/e2b_docling.json

Two comparisons:
  text     against hand-checked transcriptions, for the pages that have one in
           DATA/gold/text/<doc_id>_p<NNN>.txt (draft gold, see DATA/README.md):
           character and word error rate, which also count a different reading order as errors, and
           word recall and precision with order ignored (which words of the page did the engine get)
  fields   the same typed rules run on each engine's text, scored against the field labels of the
           scanned documents. The rules were written on Tesseract's line layout, so this favours Tesseract;
           it shows how much a downstream extractor depends on the text engine's reading order.
"""
import json, re, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jiwer
from veridoc import config, extract, fields
from run_evals import load_gold, score

GOLD_TEXT = config.GOLD / "text"
SCANNED = ("master_app", "certificate")


def norm(t):
    t = t.translate(str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"}))
    return re.sub(r"\s+", " ", t.lower()).strip()


def bag(ref: str, hyp: str) -> tuple:
    """(words in common, reference words, engine words), order ignored."""
    r, h = Counter(ref.split()), Counter(hyp.split())
    return sum((r & h).values()), sum(r.values()), sum(h.values())


def main():
    engines = {"tesseract reading 1": config.PAGES, "tesseract reading 2": config.PROCESSED / "pages_alt",
               "docling": config.PROCESSED / "pages_docling"}
    pages = {name: {(p.doc_id, p.page): p for p in extract.load_pages(in_dir=d)} for name, d in engines.items() if d.exists()}
    if not pages.get("docling"):
        pages.pop("docling", None)
        print("No Docling pages yet (cd CODE && python -m veridoc.docling_extract --docs master_app certificate): Tesseract only.")
    out = {"engines": list(pages), "text": {}, "fields": {}, "pages_per_engine": {k: len(v) for k, v in pages.items()}}
    files = sorted(GOLD_TEXT.glob("*_p*.txt"))
    for name, pg in pages.items():
        refs, hyps, per, hit, nr, nh = [], [], {}, 0, 0, 0
        for f in files:
            doc, n = f.stem.rsplit("_p", 1)
            p = pg.get((doc, int(n)))
            if p is None:
                continue
            r, h = norm(f.read_text(encoding="utf-8")), norm(p.text)
            refs.append(r); hyps.append(h); per[f.stem] = round(jiwer.cer(r, h), 3)
            a, b, c = bag(r, h); hit += a; nr += b; nh += c
        if refs:
            out["text"][name] = {"pages": len(refs), "cer": round(jiwer.cer(refs, hyps), 3), "wer": round(jiwer.wer(refs, hyps), 3),
                                 "word_recall_any_order": round(hit / nr, 3), "word_precision_any_order": round(hit / max(1, nh), 3),
                                 "cer_by_page": per}
    gold = {k: v for k, v in load_gold().items() if k[0] in SCANNED}
    for name, pg in pages.items():
        facts = [f for (doc, _), p in sorted(pg.items()) if doc in SCANNED for f in fields.extract_rules(p)]
        out["fields"][name] = score(facts, gold)
        secs = [p.seconds for (doc, _), p in pg.items() if doc in SCANNED]
        out["fields"][name]["sec_per_page"] = round(sum(secs) / len(secs), 2) if secs else None
    (config.RESULTS / "e2b_docling.json").write_text(json.dumps(out, indent=1))
    print("gold text pages:", len(files))
    for name, r in out["text"].items():
        print(f"   text   {name:22s} CER {r['cer']:.3f}  WER {r['wer']:.3f}  words found {r['word_recall_any_order']:.3f}  "
              f"words correct {r['word_precision_any_order']:.3f}  ({r['pages']} pages)")
    for name, r in out["fields"].items():
        print(f"   fields {name:22s} emitted {r['emitted']:3d} correct {r['correct']:3d} P {r['precision']:.2f} R {r['recall']:.2f}  {r['sec_per_page']} s/page")


if __name__ == "__main__":
    main()
