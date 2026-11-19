"""Stage 3 (modern side): schema-constrained LLM field extraction.

STATUS: written and unit-tested with a fake model; not yet run against a real model
(the build machine had no model server). To run it on a laptop:

    ollama pull llama3.1:8b
    cd CODE
    python -m veridoc.fields_llm --docs real            # 72 pages -> DATA/processed/facts/facts_llm.json
    python -m veridoc.fields_llm --docs synthetic-test  # 120 pages -> DATA/synthetic/test/facts_llm.json
Then re-run EVALUATIONS/run_evals.py and EVALUATIONS/eval_synthetic.py --set test: both pick the
LLM facts up automatically. Responses are cached per page, so an interrupted run resumes.
The LLM reads only the first OCR reading of each page, so its agreement check is "skip".

The model receives one page of text and the allowed field list, and must
return strict JSON: for every field either null or {"value", "span"} where
span is copied verbatim from the page. Output is validated with Pydantic and
then goes through exactly the same verifier as the rule-based facts.
"""
from __future__ import annotations
import json, re, sys, urllib.request
from pydantic import BaseModel, ValidationError
from . import config, extract, verifier
from .fields import FIELD_TYPES, norm_value, _word_offsets, _bbox
from .schema import Fact, save_facts

OLLAMA_URL = "http://localhost:11434/api/chat"
PROMPT = """You extract insurance fields from ONE page of a document.
Return JSON only: an object with exactly these keys: {fields}.
For each key give null if the page does not state it, otherwise
{{"value": "<as written>", "span": "<the exact line from the page that contains the value>"}}.
Never guess. A blank form field, a label with nothing filled in, or a redacted value is null.

PAGE TEXT:
{text}"""


class Item(BaseModel):
    value: str
    span: str


def call_ollama(prompt: str, model: str) -> str:
    body = json.dumps({"model": model, "stream": False, "format": "json", "options": {"temperature": 0},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["message"]["content"]


def extract_llm(page, model: str = "llama3.1:8b", call=call_ollama) -> list:
    names = sorted(FIELD_TYPES)
    raw = call(PROMPT.format(fields=", ".join(names), text=page.text[:6000]), model)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    facts, offsets = [], _word_offsets(page)
    for field in names:
        item = data.get(field)
        if not item:
            continue
        try:
            it = Item(**item) if isinstance(item, dict) else None
        except ValidationError:
            it = None
        if it is None or not it.value.strip():
            continue
        m = re.search(re.escape(it.value.strip()), page.text)
        bbox = _bbox(offsets, m.start(), m.end()) if m else None
        facts.append(Fact(fact_id=f"{page.doc_id}:{page.page}:{field}:llm:{len(facts)}", doc_id=page.doc_id, page=page.page,
                          field=field, value=it.value.strip(), normalized=norm_value(FIELD_TYPES[field], it.value) or "",
                          span=it.span.strip(), bbox=bbox, extractor=f"llm:{model}"))
    return facts


def cached_call(cache_dir):
    """Wrap call_ollama with a per-prompt cache on disk (resume after interruption)."""
    import hashlib
    cache_dir.mkdir(parents=True, exist_ok=True)

    def call(prompt, model):
        f = cache_dir / (hashlib.sha1(f"{model}\n{prompt}".encode()).hexdigest()[:16] + ".json")
        if f.exists():
            return f.read_text()
        out = call_ollama(prompt, model)
        f.write_text(out)
        return out
    return call


def main(model: str = "llama3.1:8b", docs: str = "real", limit: int | None = None) -> None:
    import time
    from collections import defaultdict
    if docs == "real":
        from .pipeline import load_readings
        primary, alt, texts = load_readings()
        out = config.FACTS / "facts_llm.json"
    else:
        from .synthetic import paths
        root = paths(docs.split("-")[1])[0]
        primary = extract.load_pages(in_dir=root / "pages" / "primary")
        texts = {(p.doc_id, p.page): [p.text] for p in primary}
        out = root / "facts_llm.json"
    pages = primary[:limit] if limit else primary
    if limit:                                  # smoke test: never overwrite the full run's output
        out = out.with_name(out.stem + "_smoke.json")
    call = cached_call(config.PROCESSED / "llm_cache" / model.replace(":", "_"))
    facts, t0 = [], time.time()
    for i, p in enumerate(pages, 1):
        facts += extract_llm(p, model, call=call)
        print(f"\r  {i}/{len(pages)} pages, {len(facts)} facts, {time.time() - t0:.0f}s", end="", flush=True)
    print()
    by = defaultdict(list)
    for f in facts:
        by[f.doc_id if docs != "real" else "all"].append(f)
    for fs in by.values():                 # synthetic forms are verified one form at a time
        verifier.verify(fs, texts)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_facts(out, facts)
    print(len(facts), "LLM facts;", sum(f.status == "verified" for f in facts), "verified ->", out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--docs", choices=["real", "synthetic-dev", "synthetic-test"], default="real")
    ap.add_argument("--limit", type=int, default=None, help="only the first N pages (smoke test)")
    a = ap.parse_args()
    main(a.model, a.docs, a.limit)
