"""LLM reader on top of the retrieval tiers (needs Ollama, like fields_llm.py).

    cd CODE && python -m veridoc.qa_llm --model llama3.1:8b
writes DATA/processed/qa_llm_answers.json; EVALUATIONS/eval_qa.py then scores it.

Context = every verified graph fact (with its document and page) + the hybrid tier's top pages.
The model must answer from the context only and return null when the context does not say.
"""
from __future__ import annotations
import argparse, json, re

from . import config
from .fields import norm_date, norm_money, norm_value
from .fields_llm import cached_call
from .qa import QA

PROMPT = """Answer the question about a group life insurance case using ONLY the context.
Return JSON only: {{"answer": <short answer or null>, "evidence": [["doc_id", page], ...]}}
- Give dates as YYYY-MM-DD, money as a plain number with two decimals (e.g. 136.80), yes/no questions as "yes" or "no".
- If the context does not contain the answer, or the value is redacted or blank, answer null. Never guess.

VERIFIED FACTS (field = value, document, page):
{facts}

PAGES:
{pages}

QUESTION: {question}"""


def normalize(a):
    """Bring a free-text answer to the normalized form used in the gold file."""
    if a is None:
        return None
    a = str(a).strip().rstrip(".")
    if a.lower() in ("null", "none", "unknown", "not stated", ""):
        return None
    if norm_value("state", a):                       # "Oregon" and "OR" are the same answer
        return norm_value("state", a)
    for fn in (norm_date, norm_money):
        if re.search(r"\d", a) and fn(a) and not re.fullmatch(r"\d{1,3}", a):
            return fn(a)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:%|years?|years old)?", a)
    return m[1] if m else a.lower()


def main(model: str = "llama3.1:8b", k: int = 3, chars: int = 2500, which: str = "draft") -> None:
    qa = QA()
    facts = "\n".join(f"{f.field} = {f.value} ({f.doc_id}, p{f.page})" for fs in qa.graph.values() for f in fs)
    gold = config.GOLD / ("qa_gold_draft.json" if which == "draft" else f"qa_{which}.json")
    out_path = config.PROCESSED / ("qa_llm_answers.json" if which == "draft" else f"qa_llm_answers_{which}.json")
    qs = json.loads(gold.read_text())["questions"]
    call = cached_call(config.PROCESSED / "llm_cache" / model.replace(":", "_"))
    out = {}
    for i, q in enumerate(qs, 1):
        h = qa.hybrid(q["question"], k)
        pages = "\n\n".join(f"[{d} p{p}]\n{qa.page_text[(d, p)][:chars]}" for d, p in h["pages"][:k])
        raw = call(PROMPT.format(facts=facts, pages=pages, question=q["question"]), model)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"answer": None, "evidence": [], "error": "not JSON"}
        out[q["id"]] = {"raw": data.get("answer"), "answer": normalize(data.get("answer")), "evidence": data.get("evidence", [])}
        print(f"\r  {i}/{len(qs)}", end="", flush=True)
    print()
    out_path.write_text(json.dumps(out, indent=1))
    print("wrote", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--set", default="draft", help="draft (default) or paraphrased")
    a = ap.parse_args()
    main(a.model, which=a.set)
