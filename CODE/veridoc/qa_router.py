"""Route each question to an ontology field with a local LLM (needs Ollama).

    cd CODE && python -m veridoc.qa_router [--set paraphrased] [--model llama3.1:8b]
writes DATA/processed/qa_routes[_<set>].json  {question id: field name or null};
EVALUATIONS/eval_qa.py then scores the graph tier with these routes next to the keyword match.

The LLM only picks a field name from the ontology's property list (or "none"). The answer is the
verified value stored in the graph for that field, so a wrong route gives a wrong field, never an invented value.
"""
from __future__ import annotations
import argparse, json

from . import config
from .fields_llm import cached_call
from .qa import route_llm


def main(model: str = "llama3.1:8b", which: str = "draft") -> None:
    gold = config.GOLD / ("qa_gold_draft.json" if which == "draft" else f"qa_{which}.json")
    out = config.PROCESSED / ("qa_routes.json" if which == "draft" else f"qa_routes_{which}.json")
    call = cached_call(config.PROCESSED / "llm_cache" / model.replace(":", "_"))
    qs = json.loads(gold.read_text())["questions"]
    routes = {}
    for i, q in enumerate(qs, 1):
        routes[q["id"]] = route_llm(q["question"], model, call)
        print(f"\r  {i}/{len(qs)}", end="", flush=True)
    out.write_text(json.dumps({"model": model, "routes": routes}, indent=1))
    print(f"\n{sum(v is not None for v in routes.values())} of {len(routes)} questions routed to a field -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--set", default="draft", help="draft (default) or paraphrased")
    a = ap.parse_args()
    main(a.model, a.set)
