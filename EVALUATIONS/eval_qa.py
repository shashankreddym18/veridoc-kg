"""E9: retrieval tiers and graph question answering on the draft question set (G-D).

Run from the repo root:  python EVALUATIONS/eval_qa.py                  -> EVALUATIONS/results/e9_qa.json
                         python EVALUATIONS/eval_qa.py --set paraphrased -> e9_qa_paraphrased.json
The paraphrased set is the same questions reworded by a local LLM (veridoc/qa_paraphrase.py), i.e. wording
the field descriptions were not written against. If qa_llm.py has been run, its answers are scored too.
The graph tier is scored once per way of matching a question to a field: keywords (bm25), embeddings
(dense, if sentence-transformers is installed) and the LLM router (if qa_router.py has been run).

Evidence recall@k: all evidence pages of the question are in the top k pages.
Answer metrics: correct / wrong / abstained, by question type. A wrong answer is the costly
outcome (an unsupported value given to a user); abstaining on an unanswerable question is correct.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from veridoc import config
from veridoc.qa import QA

GOLD = config.GOLD / "qa_gold_draft.json"
LLM = config.PROCESSED / "qa_llm_answers.json"
ROUTES = config.PROCESSED / "qa_routes.json"
OUT = config.RESULTS / "e9_qa.json"


def recall_at(pages, ev, k):
    top = set(map(tuple, pages[:k]))
    return all(tuple(e) in top for e in ev)


def outcome(ans, gold):
    if ans is None:
        return "abstained_correct" if gold is None else "abstained"
    if gold is None:
        return "wrong"
    return "correct" if str(ans).strip().lower() == str(gold).strip().lower() else "wrong"


def main(which="draft"):
    global GOLD, LLM, OUT, ROUTES
    if which != "draft":
        GOLD = config.GOLD / f"qa_{which}.json"
        LLM = config.PROCESSED / f"qa_llm_answers_{which}.json"
        ROUTES = config.PROCESSED / f"qa_routes_{which}.json"
        OUT = config.RESULTS / f"e9_qa_{which}.json"
        if not GOLD.exists():
            sys.exit(f"{GOLD} not found: run  cd CODE && python -m veridoc.qa_paraphrase")
    qs = json.loads(GOLD.read_text())["questions"]
    qa = QA()
    answerable = [q for q in qs if q["answer"] is not None]
    tiers = {"bm25": lambda q: qa.bm25_pages(q["question"], 10)}
    if qa.dense_pages("test", 1) is not None:
        tiers["dense"] = lambda q: qa.dense_pages(q["question"], 10)
    tiers["hybrid (graph + bm25)"] = lambda q: qa.hybrid(q["question"], 10)["pages"]
    retrieval = {}
    for name, fn in tiers.items():
        got = {q["id"]: fn(q) for q in answerable}
        retrieval[name] = {f"recall@{k}": round(sum(recall_at(got[q["id"]], q["evidence"], k) for q in answerable) / len(answerable), 3)
                           for k in (1, 3, 5)}
    rows, per_q = defaultdict(lambda: defaultdict(int)), []
    for q in qs:
        g = qa.graph_answer(q["question"])
        o = outcome(g["answer"], q["answer"])
        rows[q["type"]][o] += 1
        rows["all"][o] += 1
        grounded = bool(g["evidence"]) and any(tuple(e) in set(map(tuple, q["evidence"])) for e in g["evidence"])
        per_q.append({"id": q["id"], "type": q["type"], "question": q["question"], "gold": q["answer"],
                      "graph_field": g["field"], "graph_answer": g["answer"], "outcome": o,
                      "evidence_matches_gold_page": grounded if g["answer"] is not None else None, "why": g["why"]})
    out = {"questions": len(qs), "by_type": {t: len([q for q in qs if q["type"] == t]) for t in ("single", "multi", "unanswerable")},
           "retrieval": retrieval, "graph_answers": {k: dict(v) for k, v in rows.items()}, "per_question": per_q,
           "set": which, "note": "Draft question set written by one person; the field descriptions were written from the form labels, not the questions."}
    matchers = {}
    if qa.field_dense("test") is not None:
        matchers["dense"] = lambda q: qa.graph_answer(q["question"], "dense")
    if ROUTES.exists():
        routes = json.loads(ROUTES.read_text())["routes"]
        matchers["llm router"] = lambda q: qa.graph_answer(q["question"], field=routes.get(q["id"]))
    out["graph_answers_by_matcher"] = {"bm25": out["graph_answers"]}
    for name, fn in matchers.items():
        mrows, fields = defaultdict(lambda: defaultdict(int)), {}
        for q in qs:
            g = fn(q)
            o = outcome(g["answer"], q["answer"])
            mrows[q["type"]][o] += 1; mrows["all"][o] += 1
            fields[q["id"]] = {"field": g["field"], "answer": g["answer"], "outcome": o}
        out["graph_answers_by_matcher"][name] = {k: dict(v) for k, v in mrows.items()}
        out.setdefault("per_question_by_matcher", {})[name] = fields
    if LLM.exists():
        llm = json.loads(LLM.read_text())
        lrows = defaultdict(lambda: defaultdict(int))
        for q in qs:
            a = llm.get(q["id"], {}).get("answer")
            o = outcome(a, q["answer"])
            lrows[q["type"]][o] += 1; lrows["all"][o] += 1
        out["llm_answers"] = {k: dict(v) for k, v in lrows.items()}
    OUT.write_text(json.dumps(out, indent=1))
    print("retrieval", json.dumps(retrieval))
    print("graph answers", json.dumps(out["graph_answers"]))
    for r in per_q:
        if r["outcome"] not in ("correct", "abstained_correct"):
            print("  ", r["id"], r["outcome"], r["graph_field"], r["graph_answer"], "| gold:", r["gold"], "|", r["why"])
    for name, r in out["graph_answers_by_matcher"].items():
        if name != "bm25":
            print(f"graph answers, {name}", json.dumps(r))
            for qid, v in out["per_question_by_matcher"][name].items():
                if v["outcome"] not in ("correct", "abstained_correct"):
                    print("  ", qid, v["outcome"], v["field"], v["answer"])
    if "llm_answers" in out:
        print("llm answers", json.dumps(out["llm_answers"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="draft", help="draft (default) or paraphrased")
    main(ap.parse_args().set)
