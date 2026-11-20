"""Reword the question set with a local LLM (needs Ollama), to test wording the system was not built on.

    cd CODE && python -m veridoc.qa_paraphrase [--model llama3.1:8b]
writes DATA/gold/qa_paraphrased.json: same ids, answers and evidence pages, new wording.

A rewrite that changes the meaning makes its question unfair, not harder. The first version of this
script asked only for "everyday wording": the model turned "date of birth" into "age at the time of
application" and "certificate" into "group life insurance policy" (a different document in this case).
So the prompt now keeps the terms that name specific things, and a second call checks that both
wordings ask for the same thing. A rewrite that fails the check is retried, then dropped (the
original wording is kept and counted). The check is made by the same small model, so a person
should still read the pairs: DATA/gold/qa_paraphrased_review.md lists them side by side.
"""
from __future__ import annotations
import argparse, json

from . import config
from .fields_llm import cached_call

KEEP = ["certificate", "rider", "master application", "proposal", "situs state", "group policyholder",
        "face amount", "paid-up", "issue age", "insured", "employee"]
PROMPT = """Rewrite this question about a group life insurance case in different words, the way a colleague might ask it.
Rules:
- It must ask for exactly the same piece of information: the same answer must be correct for both questions.
- These terms name specific things in the case documents. If one appears in the question, keep it unchanged: {keep}.
- A certificate is not a policy and a rider is not a certificate. Do not swap one for another.
- Do not turn a date into an age, an amount into a rate, or a name into a description.
- Change the sentence structure and the other words. Do not answer the question.{attempt}
Return JSON only: {{"question": "<the rewritten question>"}}

QUESTION: {q}"""
CHECK = """Two questions about the same group life insurance case are given.
Do they ask for exactly the same piece of information, so that one answer is correct for both?
Answer "no" if B asks for a different kind of value (an age instead of a date, a policy instead of a certificate),
adds a condition, or could be answered correctly with something that would be wrong for A.
Return JSON only: {{"same": true or false, "difference": "<a few words, or empty>"}}

A: {a}
B: {b}"""
TRIES = 3


def reword(q: str, model: str, call) -> tuple:
    """(new wording or None, attempts used, notes on rejected attempts)."""
    notes = []
    for n in range(1, TRIES + 1):
        attempt = "" if n == 1 else f"\n- This is attempt {n}: an earlier rewrite changed the meaning. Stay closer to the original."
        try:
            new = (json.loads(call(PROMPT.format(keep=", ".join(KEEP), attempt=attempt, q=q), model)).get("question") or "").strip()
        except (json.JSONDecodeError, AttributeError):
            notes.append("unreadable output"); continue
        if not new or new.lower() == q.lower():
            notes.append("no change"); continue
        lost = [t for t in KEEP if t in q.lower() and t not in new.lower()]
        if lost:
            notes.append(f"dropped the term {lost[0]!r}: {new}"); continue
        try:
            verdict = json.loads(call(CHECK.format(a=q, b=new), model))
        except json.JSONDecodeError:
            verdict = {}
        if verdict.get("same") is True:
            return new, n, notes
        notes.append(f"check said different ({verdict.get('difference', '')}): {new}")
    return None, TRIES, notes


def main(model: str = "llama3.1:8b") -> None:
    src = json.loads((config.GOLD / "qa_gold_draft.json").read_text())
    call = cached_call(config.PROCESSED / "llm_cache" / model.replace(":", "_"))
    out, kept = [], 0
    for i, q in enumerate(src["questions"], 1):
        new, n, notes = reword(q["question"], model, call)
        kept += new is None
        out.append({**q, "question": new or q["question"], "original": q["question"], "reworded": new is not None,
                    "attempts": n, "rejected_attempts": notes})
        print(f"\r  {i}/{len(src['questions'])}", end="", flush=True)
    dst = config.GOLD / "qa_paraphrased.json"
    dst.write_text(json.dumps({"_status": f"Questions of qa_gold_draft.json reworded by {model}. Each rewrite keeps the case's own terms "
                               f"and passed a same-meaning check by the same model; {kept} of {len(out)} kept their original wording "
                               "because no rewrite passed. Not yet read by a person (see qa_paraphrased_review.md).",
                               "_format": src.get("_format", ""), "questions": out}, indent=1))
    lines = ["# Reworded questions: read each pair", "",
             "Does the reworded question ask for exactly the same thing as the original? If not, fix the wording in",
             "`qa_paraphrased.json` (field `question`) or set it back to the original, then re-run `python EVALUATIONS/eval_qa.py --set paraphrased`.", "",
             "| id | original | reworded | answer |", "|---|---|---|---|"]
    lines += [f"| {q['id']} | {q['original']} | {q['question'] if q['reworded'] else '(kept original)'} | {q['answer']} |" for q in out]
    (config.GOLD / "qa_paraphrased_review.md").write_text("\n".join(lines) + "\n")
    print(f"\n{len(out)} questions, {kept} kept their original wording -> {dst}")
    for q in out[:5]:
        print("  ", q["original"], "=>", q["question"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    main(ap.parse_args().model)
