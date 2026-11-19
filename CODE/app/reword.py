"""Write the question set in your own words (a person, not a model), in the terminal.

    python CODE/app/reword.py            from the repo root; answers are saved after every question

For each question, type how you would ask a colleague for the same piece of information, then Enter.
Use your own words: avoid the label printed on the form where you can ("face amount", "situs state").
Press Enter on an empty line to skip a question; type q to stop (you can run it again to continue).
Writes DATA/gold/qa_human.json; score it with  python EVALUATIONS/eval_qa.py --set human
(run_laptop.sh does this when the file exists).

Why a person: a local 8B model either changed the meaning of 5 of 31 questions (date of birth became
"age at application") or, with a same-meaning check added, rejected every rewrite (see qa_paraphrase.py).
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from veridoc import config

SRC = config.GOLD / "qa_gold_draft.json"
DST = config.GOLD / "qa_human.json"


def main():
    src = json.loads(SRC.read_text())["questions"]
    done = {q["id"]: q for q in json.loads(DST.read_text())["questions"]} if DST.exists() else {}
    todo = [q for q in src if q["id"] not in done]
    print(f"{len(done)} of {len(src)} done.\n"
          "Rewrite each QUESTION in your own words. Do not answer it.\n"
          "  example:  What is the certificate date?  ->  On what date did this employee's coverage start?\n"
          "Empty Enter = skip, q = stop.\n")
    for n, q in enumerate(todo, len(done) + 1):
        print(f"[{n}/{len(src)}] {q['question']}")
        while True:
            new = input("   same question, your words: ").strip()
            if new.lower() == "q" or not new or (len(new.split()) >= 4 and new.endswith("?")):
                break
            print("   that looks like an answer: write a question of 4+ words ending in '?'")
        if new.lower() == "q":
            break
        if not new or new.lower() == q["question"].lower():
            continue
        done[q["id"]] = {**q, "question": new, "original": q["question"], "reworded": True}
        out = [done[x["id"]] for x in src if x["id"] in done]
        DST.write_text(json.dumps({"_status": "Questions of qa_gold_draft.json reworded by a team member (not the person who "
                                   "wrote them). Answers and evidence pages are unchanged.",
                                   "questions": out}, indent=1))
    print(f"\n{len(done)} of {len(src)} questions saved in your words -> {DST}")


if __name__ == "__main__":
    main()
