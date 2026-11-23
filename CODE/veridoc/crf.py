"""CRF baseline: token tagging (BIO) over page words, with layout features.

Trained on the synthetic dev forms (labels come from the generated values), scored on the
held-out synthetic test forms (E7) and on the real documents (E3/E4).

Train:  cd CODE && python -m veridoc.crf          (needs the dev forms extracted by eval_synthetic.py --set dev)
Use:    from veridoc.crf import extract_crf        (same interface as fields.extract_rules)
"""
from __future__ import annotations
import json, pickle, re
from pathlib import Path

from rapidfuzz import fuzz

from . import config, extract
from .fields import _word_offsets, _fact
from .synthetic import SCORED as SCORED_ON_FORM
from .schema import Page

MODEL = config.SYNTH / "crf.pkl"
FORM_MARK = "MassMutual@WORK"       # pages of the employee application carry this footer/header


def _shape(t: str) -> str:
    s = re.sub(r"[A-Z]", "X", t); s = re.sub(r"[a-z]", "x", s); s = re.sub(r"\d", "d", s)
    return re.sub(r"(.)\1{2,}", r"\1\1", s)


def _above(words, i, by_line):
    """Word on the nearest earlier line (up to 3 lines up) that overlaps token i horizontally."""
    w = words[i]
    for dl in (1, 2, 3):
        cands = [u for u in by_line.get(w.line - dl, []) if u.x0 <= w.x1 and u.x1 >= w.x0]
        if cands:
            return cands[0].text.lower().strip(":")
    return "<none>"


def features(page: Page) -> list:
    ws = page.words
    by_line = {}
    for w in ws:
        by_line.setdefault(w.line, []).append(w)
    out = []
    for i, w in enumerate(ws):
        t = w.text
        f = {"bias": 1.0, "low": t.lower(), "shape": _shape(t), "digit": t.isdigit(), "dollar": "$" in t,
             "slash": "/" in t, "title": t.istitle(), "upper": t.isupper(), "len": min(len(t), 10),
             "xb": int(10 * w.x0 / page.width), "yb": int(20 * w.y0 / page.height),
             "above": _above(ws, i, by_line), "first_on_line": by_line[w.line][0].text.lower()}
        for d in (-2, -1, 1, 2):
            j = i + d
            f[f"w{d}"] = ws[j].text.lower() if 0 <= j < len(ws) else "<pad>"
            f[f"sh{d}"] = _shape(ws[j].text) if 0 <= j < len(ws) else "<pad>"
        for u in by_line.get(w.line - 1, [])[:12]:
            f[f"prev_line:{u.text.lower().strip(':')}"] = 1.0
        out.append(f)
    return out


def _clean(s: str) -> str:
    return s.replace("$", "").replace(",", "").strip().lower()


def labels(page: Page, values: dict, positions: dict) -> list:
    """Weak BIO labels: for every known value, the window of words that best matches it,
    preferring the window closest to where the generator wrote it."""
    ws = page.words
    tags = ["O"] * len(ws)
    for field, value in values.items():
        if not value or positions.get(field, (None,))[0] != page.page - 1:
            continue
        n = len(value.split())
        _, x, top = positions[field]
        best = None
        for i in range(len(ws) - n + 1):
            window = " ".join(w.text for w in ws[i:i + n])
            score = fuzz.ratio(_clean(window), _clean(value))
            if score < 85:
                continue
            dist = abs(ws[i].x0 - x) + abs(ws[i].y1 - top)
            key = (-score, dist)
            if best is None or key < best[0]:
                best = (key, i)
        if best and all(t == "O" for t in tags[best[1]:best[1] + n]):
            i = best[1]
            tags[i] = f"B-{field}"
            for k in range(i + 1, i + n):
                tags[k] = f"I-{field}"
    return tags


def train(c1: float = 0.1, c2: float = 0.05, iters: int = 200) -> dict:
    import sklearn_crfsuite
    from .synthetic import paths, POS
    root, _, gold_path = paths("dev")
    gold = {g["form"]: g for g in json.loads(gold_path.read_text())["forms"]}
    X, Y = [], []
    for variant in ("primary", "alt"):
        for p in extract.load_pages(in_dir=root / "pages" / variant):
            g = gold[p.doc_id.rsplit("_", 1)[0]]
            vals = {f: v for f, v in g["values"].items() if v}
            if p.doc_id.endswith("_native"):
                vals = {f: v for f, v in vals.items() if f not in ("payroll_frequency", "payroll_deduction_amount")}
            X.append(features(p)); Y.append(labels(p, vals, POS))
    n_lab = sum(t.startswith("B-") for y in Y for t in y)
    if not X or not n_lab:
        MODEL.unlink(missing_ok=True)
        raise SystemExit("CRF: no labelled dev pages. Run EVALUATIONS/eval_synthetic.py --set dev first.")
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=iters, all_possible_transitions=True)
    crf.fit(X, Y)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    MODEL.write_bytes(pickle.dumps(crf))
    return {"pages": len(X), "labelled_values": n_lab, "model": str(MODEL)}


_crf = None


def extract_crf(page: Page, tag: str = "crf") -> list:
    global _crf
    if not (config.doc_type(page.doc_id) == "employee_app_blank" or FORM_MARK in page.text) or not page.words:
        return []
    if _crf is None:
        _crf = pickle.loads(MODEL.read_bytes())
    if not list(getattr(_crf, "classes_", None) or []):      # a model trained on nothing must not be asked to tag
        return []
    pred = _crf.predict_single(features(page))
    offsets = _word_offsets(page)
    facts, i = [], 0
    while i < len(pred):
        if pred[i].startswith("B-"):
            field = pred[i][2:]
            j = i + 1
            while j < len(pred) and pred[j] == f"I-{field}":
                j += 1
            a, b = offsets[i][0], offsets[j - 1][1]
            if field in SCORED_ON_FORM:
                facts.append(_fact(page, field, page.text[a:b], a, b, offsets, tag, len(facts)))
            i = j
        else:
            i += 1
    return facts


if __name__ == "__main__":
    print(json.dumps(train(), indent=1))
