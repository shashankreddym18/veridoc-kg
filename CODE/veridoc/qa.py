"""Question answering over the verified graph and the page text (retrieval tiers).

Tiers
  bm25        rank pages by BM25 over the extracted text (both OCR readings of a page are indexed)
  dense       rank pages by embedding similarity (needs sentence-transformers; skipped if missing)
  graph       map the question to one ontology field, return the verified value with its provenance,
              or abstain when the graph has no verified value. Three ways to pick the field:
                bm25   keyword match against the field descriptions (default, no model)
                dense  embedding similarity to the field descriptions (needs sentence-transformers)
                llm    a local LLM picks a field name from the ontology's property list, or "none"
                       (needs Ollama). The LLM never writes the value: the answer is still the
                       verified value in the graph, so it cannot invent one.
  hybrid      graph answer + the union of its provenance pages and the BM25 top pages as evidence

The graph tier answers single-field questions only; a multi-hop question needs a reader on top
(qa_llm.py, run with Ollama). Field descriptions use the labels printed on the forms, not the
wording of the questions.
"""
from __future__ import annotations
import math, re
from collections import Counter, defaultdict

from . import config, extract
from .kg import LOADED
from .schema import load_facts

TOK = re.compile(r"[a-z0-9]+")
STOP = set("a an the of on in is what which who when how does do did for to and or by with was were are be".split())


def tokens(s: str) -> list:
    return [t for t in TOK.findall(s.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs: list, k1: float = 1.5, b: float = 0.75):
        self.docs = [tokens(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = sum(map(len, self.docs)) / max(1, len(self.docs))
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, q: str) -> list:
        qt = tokens(q)
        out = []
        for tf, d in zip(self.tf, self.docs):
            s = 0.0
            for t in qt:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            out.append(s)
        return out


# Field descriptions: the label text printed on the documents plus the field name.
FIELD_DESC = {
    "face_amount": "face amount coverage", "certificate_date": "certificate date", "date_of_birth": "date of birth insured",
    "issue_age": "issue age insured", "maturity_date": "maturity date certificate matures", "risk_class": "risk class",
    "loan_interest_rate": "loan interest rate", "annual_premium": "annual certificate premium excluding riders",
    "rider_annual_premium": "rider annual premium accidental death benefit rider", "rider_date": "rider date",
    "rider_benefit_amount": "rider benefit amount", "rider_termination_date": "rider termination date terminate",
    "min_paidup_amount": "minimum paid-up life insurance amount", "premium_frequency": "initial premium payment frequency",
    "payroll_frequency": "payroll frequency your semi-monthly premium deduction", "payroll_deduction_amount": "payroll deduction amount your premium",
    "ee_city": "employee city residential address", "ee_state": "employee state residential address", "ee_zip": "employee zip code",
    "date_of_hire": "date of hire employee", "annual_salary": "annual salary compensation", "job_title": "job title occupation",
    "group_name": "group name group policyholder name", "group_number": "group number", "group_street": "group policyholder street address",
    "group_state": "group policyholder state", "situs_state": "situs state", "presented_by": "presented by proposal",
    "proposal_date": "proposal date", "eligible_employees": "number of eligible employees members",
    "proposed_effective_date": "proposed effective date master application", "signature_date": "signature date signed",
    "insured_name": "insured name", "certificate_number": "certificate number",
}


ROUTE_PROMPT = """You route a question about a group life insurance case to the one data field that holds its answer.

FIELDS (name: label printed on the documents, and what the field describes)
{fields}

Rules
- Pick the single field whose value answers the question.
- The certificate, its rider, the insured employee and the group are different things. Use a rider field only for a question about the rider.
- If the answer needs two or more fields, a calculation or a comparison, return "none".
- If no field fits, return "none".
Return JSON only: {{"field": "<field name or none>"}}

QUESTION: {q}"""
DENSE_MIN = 0.3     # below this cosine similarity no field is chosen (set once, not tuned on the questions)


def route_fields() -> str:
    """The property list given to the LLM router: taken from the ontology (label and class of each property)."""
    from .ontology import LABEL
    from .verifier import ENTITY_OF
    return "\n".join(f"- {f}: {LABEL.get(f, f.replace('_', ' '))} ({ENTITY_OF.get(f, 'case')})" for f in FIELD_DESC)


def route_llm(q: str, model: str = "llama3.1:8b", call=None):
    """Field name chosen by the LLM, or None. Raises OSError when Ollama is not reachable."""
    import json
    if call is None:
        from .fields_llm import cached_call
        call = cached_call(config.PROCESSED / "llm_cache" / model.replace(":", "_"))
    try:
        field = json.loads(call(ROUTE_PROMPT.format(fields=route_fields(), q=q), model)).get("field")
    except (json.JSONDecodeError, AttributeError):
        return None
    return field if field in FIELD_DESC else None


class QA:
    def __init__(self, facts=None):
        primary = extract.load_pages()
        alt = {(p.doc_id, p.page): p for p in extract.load_pages(in_dir=config.PROCESSED / "pages_alt")}
        self.keys = [(p.doc_id, p.page) for p in primary]
        texts = [p.text + ("\n" + alt[k].text if p.route == "ocr" and k in alt else "") for p, k in zip(primary, self.keys)]
        self.page_text = dict(zip(self.keys, texts))
        self.bm25 = BM25(texts)
        self.fields = list(FIELD_DESC)
        self.field_bm25 = BM25([f"{f.replace('_', ' ')} {FIELD_DESC[f]}" for f in self.fields])
        facts = facts if facts is not None else load_facts(config.FACTS / "facts.json")
        self.graph = defaultdict(list)
        for f in facts:
            if f.status in LOADED:
                self.graph[f.field].append(f)
        self._dense = None

    # ------------------------------------------------------------- page retrieval
    def bm25_pages(self, q: str, k: int = 5) -> list:
        s = self.bm25.scores(q)
        return [self.keys[i] for i in sorted(range(len(s)), key=lambda i: -s[i])[:k]]

    def dense_pages(self, q: str, k: int = 5, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None
        if self._dense is None:
            m = SentenceTransformer(model)
            self._dense = (m, m.encode([self.page_text[k][:2000] for k in self.keys], normalize_embeddings=True))
        m, emb = self._dense
        qv = m.encode([q], normalize_embeddings=True)[0]
        s = emb @ qv
        return [self.keys[i] for i in sorted(range(len(s)), key=lambda i: -s[i])[:k]]

    # ------------------------------------------------------------- graph answer
    def field_for(self, q: str, min_score: float = 2.0):
        s = self.field_bm25.scores(q)
        i = max(range(len(s)), key=lambda i: s[i])
        return (self.fields[i], round(s[i], 2)) if s[i] >= min_score else (None, round(s[i], 2))

    def field_dense(self, q: str, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """(field, cosine) by embedding similarity to the field descriptions; None without sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None
        if getattr(self, "_fdense", None) is None:
            m = SentenceTransformer(model)
            self._fdense = (m, m.encode([f"{f.replace('_', ' ')}: {FIELD_DESC[f]}" for f in self.fields], normalize_embeddings=True))
        m, emb = self._fdense
        s = emb @ m.encode([q], normalize_embeddings=True)[0]
        i = int(s.argmax())
        return (self.fields[i], round(float(s[i]), 2)) if s[i] >= DENSE_MIN else (None, round(float(s[i]), 2))

    def graph_answer(self, q: str, matcher: str = "bm25", field="?") -> dict:
        """matcher: bm25 | dense | llm. Pass field (a name or None) to use a choice made elsewhere."""
        if field != "?":
            score = "chosen by the router"
        elif matcher == "llm":
            field, score = route_llm(q), "LLM router"
        elif matcher == "dense":
            field, score = self.field_dense(q) or (None, "sentence-transformers not installed")
        else:
            field, score = self.field_for(q)
        if field is None:
            return {"answer": None, "field": None, "why": f"no field matches the question ({score})" if isinstance(score, str)
                    else f"no field matches the question (best score {score})", "evidence": []}
        fs = self.graph.get(field, [])
        if not fs:
            return {"answer": None, "field": field, "why": "no verified value for this field in the graph", "evidence": []}
        votes = Counter(f.normalized for f in fs)
        value = votes.most_common(1)[0][0]
        ev = sorted({(f.doc_id, f.page) for f in fs if f.normalized == value})
        return {"answer": value, "field": field, "why": f"{len(ev)} evidence page(s)", "evidence": ev,
                "status": sorted({f.status for f in fs if f.normalized == value})}

    def hybrid(self, q: str, k: int = 3, matcher: str = "bm25") -> dict:
        g = self.graph_answer(q, matcher)
        pages = list(dict.fromkeys(list(g["evidence"]) + self.bm25_pages(q, k)))
        return {**g, "pages": pages}
