"""The two shared formats every stage reads and writes.

Page  : output of routing + text extraction (one JSON file per page).
Fact  : one extracted value with its evidence and the result of every check.

Coordinates are PDF points (1/72 inch), origin at the top-left of the page,
whatever engine produced them, so every stage and the UI agree.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float = 100.0        # OCR confidence 0-100 (native text = 100)
    line: int = 0              # line index on the page, in reading order


@dataclass
class Page:
    doc_id: str
    page: int                  # 1-based
    width: float
    height: float
    route: str                 # "native" | "ocr"
    engine: str                # "pdfplumber" | "tesseract" | ...
    text: str
    words: list = field(default_factory=list)      # list[Word]
    features: dict = field(default_factory=dict)   # router features
    seconds: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "Page":
        d = json.loads(s)
        d["words"] = [Word(**w) for w in d["words"]]
        return Page(**d)


# check outcomes
PASS, FAIL, SKIP = "pass", "fail", "skip"
CHECKS = ("agree", "span", "nli", "shacl", "crossdoc", "template")


@dataclass
class Fact:
    fact_id: str
    doc_id: str
    page: int
    field: str                 # canonical field name (see fields.py FIELD_SPECS)
    value: str                 # value as written in the document
    normalized: str = ""       # canonical form used for comparison
    span: str = ""             # evidence text copied from the page
    bbox: Optional[list] = None  # [x0, y0, x1, y1] in PDF points
    extractor: str = "rules"   # "rules" | "proximity" | "llm:<model>"
    checks: dict = field(default_factory=dict)     # check -> {"result", "detail"}
    status: str = "candidate"  # candidate | verified | quarantined | human_verified | rejected
    reason: str = ""           # reason code when quarantined, e.g. "span:not_found"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Fact":
        return Fact(**d)


def save_facts(path, facts) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([x.to_dict() for x in facts], f, ensure_ascii=False, indent=1)


def load_facts(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [Fact.from_dict(d) for d in json.load(f)]
