"""LLM second opinion (needs Ollama, like fields_llm.py).

The cross-encoder NLI check is cheap but rejects some true facts. Instead of quarantining what it
rejects, a local LLM is asked whether the evidence supports the claim. A fact is rejected only when
both say no ("cascade"). Measured in EVALUATIONS/eval_judge.py (E5c) before it is used in the pipeline.
"""
from __future__ import annotations
import json

from . import config, verifier
from .fields_llm import cached_call

PROMPT = """You check one value extracted from an insurance document against the text it was taken from.

TEXT FROM THE PAGE:
{premise}

CLAIM: {hypothesis}

Is the claim supported by the text? Answer true only if the text states this value for this field.
Answer false if the value is not in the text, if it differs in any character, if it belongs to a
different field, or if it is only a label or heading with nothing filled in.
Return JSON only: {{"supported": true}} or {{"supported": false}}"""


def judge(pairs: list, model: str = "llama3.1:8b", call=None) -> list:
    """pairs: (premise, hypothesis). Returns True / False per pair, or None when the reply is unusable."""
    call = call or cached_call(config.PROCESSED / "llm_cache" / model.replace(":", "_"))
    out = []
    for premise, hypothesis in pairs:
        try:
            v = json.loads(call(PROMPT.format(premise=premise, hypothesis=hypothesis), model)).get("supported")
        except (json.JSONDecodeError, AttributeError):
            v = None
        out.append(v if isinstance(v, bool) else None)
    return out


def pair(src, claim, texts) -> tuple:
    """Premise with context (title + lines above + evidence line) and the fully qualified claim."""
    return verifier.premise(src, texts, "context"), verifier.verbalize(claim, "field")
