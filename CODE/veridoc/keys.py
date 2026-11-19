"""Canonical forms and deterministic entity keys (applied before any similarity matching).

Two documents that spell the same value differently must not look like a conflict, and two
different values must not look the same. So every value is compared in one canonical form, and
every entity gets a key built only from such forms, by fixed rules:

  value      canon(field, raw)      dates -> YYYY-MM-DD, money -> 0.00, states -> two letters,
                                    frequencies -> fixed list, names -> name_key (below), ids -> upper case
  name       name_key("Acme Widgets, Inc.") == name_key("ACME WIDGETS INC") == "acme widgets"
  entity     entity_key(cls, values) -> (key, basis); basis says which rule built it, or None when the
             fields a key needs are missing (the entity then stays local to its case and is never merged)

Matching order for entity resolution: (1) same key; (2) nothing else is merged automatically.
Similarity matching (fuzzy or embedding) may only propose a pair for human review; it never merges.

Personal identifiers (SSN, employee id) are never stored: id_hash() keeps a keyed hash (HMAC-SHA256)
with a secret from the environment variable VERIDOC_ID_SECRET, and refuses to run without one.
"""
from __future__ import annotations
import hashlib, hmac, os, re, unicodedata

from .fields import FIELD_TYPES, norm_value

LEGAL_SUFFIXES = {"inc", "incorporated", "llc", "l l c", "ltd", "limited", "corp", "corporation", "co", "company",
                  "plc", "lp", "llp", "pc", "pllc"}
LEADING = {"the"}


def name_key(v: str | None) -> str | None:
    """Organisation or person name in one form: NFKC, case-folded, '&' -> 'and', punctuation removed,
    leading 'the' and trailing legal suffixes (Inc, LLC, Corp, ...) dropped."""
    if not v:
        return None
    s = unicodedata.normalize("NFKC", v).casefold().replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    toks = s.split()
    while toks and toks[0] in LEADING:
        toks = toks[1:]
    changed = True
    while changed and toks:
        changed = False
        for n in (3, 2, 1):                       # "l l c" comes out of "L.L.C."
            if len(toks) > n and " ".join(toks[-n:]) in LEGAL_SUFFIXES:
                toks, changed = toks[:-n], True
                break
    return " ".join(toks) or None


def canon(field: str, raw: str | None) -> str | None:
    """Canonical form of one value of a field, or None when it does not parse as the field's type."""
    if raw is None:
        return None
    ftype = FIELD_TYPES.get(field, "text")
    if ftype == "name":
        return name_key(raw)
    if ftype == "text":
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", raw)).strip().casefold() or None
    return norm_value(ftype, raw)


def id_hash(raw: str) -> str:
    """Keyed hash of a personal identifier (digits and letters only, so '123-45-6789' == '123456789')."""
    secret = os.environ.get("VERIDOC_ID_SECRET")
    if not secret:
        raise RuntimeError("set VERIDOC_ID_SECRET to hash personal identifiers; raw identifiers are never stored")
    clean = re.sub(r"[^0-9A-Za-z]", "", raw).upper()
    return hmac.new(secret.encode(), clean.encode(), hashlib.sha256).hexdigest()[:16]


# key rules per class, tried in order: (basis name, fields it needs). The first rule whose fields are all
# present builds the key. Fields are canonical values.
KEY_RULES = {
    "Group": [("group_number", ["group_number"]),
              ("name+state", ["group_name", "situs_state"]),
              ("name+state", ["group_name", "group_state"])],
    "Insured": [("employee_id_hash", ["employee_id_hash"]),
                ("name+dob", ["insured_name", "date_of_birth"])],
    "Certificate": [("certificate_number", ["certificate_number"]),
                    ("group+insured+date", ["@Group", "@Insured", "certificate_date"])],
    "Rider": [("certificate+rider_date", ["@Certificate", "rider_date"])],
}


def entity_key(cls: str, values: dict, keys: dict | None = None) -> tuple:
    """(key, basis) for an entity of class cls from its canonical field values; '@Class' in a rule refers
    to the key of a related entity (keys: class -> key). (None, None) when no rule applies."""
    keys = keys or {}
    for basis, need in KEY_RULES.get(cls, []):
        parts = [keys.get(n[1:]) if n.startswith("@") else values.get(n) for n in need]
        if all(parts):
            return f"{cls.lower()}:" + "|".join(parts), basis
    return None, None


def all_keys(values_by_class: dict) -> dict:
    """class -> (key, basis), resolving classes in dependency order (Group, Insured, then Certificate, Rider)."""
    out, keys = {}, {}
    for cls in ("Group", "Insured", "Certificate", "Rider"):
        if cls in values_by_class:
            out[cls] = entity_key(cls, values_by_class[cls], keys)
            if out[cls][0]:
                keys[cls] = out[cls][0]
    return out
