"""Stage 3 (classical side): field extraction from Page text.

Two classical extractors, both from the proposal's baseline row:
  rules      label-anchored regular expressions with a type gate
             (a value is emitted only if it parses as its field type)
  proximity  naive label proximity: take the tokens next to the label,
             no type gate (the weaker baseline; over-generates on blank forms)

Every emitted Fact carries its evidence span and bounding box.
"""
from __future__ import annotations
import re
from datetime import date
from rapidfuzz import process, fuzz
from . import config
from .schema import Fact, Page

# ------------------------------------------------------------------ types
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
STATES = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
          "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
          "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
          "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
          "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
          "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
          "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
          "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
          "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
          "Wisconsin": "WI", "Wyoming": "WY"}
STATE_CODES = set(STATES.values())
FREQ = {"weekly": "Weekly", "bi-weekly": "Bi-Weekly", "biweekly": "Bi-Weekly", "semi-monthly": "Semi-Monthly",
        "monthly": "Monthly", "quarterly": "Quarterly", "annual": "Annual"}
PERIODS_PER_YEAR = {"Weekly": 52, "Bi-Weekly": 26, "Semi-Monthly": 24, "Monthly": 12, "Quarterly": 4, "Annual": 1}


def norm_date(v: str) -> str | None:
    v = v.strip()
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", v)
    try:
        if m:
            return date(int(m[3]), int(m[1]), int(m[2])).isoformat()
        m = re.fullmatch(r"([A-Za-z]{3,9})\.? (\d{1,2})[,.]? ?(\d{4})", v)
        if m:
            # OCR-tolerant month: "Auaust" -> August
            hit = process.extractOne(m[1].capitalize(), MONTHS, scorer=fuzz.ratio, score_cutoff=75) or \
                  process.extractOne(m[1].capitalize(), [x[:3] for x in MONTHS], scorer=fuzz.ratio, score_cutoff=100)
            if hit:
                return date(int(m[3]), hit[2] + 1, int(m[2])).isoformat()
    except ValueError:
        return None
    return None


def norm_money(v: str) -> str | None:
    v = re.sub(r",\s+(?=\d{3}\b)", ",", v.strip())          # OCR: "$20, 000"
    v = re.sub(r"(?<=\d)\.(?=\d{3}$)", ",", v)              # OCR: "$15.000" (three digits after a period = thousands)
    m = re.fullmatch(r"\$?\s?([\d,]+(?:\.\d{1,2})?)", v)
    return f"{float(m[1].replace(',', '')):.2f}" if m else None


def norm_value(ftype: str, v: str) -> str | None:
    """Canonical form of a value, or None when it does not parse as its type."""
    v = v.strip()
    if ftype == "date":
        return norm_date(v)
    if ftype == "money":
        return norm_money(v)
    if ftype == "int":
        return str(int(v)) if re.fullmatch(r"\d{1,6}", v) else None
    if ftype == "percent":
        m = re.fullmatch(r"(\d{1,2}(?:\.\d+)?) ?%", v)
        return m[1] if m else None
    if ftype == "state":
        if v.upper() in STATE_CODES:
            return v.upper()
        return STATES.get(v.title())
    if ftype == "zip":
        return v[:5] if re.fullmatch(r"\d{5}(-\d{4})?", v) else None
    if ftype == "freq":
        return FREQ.get(v.lower())
    if ftype == "id":
        return v.upper() if re.fullmatch(r"[A-Za-z0-9-]{4,14}", v) and re.search(r"\d", v) else None
    if ftype == "name":
        return re.sub(r"\s+", " ", v).lower() if re.fullmatch(r"[A-Z][A-Za-z&.,'-]*(?: [A-Z&][A-Za-z&.,'-]*){0,5}", v) else None
    return re.sub(r"\s+", " ", v).lower() or None    # free text


# field -> type.  Deduction-file columns that each field feeds are in deduction.py.
FIELD_TYPES = {
    "group_name": "name", "group_number": "id", "presented_by": "name", "proposal_date": "date",
    "situs_state": "state", "group_street": "text", "group_state": "state", "eligible_employees": "int",
    "proposed_effective_date": "date", "signature_date": "date",
    "face_amount": "money", "certificate_number": "id", "insured_name": "name", "certificate_date": "date",
    "maturity_date": "date", "risk_class": "text", "date_of_birth": "date", "issue_age": "int", "plan": "text",
    "min_paidup_amount": "money", "premium_frequency": "freq", "loan_interest_rate": "percent",
    "rider_date": "date", "rider_benefit_amount": "money", "rider_annual_premium": "money",
    "rider_termination_date": "date", "annual_premium": "money",
    "ee_city": "text", "ee_state": "state", "ee_zip": "zip", "date_of_hire": "date", "annual_salary": "money",
    "payroll_frequency": "freq", "payroll_deduction_amount": "money", "job_title": "text",
}

_D = r"(?:\d{1,2}/\d{1,2}/\d{4}|[A-Z][a-z]{2,8}\.? \d{1,2}[,.] ?\d{4})"
_STREET_SFX = r"(?:St|Ave|Ln|Rd|Dr|Way|Ct|Blvd|Pl|Street|Avenue|Lane|Road|Drive|Court)"
_NAME = r"[A-Z][A-Za-z&.'-]+(?: [A-Z&][A-Za-z&.'-]*){0,4}"
_FREQ = r"Semi-Monthly|Bi-Weekly|Monthly|Weekly|Quarterly|Annual"
_STATE = "|".join(sorted(STATES, key=len, reverse=True))

# (pattern, docs it applies to or None).  Each named group is a field.
RULES = [
    (rf"Group Insurance Proposal for\s*\n(?P<group_name>{_NAME})\s*$", ["proposal"]),
    (rf"Presented By:\s*(?P<presented_by>{_NAME})", ["proposal"]),
    (rf"^Date:\s*(?P<proposal_date>{_D})", ["proposal"]),
    (rf"Situs State:\s*(?P<situs_state>{_STATE})\b", ["proposal"]),
    (rf"Group Policyholder Name\W*(?P<group_name>{_NAME})\s*$", ["master_app"]),
    (r"PO Box or Street, Apt\. or Suite City\s*\n(?P<group_street>\d{1,6}(?: [A-Z][a-z]+){1,4})", ["master_app"]),
    (rf"State or Country ZIP/Postal Code\s*\n(?P<group_state>{_STATE})\b", ["master_app"]),
    (r"(?P<eligible_employees>\d{1,5})\s*\n\s*2\. Number of eligible employees/members", ["master_app"]),
    (r"Number of eligible employees/members:[ _]*(?P<eligible_employees>\d{1,5})\b", ["master_app"]),
    (r"Proposed effective date[^:\n]*:\s*(?P<proposed_effective_date>\S+)", ["master_app"]),
    (r"[Dd]ate:\s*\|?\s*(?P<signature_date>\d{1,2} ?\d?/\d{1,2}/\d{4})", ["master_app"]),
    (r"FACE AMOUNT\s*\|?\s*(?P<face_amount>\$ ?[\d,]+(?:\.\d\d)?)", ["certificate", "employee_app_blank"]),
    (r"CERTIFICATE NUMBER\W*(?P<certificate_number>[A-Z]{0,3}\d{5,12})\b", ["certificate"]),
    (rf"INSURED\s*:?\s*(?P<insured_name>[A-Z][a-z]+ (?:[A-Z]\. )?[A-Z][a-z]+)\s*(?:FACE AMOUNT|$)", ["certificate"]),
    (rf"CERTIFICATE DATE\s+(?P<certificate_date>{_D})", ["certificate"]),
    (rf"MATURITY DATE\s+(?P<maturity_date>{_D})", ["certificate"]),
    (r"RISK CLASS\s+(?P<risk_class>Non-Tobacco|Tobacco)", ["certificate"]),
    (rf"DATE OF BIRTH\s+(?P<date_of_birth>{_D})", ["certificate"]),
    (r"ISSUE AGE\s+(?P<issue_age>\d{1,3})\b", ["certificate"]),
    (r"^PLAN\s+(?P<plan>Whole Life)\s*$", ["certificate"]),
    (r"MINIMUM PAID-UP LIFE INSURANCE AMOUNT\s+(?P<min_paidup_amount>\$[\d,]+)", ["certificate"]),
    (rf"INITIAL PREMIUM PAYMENT FREQUENCY\s+(?P<premium_frequency>{_FREQ})", ["certificate"]),
    (r"LOAN INTEREST RATE\s+(?P<loan_interest_rate>\d{1,2}(?:\.\d+)?%)", ["certificate"]),
    (rf"^Rider Date\s+(?P<rider_date>{_D})", ["certificate"]),
    (r"^Benefit Amount\s+(?P<rider_benefit_amount>\$[\d,]+)", ["certificate"]),
    (r"^Annual Premium\s+(?P<rider_annual_premium>\$[\d,]+\.\d\d)", ["certificate"]),
    (rf"^Termination Date\s+(?P<rider_termination_date>{_D})", ["certificate"]),
    (r"Annual Certificate Premium[^\n$]*\$\s*(?P<annual_premium>[\d,]+\.\d\d)", ["certificate"]),
    (rf"GROUP POLICYHOLDER\s+(?P<group_name>{_NAME}?)\s+CERTIFICATE OWNER", ["certificate"]),
    (rf"Group Name:\s*(?P<group_name>{_NAME}?)\s+Group Number", ["certificate", "employee_app_blank"]),
    (r"Group Number:\s*(?P<group_number>[A-Z0-9-]{4,12})\b", ["certificate", "employee_app_blank"]),
    # v2 (fixed on the synthetic dev set): optional street before the city, zip optional,
    # hire date and salary matched separately so one empty cell does not hide the other
    (rf"Residential Address[^\n]*\n(?:Street\b ?)?(?:\d+ (?:[A-Z][a-z]+ )+{_STREET_SFX}\.? )?"
     r"(?P<ee_city>[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)?) (?P<ee_state>[A-Z]{2})\b(?: (?P<ee_zip>\d{5})\b)?",
     ["certificate", "employee_app_blank"]),
    (r"Date of Hire[^\n]*\n(?P<date_of_hire>\d{1,2}/\d{1,2}/\d{4})\b", ["certificate", "employee_app_blank"]),
    (r"Date of Hire[^\n]*\n(?:\d{1,2}/\d{1,2}/\d{4}\s+)?(?P<annual_salary>\$[\d,]+)", ["certificate", "employee_app_blank"]),
    (r"Coverage Face Amount \$[ _]*(?P<face_amount>\$ ?\d{1,3}(?:[,.] ?\d{3})+)", ["employee_app_blank"]),
    (rf"Your (?P<payroll_frequency>{_FREQ}) Premium\s*\|?\s*(?P<payroll_deduction_amount>\$ ?[\d,]+\.\d\d)",
     ["certificate", "employee_app_blank"]),
]
_COMPILED = [(re.compile(p, re.M | (re.I if p.startswith("FACE AMOUNT") else 0)), docs) for p, docs in RULES]

# labels for the naive proximity baseline: field -> label pattern
PROX_LABELS = {
    "group_name": r"Group (?:Policyholder )?Name:?|GROUP POLICYHOLDER|Group Insurance Proposal for",
    "group_number": r"Group Number:?", "presented_by": r"Presented By:", "situs_state": r"Situs State:",
    "eligible_employees": r"Number of eligible employees/members:?",
    "proposed_effective_date": r"Proposed effective date[^:\n]*:", "face_amount": r"(?i:face amount)",
    "certificate_number": r"CERTIFICATE NUMBER", "insured_name": r"INSURED\b", "certificate_date": r"CERTIFICATE DATE",
    "maturity_date": r"MATURITY DATE", "risk_class": r"RISK CLASS", "date_of_birth": r"(?i:date of birth)",
    "issue_age": r"ISSUE AGE", "plan": r"^PLAN\b", "premium_frequency": r"INITIAL PREMIUM PAYMENT FREQUENCY",
    "loan_interest_rate": r"LOAN INTEREST RATE", "ee_city": r"\bCity\b", "ee_state": r"\bState\b",
    "ee_zip": r"Zip Code", "date_of_hire": r"Date of Hire(?: \(mm/dd/yyyy\))?", "annual_salary": r"Annual Salary/Compensation",
    "job_title": r"Job Title/Occupation", "payroll_deduction_amount": r"Your [\w-]* ?Premium",
}


# ------------------------------------------------------------------ helpers
def _word_offsets(page: Page) -> list:
    """Character range of every word inside page.text (text is words joined by
    single spaces within a line and newlines between lines)."""
    out, pos, last = [], 0, None
    for w in page.words:
        if last is not None:
            pos += 1                      # the space or newline separator
        out.append((pos, pos + len(w.text), w))
        pos += len(w.text); last = w.line
    return out


def _bbox(offsets: list, a: int, b: int):
    ws = [w for s, e, w in offsets if s < b and e > a]
    if not ws:
        return None
    return [min(w.x0 for w in ws), min(w.y0 for w in ws), max(w.x1 for w in ws), max(w.y1 for w in ws)]


def _line_span(text: str, a: int, b: int) -> str:
    s = text.rfind("\n", 0, a) + 1
    e = text.find("\n", b)
    return text[s: e if e != -1 else len(text)]


def _fact(page: Page, field: str, value: str, a: int, b: int, offsets, extractor: str, n: int, span_from: int | None = None) -> Fact:
    ftype = FIELD_TYPES.get(field, "text")
    return Fact(fact_id=f"{page.doc_id}:{page.page}:{field}:{extractor}:{n}", doc_id=page.doc_id, page=page.page, field=field,
                value=value.strip(), normalized=norm_value(ftype, value) or "",
                span=_line_span(page.text, span_from if span_from is not None else a, b), bbox=_bbox(offsets, a, b), extractor=extractor)


# ------------------------------------------------------------------ extractors
def extract_rules(page: Page, tag: str = "rules") -> list:
    facts, offsets = [], _word_offsets(page)
    for rx, docs in _COMPILED:
        if docs and config.doc_type(page.doc_id) not in docs:
            continue
        for m in rx.finditer(page.text):
            for field, value in m.groupdict().items():
                if not value or not value.strip():
                    continue
                if norm_value(FIELD_TYPES[field], value) is None:
                    continue              # type gate: not a valid value for this field
                a, b = m.span(field)
                facts.append(_fact(page, field, value, a, b, offsets, tag, len(facts), span_from=m.start()))
    return facts


def extract_proximity(page: Page, max_tokens: int = 3) -> list:
    """Naive baseline: first occurrence of each label; value = next tokens on
    the line, else the first tokens of the next line. No type gate."""
    facts, offsets = [], _word_offsets(page)
    for field, label in PROX_LABELS.items():
        m = re.search(label, page.text, re.M)
        if not m:
            continue
        eol = page.text.find("\n", m.end())
        eol = len(page.text) if eol == -1 else eol
        rest = page.text[m.end():eol]
        start = m.end()
        if not rest.strip():              # nothing after the label: look at the next line
            nxt = page.text.find("\n", eol + 1)
            rest = page.text[eol + 1: nxt if nxt != -1 else len(page.text)]
            start = eol + 1
        toks = list(re.finditer(r"\S+", rest))[:max_tokens]
        if not toks:
            continue
        a, b = start + toks[0].start(), start + toks[-1].end()
        value = page.text[a:b].strip(" :|_")
        if value:
            facts.append(_fact(page, field, value, a, b, offsets, "proximity", len(facts), span_from=m.start()))
    return facts
