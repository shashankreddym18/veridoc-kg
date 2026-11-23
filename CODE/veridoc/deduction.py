"""Stage 5b: fill the 22-column payroll-deduction file from verified facts.

A column is filled only from a loaded fact. A column with no supporting
fact stays empty: the system never guesses a value.
"""
from __future__ import annotations
import csv, json
import xlrd
from . import config
from .kg import LOADED

# deduction-file column -> canonical field (None: no source field in this document set)
COLUMN_FIELD = {
    "Group Number": "group_number", "Group Name": "group_name", "Group Premium Due Date": None, "Employee Id": None,
    "EE Last Name": None, "EE First Name": None, "EE MI": None, "EE Status": None,
    "Certificate Number": "certificate_number", "Insured Name": "insured_name", "EE Effective Date": "certificate_date",
    "Certificate Premium Due Date": None, "Payroll Frequency": "payroll_frequency", "Number of Deductions": None,
    "Payroll Deduction Amount": "payroll_deduction_amount", "Premium Due Amount": None, "EE Address 1": None,
    "EE Address 2": None, "EE Address 3": None, "EE City": "ee_city", "EE State": "ee_state", "EE Zip Code": "ee_zip",
}


def template_columns() -> list:
    sh = xlrd.open_workbook(config.RAW / config.DEDUCTION_TEMPLATE).sheet_by_index(0)
    return [str(c).strip() for c in sh.row_values(0)]


def fill(facts: list) -> tuple[list, dict, dict]:
    """Returns (columns, row, sources). Prefers the longest value for names."""
    cols = template_columns()
    row, src = {}, {}
    for col in cols:
        field = COLUMN_FIELD.get(col)
        cands = [f for f in facts if field and f.field == field and f.status in LOADED]
        if not cands:
            row[col] = ""; continue
        best = max(cands, key=lambda f: len(f.value))
        row[col] = best.value.replace("$ ", "$")
        src[col] = {"fact_id": best.fact_id, "doc_id": best.doc_id, "page": best.page, "status": best.status}
    return cols, row, src


def save(facts: list, out_dir=None) -> dict:
    out_dir = out_dir or config.PROCESSED / "deduction"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols, row, src = fill(facts)
    with open(out_dir / "deduction_file.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerow(row)
    (out_dir / "deduction_sources.json").write_text(json.dumps(src, indent=1))
    return {"columns": len(cols), "filled": sum(bool(v) for v in row.values()), "empty": sum(not v for v in row.values())}
