"""Paths and the document registry for the insurance workflow set (D1)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "DATA" / "raw" / "Insurance"
PROCESSED = ROOT / "DATA" / "processed"
PAGES = PROCESSED / "pages"          # one JSON per page (shared page format)
IMAGES = PROCESSED / "images"        # page renders used by OCR and the UI
FACTS = PROCESSED / "facts"          # candidate + verified facts
GOLD = ROOT / "DATA" / "gold"
RESULTS = ROOT / "EVALUATIONS" / "results"
ONTOLOGY = ROOT / "CODE" / "ontology"

# file name -> (doc_id, workflow stage, human title)
DOCS = {
    "Prosposal_de_id.pdf": ("proposal", "1_quote", "Group proposal"),
    "Quest MasterApp_de_id.pdf.pdf": ("master_app", "2_master_application", "Master application"),
    "Employee_Application_mm-wsea6.pdf": ("employee_app_blank", "3_enrollment", "Employee application (blank)"),
    "Sample Policy Specimen.pdf": ("policy_specimen", "4_policy", "Policy specimen"),
    "Employee certificate example_de_id.pdf": ("certificate", "5_certificate", "Employee certificate"),
    "WS Map.pdf": ("ws_map", "0_process_map", "Worksite process map"),
}
DOC_FILES = {v[0]: k for k, v in DOCS.items()}
DOC_TITLES = {v[0]: v[2] for v in DOCS.values()}
DOC_STAGE = {v[0]: v[1] for v in DOCS.values()}
DEDUCTION_TEMPLATE = "WS-sample-deduction-file.xls"

# Synthetic filled forms (G-C) are copies of the blank employee application, so the
# extractors treat them as that document type.
SYNTH = ROOT / "DATA" / "synthetic"


def doc_type(doc_id: str) -> str:
    return "employee_app_blank" if doc_id.startswith("syn_") else doc_id


RENDER_DPI = 300           # OCR render resolution
NATIVE_MIN_CHARS = 50      # router: fewer text-layer characters than this -> OCR
