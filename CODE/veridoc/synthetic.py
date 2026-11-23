"""Synthetic filled employee applications (gold set G-C).

Fills pages 1-2 of the blank MassMutual@WORK employee application with generated
personas, then makes three versions of each form:
  native   the filled PDF with a text layer (as if printed by enrollment software)
  scan300  rasterized at 300 dpi, light noise            -> OCR route
  scan150  rasterized at 150 dpi, blur, noise, 1 deg skew -> OCR route
About 20% of fields are left empty on purpose, so the test also counts values
invented for empty fields. Every value is generated, so the answers are known.
None of the rules were written on these forms, so this is held-out data.

Two sets with different seeds:
  dev   (seed 573)  used to look at errors, fix rules and train the CRF
  test  (seed 2026) generated after the rules were frozen; only scored
Run:  cd CODE && python -m veridoc.synthetic --set dev && python -m veridoc.synthetic --set test
Writes DATA/synthetic/<set>/forms/*.pdf and DATA/synthetic/<set>/gold.json.
"""
from __future__ import annotations
import argparse, io, json, random, subprocess, tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

from . import config

BLANK = config.RAW / config.DOC_FILES["employee_app_blank"]
SEEDS = {"dev": 573, "test": 2026}


def paths(which: str) -> tuple:
    root = config.SYNTH / which
    return root, root / "forms", root / "gold.json"
H = 792.0  # page height in points; positions below are measured from the top, like pdfplumber

# The fields the extractors target on this form type (scored), and where each value is written.
SCORED = ["group_name", "group_number", "ee_city", "ee_state", "ee_zip", "date_of_hire", "annual_salary",
          "face_amount", "payroll_frequency", "payroll_deduction_amount"]
POS = {  # field -> (page index, x, baseline from top)
    "group_name": (0, 78, 146), "group_number": (0, 416, 146),
    "insured_name": (0, 30, 222), "date_of_birth": (0, 342, 226),
    "ee_street": (0, 56, 251), "ee_city": (0, 313, 251), "ee_state": (0, 421, 251), "ee_zip": (0, 524, 251),
    "date_of_hire": (0, 30, 380), "annual_salary": (0, 161, 380), "job_title": (0, 313, 380),
    "face_amount": (1, 214, 508), "annual_premium": (1, 490, 531), "payroll_deduction_amount": (1, 490, 634),
}

FIRST = ["Maria", "James", "Aisha", "Wei", "Carlos", "Priya", "Daniel", "Fatima", "Tomas", "Grace", "Omar", "Lena",
         "Hiro", "Nadia", "Samuel", "Elena", "Kofi", "Ruth", "Ivan", "Mei"]
LAST = ["Alvarez", "Brennan", "Chowdhury", "Dubois", "Eriksen", "Fontaine", "Garza", "Haddad", "Ito", "Jensen",
        "Kowalski", "Lindqvist", "Morales", "Nakamura", "Okafor", "Petrov", "Quinn", "Reyes", "Sato", "Thompson"]
COMPANY = ["Cascade Valley Foods", "Northwind Logistics", "Juniper Health Partners", "Redrock Tooling",
           "Bluewater Credit Union", "Summit Ridge Schools", "Harbor Point Hotels", "Ironwood Manufacturing"]
SUFFIX = ["Inc", "LLC", "Co", "Group", ""]
PLACES = [("Salem", "OR", "973"), ("Eugene", "OR", "974"), ("Bend", "OR", "977"), ("Tacoma", "WA", "984"),
          ("Spokane", "WA", "992"), ("Boise", "ID", "837"), ("Tempe", "AZ", "852"), ("Reno", "NV", "895"),
          ("Fresno", "CA", "937"), ("Ogden", "UT", "844")]
STREETS = ["Oak St", "Maple Ave", "Cedar Ln", "Pine Rd", "Elm Dr", "Birch Way", "Willow Ct"]
JOBS = ["Warehouse Associate", "Registered Nurse", "Line Cook", "Machinist", "Teller", "Teacher", "Driver",
        "Accountant", "Electrician", "Front Desk Agent"]
FREQ = {"Weekly": 52, "Bi-Weekly": 26, "Semi-Monthly": 24, "Monthly": 12}
FACES = [10000, 15000, 20000, 25000, 50000]


def persona(rng: random.Random, i: int, prefix: str) -> dict:
    city, st, z3 = rng.choice(PLACES)
    face = rng.choice(FACES)
    annual = round(face / 1000 * rng.uniform(9.0, 16.0), 2)        # premium per $1,000 of coverage
    freq = rng.choice(list(FREQ))
    hire = f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.randint(2005, 2023)}"
    dob = f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.randint(1962, 2002)}"
    co = rng.choice(COMPANY)
    sfx = rng.choice(SUFFIX)
    v = {
        "group_name": f"{co} {sfx}".strip(), "group_number": f"G{rng.randint(10000, 99999)}",
        "insured_name": f"{rng.choice(FIRST)} {rng.choice('ABCDEFGHJKLMNPRSTW')}. {rng.choice(LAST)}",
        "date_of_birth": dob, "ee_street": f"{rng.randint(100, 9899)} {rng.choice(STREETS)}",
        "ee_city": city, "ee_state": st, "ee_zip": f"{z3}{rng.randint(1, 99):02d}",
        "date_of_hire": hire, "annual_salary": f"${rng.randint(28, 140) * 1000:,}", "job_title": rng.choice(JOBS),
        "face_amount": f"${face:,}", "annual_premium": f"{annual:.2f}", "payroll_frequency": freq,
        "payroll_deduction_amount": f"{annual / FREQ[freq]:.2f}",
    }
    # leave about 20% of the scored fields empty (frequency and amount are written together)
    empty = {f for f in SCORED if f not in ("payroll_frequency",) and rng.random() < 0.2}
    if "payroll_deduction_amount" in empty:
        empty.add("payroll_frequency")
    return {"form": f"syn_{prefix}{i:03d}", "values": v, "empty": sorted(empty), "font": rng.choice(["Courier", "Helvetica"])}


def overlay(p: dict) -> list[bytes]:
    """One single-page overlay PDF per form page."""
    pages = []
    for pi in (0, 1):
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=(612, H))
        c.setFont(p["font"], 9.5)
        for field, (page, x, top) in POS.items():
            if page != pi or field in p["empty"]:
                continue
            c.drawString(x, H - top, p["values"][field])
        if pi == 1:                       # print the payroll frequency over "Modal", as enrollment software does
            c.setFillColorRGB(1, 1, 1)
            c.rect(372, H - 637, 100, 13, stroke=0, fill=1)
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica-Bold", 9)
            freq = "" if "payroll_frequency" in p["empty"] else p["values"]["payroll_frequency"] + " "
            c.drawRightString(471, H - 634, f"Your {freq}Premium")
        c.save()
        pages.append(buf.getvalue())
    return pages


def write_native(p: dict, path: Path) -> None:
    blank = PdfReader(str(BLANK))
    w = PdfWriter()
    for pi, ov in enumerate(overlay(p)):
        page = w.add_page(blank.pages[pi])                 # pypdf 7: attach to the writer first, then merge
        page.merge_page(PdfReader(io.BytesIO(ov)).pages[0])
    with open(path, "wb") as fh:
        w.write(fh)


def write_scan(native: Path, path: Path, dpi: int, degrade: bool, rng: random.Random) -> None:
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-gray", "-png", str(native), f"{td}/p"], check=True)
        imgs = []
        for png in sorted(Path(td).glob("p*.png")):
            im = Image.open(png).convert("L")
            a = np.asarray(im).astype(np.float32)
            a += np.random.default_rng(rng.randint(0, 10**6)).normal(0, 12 if degrade else 5, a.shape)
            im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
            if degrade:
                im = im.filter(ImageFilter.GaussianBlur(0.8)).rotate(rng.uniform(-1.2, 1.2), fillcolor=255, expand=False)
            imgs.append(im.convert("RGB"))
        imgs[0].save(path, save_all=True, append_images=imgs[1:], resolution=dpi)


CONDITIONS = {"native": None, "scan300": (300, False), "scan150": (150, True)}


def ensure(which: str, n: int = 20) -> None:
    """Generate the set if its PDFs are missing or incomplete (the seed makes this repeatable)."""
    _, out, gold_path = paths(which)
    if len(list(out.glob("syn_*.pdf"))) < n * len(CONDITIONS) or not gold_path.exists():
        main(which, n)


def main(which: str = "dev", n: int = 20) -> None:
    rng = random.Random(SEEDS[which])
    root, out, gold_path = paths(which)
    out.mkdir(parents=True, exist_ok=True)
    gold = []
    for i in range(1, n + 1):
        p = persona(rng, i, which[0])
        native = out / f"{p['form']}_native.pdf"
        write_native(p, native)
        for cond, spec in CONDITIONS.items():
            if spec:
                write_scan(native, out / f"{p['form']}_{cond}.pdf", *spec, rng)
        gold.append({"form": p["form"], "font": p["font"], "empty": p["empty"],
                     "values": {f: (None if f in p["empty"] else p["values"][f]) for f in SCORED},
                     "unscored": {f: p["values"][f] for f in p["values"] if f not in SCORED}})
    gold_path.write_text(json.dumps({"note": "Synthetic personas; every value generated. None = field left empty.",
                                     "set": which, "seed": SEEDS[which], "conditions": list(CONDITIONS),
                                     "scored_fields": SCORED, "forms": gold}, indent=1))
    print(f"{which}: {n} personas x {len(CONDITIONS)} conditions -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SEEDS), default="dev")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    main(a.set, a.n)
