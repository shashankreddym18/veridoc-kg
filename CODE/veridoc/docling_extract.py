"""Docling path (document-AI layout + OCR) that returns the shared Page format.

Needs `pip install docling` (downloads layout and OCR models on first use). On a laptop:

    pip install docling
    cd CODE && python -m veridoc.docling_extract                 # all six documents
    python -m veridoc.docling_extract --docs master_app certificate
Writes DATA/processed/pages_docling/*.json (same format as pages/), so any extractor can run on it:
    from veridoc import extract; pages = extract.load_pages(in_dir=config.PROCESSED / "pages_docling")
Then compare against Tesseract on the hand-corrected scanned pages (G-A) with
    python EVALUATIONS/eval_docling.py
"""
from __future__ import annotations
import argparse, time
from collections import defaultdict

from . import config, extract
from .schema import Page, Word

OUT = config.PROCESSED / "pages_docling"


LINE_H = 22.0      # blocks up to this height (points) are single lines and can share a visual row
ROW_TOL = 5.0      # tops within this many points are the same visual row


def items_of(doc) -> tuple:
    """(page -> [(top, left, bottom, right, text)], page -> (width, height)) from a DoclingDocument.

    Includes what the default body-only iteration leaves out: table cells (one entry per table row),
    page headers and footers (the "furniture" layer) and text inside picture regions."""
    from docling_core.types.doc.document import ContentLayer, TableItem
    sizes = {no: (pg.size.width, pg.size.height) for no, pg in doc.pages.items()}
    items = defaultdict(list)

    def box(bbox, page_no):
        b = bbox.to_top_left_origin(page_height=sizes.get(page_no, (612.0, 792.0))[1])   # docling boxes are bottom-left by default
        return b.t, b.l, b.b, b.r

    layers = {ContentLayer.BODY, ContentLayer.FURNITURE}
    for item, _level in doc.iterate_items(traverse_pictures=True, included_content_layers=layers):
        if not getattr(item, "prov", None):
            continue
        prov = item.prov[0]
        if isinstance(item, TableItem):
            rows = defaultdict(list)
            for c in item.data.table_cells:
                if (c.text or "").strip():
                    rows[c.start_row_offset_idx].append(c)
            t0, l0, b0, r0 = box(prov.bbox, prov.page_no)
            step = (b0 - t0) / max(1, item.data.num_rows)
            for r, cells in sorted(rows.items()):
                cells.sort(key=lambda c: c.start_col_offset_idx)
                texts = [c.text.strip() for c in cells]
                boxes = [box(c.bbox, prov.page_no) for c in cells if c.bbox is not None]
                if boxes:
                    t, l, b, rr = min(x[0] for x in boxes), min(x[1] for x in boxes), max(x[2] for x in boxes), max(x[3] for x in boxes)
                else:                                    # no cell boxes: spread the rows over the table box
                    t, l, b, rr = t0 + r * step, l0, t0 + (r + 1) * step, r0
                items[prov.page_no].append((t, l, b, rr, " ".join(texts)))
            continue
        text = getattr(item, "text", None)
        if text and text.strip():
            items[prov.page_no].append((*box(prov.bbox, prov.page_no), text))
    return items, sizes


def visual_rows(entries: list) -> list:
    """Top-to-bottom rows; single-line blocks whose tops are within ROW_TOL share a row, left to right.

    Docling returns a label and its value as separate blocks; without this the value can come before
    its label. Multi-line blocks (paragraphs) always stay on their own."""
    rows = []
    for e in sorted(entries):
        single = (e[2] - e[0]) <= LINE_H
        if rows and single and rows[-1]["single"] and abs(e[0] - rows[-1]["top"]) <= ROW_TOL:
            rows[-1]["cells"].append(e)
        else:
            rows.append({"top": e[0], "single": single, "cells": [e]})
    return [sorted(r["cells"], key=lambda e: e[1]) for r in rows]


def to_pages(doc, doc_id: str, secs: float = 0.0) -> list:
    items, sizes = items_of(doc)
    pages = []
    for no in sorted(sizes):
        words, lines = [], []
        for li, cells in enumerate(visual_rows(items.get(no, []))):
            line = []
            for (y0, x0, y1, x1, text) in cells:
                toks = text.split()
                line += toks
                words += [Word(tok, round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1), 100.0, li) for tok in toks]   # block box per word (Docling has no word boxes)
            lines.append(" ".join(line))            # one space between words, as in the other extractors
        w, h = sizes[no]
        pages.append(Page(doc_id, no, float(w), float(h), "docling", "docling", "\n".join(lines), words, {}, secs))
    return pages


def convert(doc_id: str) -> list:
    from docling.document_converter import DocumentConverter
    t0 = time.time()
    doc = DocumentConverter().convert(str(config.RAW / config.DOC_FILES[doc_id])).document
    return to_pages(doc, doc_id, round((time.time() - t0) / max(1, len(doc.pages)), 2))


def main(docs: list) -> None:
    for d in docs:
        pages = convert(d)
        extract.save_pages(pages, OUT)
        print(f"{d:20s} {len(pages)} pages -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="*", default=list(config.DOC_FILES))
    main(ap.parse_args().docs)
