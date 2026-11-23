"""Stage 1+2 for every document: route each page, extract text, save Page JSON.

OCR pages are read twice (two Tesseract configurations). The second reading
is stored under pages_alt and used later as an agreement check on field values.
"""
import time, shutil
from veridoc import config, extract


def main():
    t0 = time.time()
    for doc_id in config.DOC_FILES:
        pages = extract.process_document(doc_id)
        extract.save_pages(pages)
        alt = extract.process_document(doc_id, variant="alt")
        extract.save_pages(alt, config.PROCESSED / "pages_alt")
        n_ocr = sum(p.route == "ocr" for p in pages)
        print(f"{doc_id:20s} pages={len(pages):2d} native={len(pages)-n_ocr:2d} ocr={n_ocr:2d} sec={sum(p.seconds for p in pages):6.1f}")
    print("total sec", round(time.time() - t0, 1))

def refresh_native():
    """Re-extract only the pages that have a text layer (seconds); OCR pages are kept as they are.
    Used after a change to native text extraction, so the two OCR readings need not be redone."""
    import pdfplumber
    from veridoc.schema import Page
    for doc_id in config.DOC_FILES:
        old = extract.load_pages(doc_id)
        alt = {p.page: p for p in extract.load_pages(doc_id, in_dir=config.PROCESSED / "pages_alt")}
        new, new_alt, changed = [], [], 0
        with pdfplumber.open(config.RAW / config.DOC_FILES[doc_id]) as pdf:
            for p in old:
                if p.route != "native":
                    new.append(p); new_alt.append(alt.get(p.page, p)); continue
                text, words, n_hidden = extract.native_page(config.RAW / config.DOC_FILES[doc_id], p.page, pdf.pages[p.page - 1])
                changed += text != p.text
                q = Page(p.doc_id, p.page, p.width, p.height, p.route, p.engine, text, words, {**p.features, "n_hidden_chars": n_hidden}, p.seconds)
                new.append(q); new_alt.append(q)
        extract.save_pages(new); extract.save_pages(new_alt, config.PROCESSED / "pages_alt")
        hidden = sum(q.features.get("n_hidden_chars", 0) for q in new)
        print(f"{doc_id:20s} native pages changed: {changed}   covered (redacted) characters dropped: {hidden}")


if __name__ == "__main__":
    import sys
    refresh_native() if "--native-only" in sys.argv else main()
