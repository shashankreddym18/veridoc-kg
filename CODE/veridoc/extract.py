"""Stage 1 + 2: page router and text extraction.

Router (classical, rule-based): a page with a usable text layer goes to native
parsing (pdfplumber); a page without one goes to OCR (Tesseract 5 + OpenCV).
Both paths emit the same Page format.
"""
from __future__ import annotations
import subprocess, time, re
from pathlib import Path
import numpy as np
import cv2
import pdfplumber
import pytesseract
from PIL import Image

from . import config
from .schema import Page, Word


# ---------------------------------------------------------------- features
def is_rotated(char: dict) -> bool:
    """True for glyphs drawn at an angle (the diagonal SPECIMEN watermark).
    pdfplumber's own `upright` flag stays True for 45-degree text, so we read
    the text matrix: any shear/rotation term means the glyph is not horizontal."""
    m = char.get("matrix")
    return bool(m) and (abs(m[1]) > 1e-3 or abs(m[2]) > 1e-3)


def page_features(pl_page) -> dict:
    """Cheap per-page features the router decides on (also the feature set
    for the logistic-regression / random-forest router baseline)."""
    chars = pl_page.chars
    area = float(pl_page.width * pl_page.height)
    upright = [c for c in chars if not is_rotated(c)]
    text = "".join(c["text"] for c in chars)
    alnum = sum(ch.isalnum() or ch.isspace() for ch in text)
    img_area = sum(abs((im["x1"] - im["x0"]) * (im["bottom"] - im["top"])) for im in pl_page.images)
    return {
        "n_chars": len(chars),
        "n_upright_chars": len(upright),
        "n_rotated_chars": len(chars) - len(upright),
        "garbage_ratio": round(1 - alnum / len(text), 4) if text else 0.0,
        "image_area_ratio": round(min(img_area / area, 1.0), 4),
        "n_fonts": len({c.get("fontname") for c in chars}),
    }


def route(features: dict) -> str:
    return "native" if features["n_upright_chars"] >= config.NATIVE_MIN_CHARS else "ocr"


# ---------------------------------------------------------------- native
def _lines_from_words(words: list) -> str:
    lines, cur, last = [], [], None
    for w in words:
        if last is not None and w.line != last:
            lines.append(" ".join(cur)); cur = []
        cur.append(w.text); last = w.line
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


HIDDEN_DPI = 100


def covering_rects(pl_page) -> list:
    """White, unstroked, filled rectangles smaller than a quarter of the page: the shape a
    redaction-by-covering leaves behind (the text stays in the text layer underneath)."""
    area = float(pl_page.width) * float(pl_page.height)
    out = []
    for r in pl_page.rects:
        c = r.get("non_stroking_color")
        white = c is not None and all(abs(float(v) - 1.0) < 0.02 for v in (c if isinstance(c, (list, tuple)) else [c]))
        if r.get("fill") and not r.get("stroke") and white and 0 < (r["x1"] - r["x0"]) * (r["bottom"] - r["top"]) < 0.25 * area:
            out.append((r["x0"], r["top"], r["x1"], r["bottom"]))
    return out


def hidden_char_test(pl_page, gray: np.ndarray, dpi: int = HIDDEN_DPI):
    """Returns a function char -> True when the character is in the text layer but not visible.
    A covering rectangle counts only if at least one letter or digit under it has no ink at all in the
    rendered page (so a white panel drawn underneath visible text hides nothing); a character is hidden
    when it lies in such a rectangle and its own box is (almost) blank."""
    k = dpi / 72.0

    def dark(c):
        reg = gray[max(0, int(c["top"] * k)): int(c["bottom"] * k) + 1, max(0, int(c["x0"] * k)): int(c["x1"] * k) + 1]
        return float((reg < 200).mean()) if reg.size else 0.0

    def inside(c, r):
        return r[0] <= (c["x0"] + c["x1"]) / 2 <= r[2] and r[1] <= (c["top"] + c["bottom"]) / 2 <= r[3]

    chars = [c for c in pl_page.chars if c.get("text", "").strip()]
    rects = [r for r in covering_rects(pl_page) if any(c["text"].isalnum() and inside(c, r) and dark(c) < 0.01 for c in chars)]
    if not rects:
        return None
    return lambda c: any(inside(c, r) for r in rects) and dark(c) < 0.04


def extract_native(pl_page, drop_rotated: bool = True, hidden=None) -> tuple[str, list]:
    """pdfplumber words with boxes. Rotated glyphs (the diagonal SPECIMEN
    watermark) are dropped before words are built when drop_rotated is set.
    Underscore glyphs (the fill-in lines of a form) are always dropped: text typed
    onto a form line is otherwise interleaved with them ("_B__lu_e_w...").
    hidden: optional test from hidden_char_test(); characters it flags (covered, redacted text) are dropped."""
    def keep(o):
        if o.get("object_type") != "char":
            return True
        return o.get("text") != "_" and not (drop_rotated and is_rotated(o)) and not (hidden and hidden(o))
    pg = pl_page.filter(keep)
    raw = pg.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=False, use_text_flow=False)
    raw.sort(key=lambda w: (round(w["top"] / 4), w["x0"]))
    words, line, last_top = [], -1, None
    for w in raw:
        if last_top is None or abs(w["top"] - last_top) > 4:
            line += 1; last_top = w["top"]
        words.append(Word(w["text"], round(w["x0"], 1), round(w["top"], 1), round(w["x1"], 1), round(w["bottom"], 1), 100.0, line))
    return _lines_from_words(words), words


def native_page(pdf_path: Path, page_no: int, pl_page, image_dir: Path | None = None) -> tuple[str, list, int]:
    """Native text of one page with covered (redacted) characters removed. Returns (text, words, n_hidden_chars).
    The page is rendered only when it has a rectangle that could cover text."""
    hidden = None
    if covering_rects(pl_page):
        png = render(pdf_path, page_no, dpi=HIDDEN_DPI, out_dir=image_dir or config.PROCESSED / "images_hidden")
        hidden = hidden_char_test(pl_page, cv2.imread(str(png), cv2.IMREAD_GRAYSCALE))
    n_hidden = sum(1 for c in pl_page.chars if hidden(c)) if hidden else 0
    text, words = extract_native(pl_page, hidden=hidden)
    return text, words, n_hidden


# ---------------------------------------------------------------- OCR
def render(pdf_path: Path, page_no: int, dpi: int = config.RENDER_DPI, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or config.IMAGES
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{pdf_path.stem.replace(' ', '_')}_p{page_no:03d}_{dpi}"
    png = Path(str(stem) + ".png")
    if not png.exists():
        subprocess.run(["pdftoppm", "-r", str(dpi), "-f", str(page_no), "-l", str(page_no), "-png", "-singlefile",
                        str(pdf_path), str(stem)], check=True, capture_output=True)
    return png


def deskew(gray: np.ndarray) -> np.ndarray:
    inv = cv2.bitwise_not(gray)
    thr = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(thr)
    if coords is None:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = angle - 90 if angle > 45 else angle
    if abs(angle) < 0.2 or abs(angle) > 10:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess(img: np.ndarray) -> np.ndarray:
    """OpenCV clean-up before Tesseract: grayscale, deskew, denoise, binarize."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = deskew(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)


def ocr_image(img: np.ndarray, scale: float, prep: bool = True, psm: int = 3) -> tuple[str, list]:
    """Tesseract LSTM on an image. scale = points per pixel (72 / dpi)."""
    work = preprocess(img) if prep else img
    d = pytesseract.image_to_data(Image.fromarray(work), config=f"--oem 1 --psm {psm}", output_type=pytesseract.Output.DICT)
    words, keys, line = [], {}, -1
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if not t:
            continue
        k = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        if k not in keys:
            line += 1; keys[k] = line
        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        words.append(Word(t, round(x * scale, 1), round(y * scale, 1), round((x + w) * scale, 1), round((y + h) * scale, 1),
                          float(d["conf"][i]), keys[k]))
    return _lines_from_words(words), words


# ---------------------------------------------------------------- driver
OCR_VARIANTS = {"primary": dict(prep=True, psm=3), "alt": dict(prep=False, psm=6)}


def process_document(doc_id: str, force_route: str | None = None, variant: str = "primary") -> list:
    """Route and extract every page. variant picks the Tesseract configuration
    for OCR pages (native pages are the same in every variant)."""
    return process_pdf(config.RAW / config.DOC_FILES[doc_id], doc_id, force_route, variant)


def process_pdf(pdf_path: Path, doc_id: str, force_route: str | None = None, variant: str = "primary",
                image_dir: Path | None = None) -> list:
    cfg = OCR_VARIANTS[variant]
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, pl in enumerate(pdf.pages, 1):
            t0 = time.time()
            feats = page_features(pl)
            r = force_route or route(feats)
            if r == "native":
                text, words, feats["n_hidden_chars"] = native_page(pdf_path, i, pl, image_dir)
                engine = "pdfplumber"
            else:
                png = render(pdf_path, i, out_dir=image_dir)
                img = cv2.imread(str(png))
                text, words = ocr_image(img, 72.0 / config.RENDER_DPI, prep=cfg["prep"], psm=cfg["psm"])
                engine = f"tesseract(psm{cfg['psm']}{'+opencv' if cfg['prep'] else ''})"
            pages.append(Page(doc_id, i, float(pl.width), float(pl.height), r, engine, text, words, feats, round(time.time() - t0, 2)))
    return pages


def save_pages(pages: list, out_dir: Path | None = None) -> None:
    out_dir = out_dir or config.PAGES
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in pages:
        (out_dir / f"{p.doc_id}_p{p.page:03d}.json").write_text(p.to_json(), encoding="utf-8")


def load_pages(doc_id: str | None = None, in_dir: Path | None = None) -> list:
    files = sorted((in_dir or config.PAGES).glob(f"{doc_id}_p*.json" if doc_id else "*.json"))
    return [Page.from_json(f.read_text(encoding="utf-8")) for f in files]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()
