"""E2: OCR text fidelity against the native text layer.

The digital pages carry a text layer. We rasterize them, optionally degrade
the image (blur, noise, 1.5 degree skew, 150 DPI) and OCR the result, and use
the native text layer as the reference. The reference is NOT hand-verified:
brochure pages have spacing defects in their text layer, which inflates the
error rates. Read the numbers as relative comparisons between settings.
This measures the OCR path on clean and degraded renders of real pages; the
41 genuinely scanned pages still need hand-corrected gold (G-A).

Run from the repo root:  python EVALUATIONS/eval_text_fidelity.py
"""
import json, os, re, sys, time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
import cv2, numpy as np, jiwer
from veridoc import config, extract

CONDITIONS = {"clean_300dpi": dict(dpi=300, degrade=False), "degraded_150dpi": dict(dpi=150, degrade=True)}
ENGINES = {"tesseract": dict(prep=False, psm=3), "tesseract+opencv": dict(prep=True, psm=3)}


def norm(t):
    return re.sub(r"\s+", " ", t.lower()).strip()


def degrade(img, seed):
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 1.5, 1.0)
    img = cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    return np.clip(img.astype(np.float32) + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)


def word_f1(ref, hyp):
    r, h = Counter(ref.split()), Counter(hyp.split())
    tp = sum((r & h).values())
    p, rc = tp / max(1, sum(h.values())), tp / max(1, sum(r.values()))
    return 2 * p * rc / (p + rc) if p + rc else 0.0


def job(args):
    doc_id, page_no, ref, cond, eng = args
    c, e = CONDITIONS[cond], ENGINES[eng]
    png = extract.render(config.RAW / config.DOC_FILES[doc_id], page_no, dpi=c["dpi"], out_dir=config.PROCESSED / "images_eval")
    img = cv2.imread(str(png))
    if c["degrade"]:
        img = degrade(img, seed=page_no)
    t0 = time.time()
    text, _ = extract.ocr_image(img, 72.0 / c["dpi"], prep=e["prep"], psm=e["psm"])
    hyp = norm(text)
    return {"doc": doc_id, "page": page_no, "condition": cond, "engine": eng, "ref_chars": len(ref),
            "cer": jiwer.cer(ref, hyp) if hyp else 1.0, "wer": jiwer.wer(ref, hyp) if hyp else 1.0,
            "word_f1": word_f1(ref, hyp), "sec": round(time.time() - t0, 2)}


if __name__ == "__main__":
    pages = [p for p in extract.load_pages() if p.route == "native" and len(norm(p.text)) >= 200]
    jobs = [(p.doc_id, p.page, norm(p.text), c, e) for p in pages for c in CONDITIONS for e in ENGINES]
    with Pool(2) as pool:
        rows = pool.map(job, jobs, chunksize=2)
    agg = defaultdict(list)
    for r in rows:
        agg[(r["condition"], r["engine"])].append(r); agg[(r["condition"], r["engine"], r["doc"])].append(r)
    def summary(rs):
        n = sum(r["ref_chars"] for r in rs)                     # character-weighted means
        return {"pages": len(rs), "cer": round(sum(r["cer"] * r["ref_chars"] for r in rs) / n, 4),
                "wer": round(float(np.mean([r["wer"] for r in rs])), 4), "word_f1": round(float(np.mean([r["word_f1"] for r in rs])), 4),
                "median_cer": round(float(np.median([r["cer"] for r in rs])), 4), "sec_per_page": round(float(np.mean([r["sec"] for r in rs])), 2)}
    out = {"note": "Reference = native text layer of digital pages (watermark glyphs removed), not hand-verified. CER/WER are order-sensitive, so "
                   "multi-column pages inflate them; word_f1 is order-free. Use for relative comparison between settings.",
           "pages": len(pages), "summary": {" | ".join(k): summary(v) for k, v in sorted(agg.items())}, "rows": rows}
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "e2_text_fidelity.json").write_text(json.dumps(out, indent=1))
    for k, v in out["summary"].items():
        if k.count("|") == 1:
            print(k, v)
