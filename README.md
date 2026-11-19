# VeriDoc-KG

Ontology-constrained, evidence-verified knowledge extraction from insurance workflow PDFs.
CSE 573 Semantic Web Mining, Fall 2026, Group 30, Project 14 (PDF/OCR Processing).

A value is loaded into the knowledge graph only if the system can point to the place on the
page that supports it and the value passes every check. A value that fails is quarantined with
a reason code, never deleted, and a reviewer can restore it.

## Pipeline

| Step | What it does | Classical method (running) | Modern method (status) |
|---|---|---|---|
| 1. Route | Decide per page: native text or OCR | Rule on text-layer features | Image-embedding classifier (to do) |
| 2. Extract text | Words with bounding boxes | pdfplumber; Tesseract 5 + OpenCV, two readings per scanned page | Docling (`docling_extract.py`, compared in E2b) |
| 3. Extract fields | Typed values with evidence spans | Label-anchored rules with a type gate; CRF trained on synthetic forms; naive proximity baseline | LLM JSON output (`fields_llm.py`, llama3.1:8b through Ollama) |
| 4. Verify | agree, span, nli, shacl, crossdoc, template | OCR agreement (three policies), span match, SHACL shapes, canonical-form agreement and arithmetic across documents, blank-form label check | NLI cross-encoder (advisory), LLM second opinion (`judge.py`) |
| 5. Output | Graph + deduction file + UI | RDF/Turtle with canonical entity keys, Neo4j load, 22-column CSV, Gradio UI with a question box (keyword or embedding field match) | LLM field router (`qa_router.py`); LLM reader (`qa_llm.py`) |

## Layout

```
CODE/veridoc/        schema.py (Page and Fact formats), extract.py, fields.py, crf.py, fields_llm.py,
                     verifier.py, keys.py (canonical forms, entity keys), judge.py, kg.py, ontology.py,
                     deduction.py, pipeline.py, synthetic.py, qa.py, qa_router.py, qa_llm.py,
                     qa_paraphrase.py, docling_extract.py
CODE/ontology/       ONTOLOGY.md (the ontology document), ontology.ttl (generated), shapes.ttl (SHACL)
CODE/app/app.py      Gradio demo UI
CODE/app/annotate.py blind second-annotation screen for the field labels (port 7861)
CODE/app/reword.py   write the question set in your own words (terminal)
CODE/run_extract.py  routing + text extraction for all documents
CODE/load_neo4j.py   load the Cypher export into Neo4j and query integrity numbers
DATA/raw/Insurance/  the seven course-provided files (not committed, see below)
DATA/gold/           field labels (two passes), question set and its human rewording, page transcriptions
DATA/synthetic/      synthetic filled forms: gold.json committed, PDFs and page JSON regenerated
DATA/processed/      generated: page JSON, facts, graph, deduction file
EVALUATIONS/         run_evals.py (E1, E3-E6), eval_text_fidelity.py (E2), eval_synthetic.py (E7),
                     eval_agree_policy.py (E8), eval_qa.py (E9), eval_nli.py (E5b), eval_judge.py (E5c),
                     eval_docling.py (E2b), load_neo4j.py output (E10), eval_agreement.py (E11),
                     eval_canonical.py (E12), results/, screenshots/
```

## Run

```bash
# Python 3.10 or newer (the UI needs Gradio 6); also needs the tesseract and poppler binaries
pip install -r requirements.txt
# put the course's Insurance folder at DATA/raw/Insurance/
cd CODE && python run_extract.py         # about 4 minutes on 2 cores (41 pages are OCR'd twice)
python -m veridoc.pipeline               # candidates -> verifier -> graph + deduction file
cd .. && python EVALUATIONS/run_evals.py
python EVALUATIONS/eval_text_fidelity.py # about 4 minutes
python CODE/app/app.py                   # UI at http://localhost:7860

# synthetic forms, CRF, policy comparison, questions (about 20 minutes, mostly OCR)
cd CODE && python -m veridoc.synthetic --set dev && python -m veridoc.synthetic --set test && cd ..
python EVALUATIONS/eval_synthetic.py --set dev     # extracts and caches the dev forms
cd CODE && python -m veridoc.crf && cd ..          # trains the CRF on the dev forms
python EVALUATIONS/eval_synthetic.py --set dev && python EVALUATIONS/eval_synthetic.py --set test
python EVALUATIONS/eval_agree_policy.py && python EVALUATIONS/eval_qa.py
python EVALUATIONS/run_evals.py                    # again, so E3/E4 include the CRF
```

### Steps that need a laptop (model downloads, a local LLM)

```bash
brew install ollama                                  # https://ollama.com
caffeinate -i bash run_laptop.sh                     # everything below, in order, logged; safe to re-run
```

`run_laptop.sh` runs the pipeline, the ontology check, the NLI evaluation (downloads about 0.7 GB once),
the synthetic forms, the CRF, E7-E9, E11, E12, then pulls `llama3.1:8b` and runs the LLM extractor, the LLM
field router and the LLM reader, Docling (E2b) and, if Neo4j is installed, the graph load (E10), and re-runs
the evaluations so their rows appear. A failed step is logged and the script goes on.
Log: `DATA/processed/laptop_run.log`; one line per step: `DATA/processed/laptop_status.txt`.

Two steps need a person and are not in the script: `python CODE/app/annotate.py` (blind second labels, E11)
and `python CODE/app/reword.py` (the questions in your own words, E9 on `qa_human.json`).

The NLI check is **advisory** by default (`NLI_MODE` in `verifier.py`): when the model is installed its
result is recorded and shown, but it does not quarantine a fact. See findings 14 and 15 for why.

```bash
brew install neo4j          # run_laptop.sh then starts it, sets the password and loads the graph
```

## Findings (Nov 17, 2026)

Every number is produced by the scripts above and stored in `EVALUATIONS/results/` (run on one laptop:
macOS, Tesseract 5.5, llama3.1:8b through Ollama, cross-encoder/nli-deberta-v3-base, Docling 2.150, Neo4j 2026.10).

1. **Routing** (`e1`). 72 pages: 31 have a usable text layer, 41 (57%) need OCR.
2. **Field extraction on the real documents, 32 labeled values + 18 fields that must stay empty** (`e3`):

   | System | Emitted | Correct | Precision | Recall | Values on empty fields |
   |---|---|---|---|---|---|
   | Naive proximity baseline | 64 | 14 | 0.22 | 0.44 | 28 |
   | LLM (llama3.1:8b), JSON output | 55 | 19 | 0.35 | 0.59 | 11 |
   | Typed rules, one OCR reading | 30 | 29 | 0.97 | 0.91 | 0 |
   | Proximity + verifier | 14 | 13 | 0.93 | 0.41 | 1 |
   | LLM + verifier | 17 | 13 | 0.76 | 0.41 | 2 |
   | Typed rules + verifier (pipeline) | 29 | 29 | 1.00 | 0.91 | 0 |

   The verifier roughly doubles the LLM's precision and quadruples the naive baseline's. The CRF (trained
   only on synthetic application forms) covers the application fields only: 5 values, all correct.
3. **Blank-form test** (`e4`). Every value extracted from the blank employee application is false by
   construction. Naive proximity emits 18 values, the LLM 6, typed rules and the CRF 0. After the verifier
   none is loaded. The LLM's "$2,500", read from an instruction sentence, now conflicts with the
   certificate's $10,000 and both go to review: the check cannot tell which one is right, so the correct
   certificate value is held back too.
4. **Verifier outcome on the real documents** (`e6`). 36 candidate facts, 34 verified, 2 quarantined by the
   OCR-agreement check. One is a true rejection: a signature date read as 2/20/2022 where the page says
   12/20/2022. One is a false rejection: a correct effective date that only one OCR reading captured.
   The review queue restores it as human-verified.
5. **Span-match threshold** (`e5`). With fuzzy span matching alone at threshold 90, a one-character
   fabricated value is caught 3% of the time. Requiring the value itself to appear verbatim on the page
   raises that to 97% with no false rejections under real OCR variation. Exact span matching catches 100%
   but falsely rejects 23% of true facts. A value attached to the wrong field is caught only 63% of the time.
6. **OCR text fidelity** (`e2`, 29 digital pages rasterized, native text layer as reference).
   On clean 300 DPI renders OpenCV preprocessing does not help (word F1 0.87 without, 0.86 with) and costs
   3.5x the time. On degraded 150 DPI renders it helps (0.69 to 0.81).
7. **Scanned pages, Tesseract vs Docling** (`e2b`, 2 certificate pages transcribed by hand and checked by a
   second person). Character error rate: Tesseract 1.2% and 1.5% (two readings), Docling 6.3%. Docling
   finds 89% of the page's words (Tesseract 99%). Fields read from each engine's text, on the 39 scanned
   pages: Tesseract 25-26 of 28 values correct, Docling 24 of 28, all with no wrong value, at 1.5-4.1 s
   per page. Docling's first run scored 23% because the extractor kept only body text: it dropped the
   premium table on certificate page 5. Reading table cells, page headers and footers too brings it to 6%.
8. **Deduction file.** 7 of 22 columns filled, all 7 correct. The other 15 are redacted or absent in the
   de-identified set, so they are left empty.
9. **Watermark.** The policy specimen's diagonal SPECIMEN watermark leaks stray letters into the text layer.
   pdfplumber's `upright` flag does not mark 45-degree glyphs; reading the glyph matrix removes all 64 of them.
10. **Cross-document observation.** The sample certificate (Aug 1, 2022) predates the master application's
    proposed effective date (Feb 1, 2023): the provided documents are examples from different cases.
11. **Held-out synthetic forms** (`e7`). 20 generated personas typed onto the blank application, each in
    three versions (native PDF, 300 dpi scan, degraded 150 dpi scan), about 20% of fields left empty.
    The rules as first written (on the real certificate) found only 36% of the values on the dev forms
    (`e7_rules_v1_on_dev.json`): they had overfit the one filled example. After fixing them on the dev set
    and freezing them, a test set generated with a different seed gives:

    | System (test set, 60 forms) | Precision | Recall | Values on empty fields |
    |---|---|---|---|
    | Naive proximity | 0.14 | 0.17 | 93 |
    | LLM (llama3.1:8b) | 0.74 | 0.68 | 57 |
    | CRF (trained on dev) | 0.92 | 0.85 | 0 |
    | Typed rules v2 | 0.99 | 0.79 | 0 |
    | Proximity + verifier | 0.83 | 0.10 | 0 |
    | LLM + verifier | 0.94 | 0.51 | 5 |
    | CRF + verifier | 1.00 | 0.66 | 0 |
    | Typed rules v2 + verifier | 1.00 | 0.54 | 0 |

    Native PDFs are nearly solved (rules 0.93, CRF 0.99, LLM 1.00 recall); scans are where values are lost.
    The LLM fills empty fields (57 values) where the typed extractors do not.
12. **False negatives of the agreement check** (`e8`), the question from the proposal review. With the
    strict policy a value that only one OCR reading produced is quarantined. On the synthetic test set that
    loads no wrong value and sends 117 values to review, 112 of them correct. Accepting a value when the
    other reading's text also contains it (corroborate) cuts the queue to 68 with no wrong value on the
    synthetic forms, but on the real documents it loads the misread signature date, because the second
    reading also contains "2/20/2022" (it read "1 2/20/2022"). Accepting unless the readings conflict
    leaves 6 in review and loads 2 wrong values. Strict stays the default: false negatives go to a review
    queue where they are restored, never silently dropped.
13. **Questions** (`e9`, 31 questions: 22 single-fact, 5 multi-hop, 4 unanswerable). The graph tier maps a
    question to one ontology field and answers with that field's verified value, so it cannot invent a value.

    | Single-fact questions answered correctly (of 22) | Original wording | A teammate's wording (29 questions) |
    |---|---|---|
    | Graph, field chosen by keywords (BM25) | 15 | 10 |
    | Graph, field chosen by embeddings (MiniLM) | 16 | 13 |
    | Graph, field chosen by the LLM from the ontology's property list | 19 | 19 |
    | LLM reader over the verified facts and the top 3 pages | 21 | 18 |

    Evidence page in the top 5: BM25 59% → 22%, embeddings 48% → 30%, graph provenance + BM25 85% → 52%.
    Keyword matching depends on the question using the form's own words; the LLM router does not. Multi-hop
    questions (sums, comparisons) need two fields: the graph tier gets at most 1 of 5, the LLM reader 4 of 5.
    Unanswerable questions: all 4 declined with the original wording; with the teammate's wording the LLM
    reader answered 1 of 2 anyway.
14. **Rewording questions with the same small LLM does not work** (`qa_paraphrase.py`). Asked for everyday
    wording, llama3.1:8b changed the meaning of 5 of 31 questions ("date of birth" became "age at the time of
    application", "certificate" became "group policy"). With a same-meaning check by the same model added,
    it rejected all 31 rewrites, including ones it described as "same meaning". The reworded set in finding
    13 is written by a team member instead (`CODE/app/reword.py`).
15. **Native PDF text.** Text typed onto a form line comes out interleaved with the underscore glyphs of
    the line ("_B__lu_e_w...") unless underscores are dropped before words are built.
16. **NLI rejects true facts, and wording matters** (`e5b`, 34 verified facts, 34 one-character
    fabrications, 16 same-page misattributions). As first written (hypothesis "The <field name> is <value>",
    premise = the evidence line) the cross-encoder rejected 9 of the 34 true facts (27%): the line
    "Termination Date August 1, 2061" does not entail "The rider termination date is ...". Using the label
    as printed on the form in the hypothesis brings that to 4 (12%). At that setting it catches 85% of
    fabrications and 100% of misattributions, which is the case the span check misses (63%). Because it
    still rejects true facts, the NLI result is recorded but does not quarantine (`NLI_MODE = "advisory"`).
17. **LLM second opinion** (`e5c`). Asking the LLM about every claim rejects 1 of 34 true facts (3%) and
    catches 97% of fabrications and 94% of misattributions. Asking it only about NLI's rejections (cascade)
    also leaves 1 true fact rejected, with 49 LLM calls instead of 84, but then catches no more
    fabrications than NLI did (85%). Neither helps with the LLM extractor's own mistakes: of its 4 wrong
    values that pass the hard checks on the real documents, NLI flags 1 (and loses 1 correct value) and the
    LLM judge none. They are wrong-entity errors (a form number taken as the certificate number, the
    premium frequency taken as the payroll frequency) whose evidence line does state the value.
18. **Labels read as values** (`template` check). A free-text "value" on the application form that is
    printed on the blank form is a label, not an answer ("Job Title/Occupation", "State Zip Code"). This
    check removes the false values that used to survive on the blank form and every value the naive
    baseline put on an empty field of the synthetic forms (8 to 0), and loses no correct value.
19. **The "de-identified" proposal still carries the covered text.** On the proposal cover, part of the
    group name and of the presenter's name is hidden under white rectangles, but the words are still in
    the PDF text layer, and a text-layer extractor reads them. Native extraction now drops a character
    when it lies under a white, unstroked rectangle and has no ink in the rendered page (29 characters on
    one page of this set; `covered_chars_dropped` in `e1`). The labels were corrected to what is visible.
    The covering is also incomplete: the full group name is printed, uncovered, on page 12 of the same
    proposal. The presenter's surname appears only under the cover box.
20. **Ontology** (`CODE/ontology/ONTOLOGY.md`, `ontology.ttl`). 8 classes, 7 object properties and 43
    datatype properties, generated from the tables the pipeline uses. `python -m veridoc.ontology --check`
    verifies that the graph, the SHACL shapes and the deduction mapping use only declared terms, inside
    their domains and ranges. Writing that check exposed a naming collision in the first graph export
    (the certificate document and the certificate entity shared one identifier); individuals now live in
    their own namespace.
21. **Identity by fixed rules, not similarity** (`keys.py`, `e12`; from the instructor's feedback on the
    presentation). Values are compared in one canonical form (dates, money, states, frequencies, names with
    legal suffixes removed), and each entity gets a key built from such values (group number, else name +
    state; certificate number, else group + insured + date). Personal identifiers are only kept as a keyed
    hash. On 640 generated pairs of the same value printed two ways and of near-miss values:

    | Comparing | Same value called a conflict | Different values called the same |
    |---|---|---|
    | Printed text | 296 of 400 (74%) | 0 of 240 |
    | Similarity (the earlier rule, token-set ratio ≥ 90) | 12 of 400 (3%) | 80 of 240 (33%), e.g. $10,000 = $100,000 |
    | Canonical forms (now) | 0 of 400 | 0 of 240 |

    The variants were generated from the same formats the canonical rules handle, so this shows those rules
    work, not that they cover every format. In the de-identified set only the Group gets a key
    (`group:quest|OR`): the certificate number and the insured's name are blacked out, so those stay unmerged.
22. **Second annotator** (`e11`). A second team member labeled the 50 fields blind, from the page images.
    Agreement on whether a field has a value: 50 of 50 (Cohen's kappa 1.0). Agreement on the value: 31 of
    32. The one disagreement is a slip (the amount next to the payroll frequency typed as the frequency);
    the adjudicated label is the frequency printed on the page.
23. **Neo4j** (`e10`). The Cypher export loads into Neo4j 2026.10 as 47 nodes and 79 relationships; every
    one of the 29 property values has at least one Evidence node and every Evidence node a document
    (provenance completeness 1.0).

### Limits of these numbers

- The first field labels, the question set and the rules come from the same person, who looked at the
  same documents. On the real documents these are development-set numbers; the synthetic test set is the
  only held-out evaluation, and its scans are cleaner than real ones.
- The second annotator and the person who rewrote the questions are team members who had seen the
  documents before; they did not see the first labels or the system's output while doing it.
- The NLI wording, the template check and the case-document scope of the agreement check were chosen
  after looking at failures on the real documents, so their numbers there are not held-out either.
- Only 2 of the 41 scanned pages have a hand-checked transcription, so E2b rests on about 450 words.
- One small LLM (llama3.1:8b, temperature 0, one prompt) stands in for "LLM extraction"; a larger model or
  a better prompt would change its rows.
- The set is one small case (72 pages), so every rate above rests on a few dozen items.

## Data note

`DATA/raw/` and `DATA/processed/` are in `.gitignore`. The proposal states that the de-identified documents
stay on ASU or local machines; decide as a team (and with the instructor) before committing them.
