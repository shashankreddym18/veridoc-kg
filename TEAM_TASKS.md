# Who leads what

Everything plugs into the two shared formats in `CODE/veridoc/schema.py` (Page and Fact).
Read the code in your module before the demo: you will be asked about it.
Results in `EVALUATIONS/results/` come from one laptop run of `run_laptop.sh` (Nov 17).

| Module lead | Module | Main files | Results |
|---|---|---|---|
| Shashank | Coordination, ontology and identity, LLM extraction, labels, laptop runs | `config.py`, `schema.py`, `pipeline.py`, `ontology.py`, `keys.py`, `fields_llm.py`, `CODE/ontology/`, `CODE/app/annotate.py`, `CODE/app/reword.py`, `DATA/gold/fields_gold_*.json`, `run_laptop.sh` | E11, E12 |
| Adnan | Page routing and text extraction | `extract.py`, `run_extract.py`, `docling_extract.py`, `DATA/gold/text/` | E1, E2, E2b |
| Ananya | Classical extraction, synthetic forms, deduction file | `fields.py`, `crf.py`, `synthetic.py`, `deduction.py`, `DATA/synthetic/` | E7 |
| Pradeep | Verifier, graph and Neo4j | `verifier.py`, `judge.py`, `kg.py`, `load_neo4j.py`, `CODE/ontology/shapes.ttl`, `run_evals.py` | E3-E6, E5b, E5c, E8, E10 |
| Mokshitha | Retrieval, questions, UI | `qa.py`, `qa_router.py`, `qa_llm.py`, `qa_paraphrase.py`, `CODE/app/app.py`, `DATA/gold/qa_gold_draft.json` | E9 |

## Open work (for the final report or later)

- More hand-checked transcriptions of scanned pages (2 of 41 so far), starting with master application p1-3
  and certificate p31-32.
- The group name on the master application and on the synthetic scans is often lost by OCR reading 1 (it
  sits on an underline). Try cropping the field region before OCR.
- The LLM extractor's remaining wrong values are wrong-entity errors (a form number taken as the
  certificate number). Neither NLI nor the LLM judge stops them. Try a section check: the evidence must sit
  in the section of the form that the field belongs to.
- `eligible_employees` has a one-token evidence span ("27"); widen it so NLI and the judge have context.
- Try one larger or different model (`MODEL=qwen2.5:7b bash run_laptop.sh`) to see how much the LLM rows move.
- FUNSD as an outside labeled check for the CRF.
