"""VeriDoc-KG demo UI (Gradio).

Run from the repo root:  python CODE/app/app.py     then open http://localhost:7860
Shows, for every extracted value, where it came from on the page and which
checks it passed; lets a reviewer accept or reject quarantined facts.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image, ImageDraw

from veridoc import config, extract, pipeline, deduction, verifier
from veridoc.schema import load_facts
from veridoc.kg import LOADED, RELATIONS
from veridoc.verifier import ENTITY_OF

UI_DPI = 110
COLORS = {"verified": (22, 140, 70), "human_verified": (30, 100, 200), "quarantined": (220, 120, 0), "rejected": (150, 150, 150)}
CHECKS = ("agree", "span", "nli", "shacl", "crossdoc", "template")
MARK = {"pass": "pass", "fail": "FAIL", "skip": "-"}
REVIEW = config.PROCESSED / "review.json"


def facts():
    return load_facts(config.FACTS / "facts.json")


def doc_choices():
    return [(f"{config.DOC_TITLES[d]} ({len(extract.load_pages(d))} pages)", d) for d in config.DOC_FILES]


COLUMNS = ["document", "page", "field", "value as read", "normalized", "status", *CHECKS, "reason", "evidence span"]
WIDTH = {"document": 130, "page": 55, "field": 230, "value as read": 160, "normalized": 130, "status": 110,
         "agree": 70, "span": 70, "nli": 60, "shacl": 70, "crossdoc": 90, "template": 90, "reason": 210, "evidence span": 320}


def widths(cols):
    """Fixed pixel widths so short headers (nli, reason) are not squeezed to one letter per line."""
    return [WIDTH[c] for c in cols]


def facts_frame(fs):
    rows = [{"document": f.doc_id, "page": f.page, "field": f.field, "value as read": f.value, "normalized": f.normalized,
             "status": f.status, **{c: MARK[f.checks.get(c, {}).get("result", "skip")] for c in CHECKS},
             "reason": f.reason or "-", "evidence span": f.span[:90]} for f in fs]
    return pd.DataFrame(rows, columns=COLUMNS)


# ---------------------------------------------------------------- evidence viewer
def view_page(doc_id, page_no):
    pages = extract.load_pages(doc_id)
    page_no = int(min(max(1, page_no), len(pages)))
    p = pages[page_no - 1]
    png = extract.render(config.RAW / config.DOC_FILES[doc_id], page_no, dpi=UI_DPI, out_dir=config.PROCESSED / "images_ui")
    img = Image.open(png).convert("RGB")
    draw, k = ImageDraw.Draw(img), UI_DPI / 72.0
    here = [f for f in facts() if f.doc_id == doc_id and f.page == page_no]
    for f in here:
        if f.bbox:
            x0, y0, x1, y1 = [v * k for v in f.bbox]
            draw.rectangle([x0 - 3, y0 - 3, x1 + 3, y1 + 3], outline=COLORS.get(f.status, (0, 0, 0)), width=3)
    info = (f"**{config.DOC_TITLES[doc_id]}**, page {page_no} of {len(pages)}  \n"
            f"Route: **{p.route}** ({p.engine}), {p.seconds}s  \n"
            f"Text-layer characters: {p.features['n_chars']}; image area: {p.features['image_area_ratio']:.0%}; "
            f"watermark glyphs removed: {p.features['n_rotated_chars']}  \n"
            f"Facts on this page: {len(here)} "
            f"(green = verified, orange = quarantined, blue = human-verified)")
    return img, info, facts_frame(here).drop(columns=["document", "page"]), p.text


# ---------------------------------------------------------------- quarantine review
def quarantine_frame():
    return facts_frame([f for f in facts() if f.status in ("quarantined", "human_verified", "rejected")])


def quarantine_ids():
    return [f.fact_id for f in facts() if f.status in ("quarantined", "human_verified", "rejected")]


def review(fact_id, decision):
    d = json.loads(REVIEW.read_text()) if REVIEW.exists() else {}
    if fact_id:
        if decision == "clear":
            d.pop(fact_id, None)
        else:
            d[fact_id] = decision
        REVIEW.write_text(json.dumps(d, indent=1))
        pipeline.run()                       # re-verify, rebuild graph and deduction file
    return quarantine_frame(), summary_md(), deduction_frame(), graph_plot()


# ---------------------------------------------------------------- question box
_QA = None


MATCHERS = ["keywords", "local LLM (needs Ollama)"]


def ask(question, how=MATCHERS[0]):
    """Graph tier answers with provenance; BM25 shows the pages a reader would look at."""
    global _QA
    if not question or not question.strip():
        return "", None
    if _QA is None:
        from veridoc.qa import QA
        _QA = QA(facts())
    note = ""
    if how == MATCHERS[1]:
        try:
            h = _QA.hybrid(question, 3, "llm")
        except OSError:                              # Ollama is not running
            h, note = _QA.hybrid(question, 3), "_The local LLM is not reachable (start Ollama), so keywords were used._  \n"
    else:
        h = _QA.hybrid(question, 3)
    if h["answer"] is None:
        md = note + f"**No verified answer.** {h['why']}" + (f" (matched field: `{h['field']}`)" if h["field"] else "")
    else:
        ev = ", ".join(f"{config.DOC_TITLES[d]} p{p}" for d, p in h["evidence"])
        md = note + (f"**{h['answer']}**  \nfield `{h['field']}`, status {', '.join(h['status'])}  \nevidence: {ev}  \n"
              "_Answers come only from verified facts. The question is matched to one field, and the answer is that field's "
              "verified value: a wrong match gives the wrong field, never an invented value (see E9)._")
    rows = [{"document": config.DOC_TITLES[d], "page": p, "excerpt": _QA.page_text[(d, p)][:220].replace("\n", " ")} for d, p in h["pages"]]
    return md, pd.DataFrame(rows)


# ---------------------------------------------------------------- graph, deduction, summary
def graph_plot():
    fs = [f for f in facts() if f.status in LOADED and f.field in ENTITY_OF]
    g = nx.Graph()
    ents = sorted({ENTITY_OF[f.field] for f in fs})
    for e in ents:
        n = len({f.field for f in fs if ENTITY_OF[f.field] == e})
        g.add_node(e, kind="entity", label=f"{e}\n{n} properties")
    for a, rel, b in RELATIONS:
        if a in ents and b in ents:
            g.add_edge(a, b, label=rel)
    for d in sorted({f.doc_id for f in fs}):
        g.add_node(d, kind="doc", label=config.DOC_TITLES[d])
        for e in ents:
            n = sum(1 for f in fs if f.doc_id == d and ENTITY_OF[f.field] == e)
            if n:
                g.add_edge(d, e, label=f"{n} evidence")
    fig, ax = plt.subplots(figsize=(10, 5.6))
    kinds = nx.get_node_attributes(g, "kind")
    slots = {"Group": (-0.55, 0.45), "Certificate": (0.0, 0.0), "Insured": (0.55, 0.45), "Rider": (0.0, -0.75)}
    pos = {e: slots.get(e, (0.0, 0.8)) for e in ents}
    docs = [n for n in g if kinds[n] == "doc"]
    doc_slots = [(-1.25, 0.1), (1.25, 0.1), (-1.0, -0.7), (1.0, -0.7), (0.0, 0.95), (-1.25, 0.8)]
    pos.update({d: doc_slots[i % len(doc_slots)] for i, d in enumerate(docs)})
    nx.draw_networkx_nodes(g, pos, nodelist=ents, node_color="#8C1D40", node_size=4600, ax=ax)
    nx.draw_networkx_nodes(g, pos, nodelist=docs, node_color="#FFC627", node_size=3800, node_shape="s", ax=ax)
    nx.draw_networkx_edges(g, pos, edge_color="#999999", ax=ax)
    nx.draw_networkx_labels(g, pos, labels={n: g.nodes[n]["label"] for n in g if kinds[n] == "entity"}, font_size=8, font_color="white", ax=ax)
    nx.draw_networkx_labels(g, pos, labels={n: g.nodes[n]["label"] for n in g if kinds[n] == "doc"}, font_size=7, font_color="black", ax=ax)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=nx.get_edge_attributes(g, "label"), font_size=7, ax=ax)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.05, 1.2); ax.set_axis_off(); fig.tight_layout()
    return fig


def deduction_frame():
    cols, row, src = deduction.fill(facts())
    return pd.DataFrame([{"column": c, "value": row[c],
                          "source": f"{src[c]['doc_id']} p{src[c]['page']} ({src[c]['status']})" if c in src else "no supporting fact: left empty"}
                         for c in cols])


def summary_md():
    s = json.loads((config.PROCESSED / "run_summary.json").read_text())
    st, kg, de = s["status"], s["kg"], s["deduction"]
    return (f"**Pages:** {s['pages']} ({s['ocr_pages']} routed to OCR) | **Candidate facts:** {s['candidates']} | "
            f"**Verified:** {st.get('verified', 0)} | **Quarantined:** {st.get('quarantined', 0)} | "
            f"**Human-verified:** {st.get('human_verified', 0)} | **Rejected:** {st.get('rejected', 0)}  \n"
            f"**Graph:** {kg['entities']} entities, {kg['property_assertions']} property assertions, {kg['evidence_nodes']} evidence nodes, "
            f"provenance completeness {kg['provenance_completeness']:.0%} | **Deduction file:** {de['filled']} of {de['columns']} columns filled  \n"
            f"**NLI check:** {s['nli']}")


def eval_tables():
    out = {}
    r = config.RESULTS
    if (r / "e3_fields.json").exists():
        e3 = json.loads((r / "e3_fields.json").read_text())["systems"]
        out["fields"] = pd.DataFrame([{"system": k, **{m: v["all"][m] for m in ("emitted", "correct", "wrong", "precision", "recall", "f1", "values_on_null_fields")}} for k, v in e3.items()])
    if (r / "e5_verifier_stress.json").exists():
        out["stress"] = pd.DataFrame(json.loads((r / "e5_verifier_stress.json").read_text())["rows"])
    for which in ("test", "dev"):
        f = r / f"e7_synthetic_{which}.json"
        if f.exists():
            e7 = json.loads(f.read_text())["systems"]
            out[f"synthetic_{which}"] = pd.DataFrame([{"system": k, **{f"{c} P/R": f"{v[c]['precision']:.2f} / {v[c]['recall']:.2f}" for c in ("native", "scan300", "scan150", "all")},
                                                       "values on empty fields": v["all"]["values_on_empty_fields"]} for k, v in e7.items()])
    if (r / "e8_agree_policy.json").exists():
        e8 = json.loads((r / "e8_agree_policy.json").read_text())
        out["agree"] = pd.DataFrame([{"data": k, "policy": p, "precision": v["precision"], "recall": v["recall"], "wrong values loaded": v["wrong"],
                                      "quarantined": v["quarantined"], "correct among quarantined": v["quarantined_but_correct"]}
                                     for k, d in e8.items() for p, v in d.items()])
    if (r / "e9_qa.json").exists():
        e9 = json.loads((r / "e9_qa.json").read_text())
        out["qa"] = pd.DataFrame([{"tier": k, **v} for k, v in e9["retrieval"].items()])
        sets = {"original wording": e9}
        if (r / "e9_qa_human.json").exists():
            sets["a teammate's wording"] = json.loads((r / "e9_qa_human.json").read_text())
        rows = []
        for name, e in sets.items():
            ans = {f"graph, field by {m}": v for m, v in e.get("graph_answers_by_matcher", {"bm25": e["graph_answers"]}).items()}
            if "llm_answers" in e:
                ans["LLM reader"] = e["llm_answers"]
            for system, v in ans.items():
                rows.append({"questions": name, "system": system,
                             "single-fact correct": f"{v.get('single', {}).get('correct', 0)} of {sum(v.get('single', {}).values())}",
                             "multi-hop correct": f"{v.get('multi', {}).get('correct', 0)} of {sum(v.get('multi', {}).values())}",
                             "unanswerable declined": f"{v.get('unanswerable', {}).get('abstained_correct', 0)} of {sum(v.get('unanswerable', {}).values())}"})
        out["qa_answers"] = pd.DataFrame(rows)
    if (r / "e2b_docling.json").exists():
        e2b = json.loads((r / "e2b_docling.json").read_text())
        out["scans"] = pd.DataFrame([{"engine": k, "CER": v["cer"], "WER": v["wer"], "words found": v.get("word_recall_any_order"),
                                      "fields correct": f"{e2b['fields'][k]['correct']} of {e2b['fields'][k]['gold_values']}",
                                      "wrong values": e2b["fields"][k]["wrong"], "s/page": e2b["fields"][k]["sec_per_page"]}
                                     for k, v in e2b["text"].items()])
    if (r / "e12_canonical.json").exists():
        e12 = json.loads((r / "e12_canonical.json").read_text())["summary"]
        out["identity"] = pd.DataFrame([{"compare by": k, "same value called a conflict": f"{v['false_conflicts']} of {v['same_pairs']}",
                                         "different values called the same": f"{v['missed_conflicts']} of {v['different_pairs']}"}
                                        for k, v in e12.items()])
    if (r / "e11_label_agreement.json").exists():
        e11 = json.loads((r / "e11_label_agreement.json").read_text())
        out["agreement"] = (f"**Second annotator (E11):** {e11['items_labeled_by_both']} fields labeled blind by both. Value present or not: "
                            f"{e11['has_value_agreement']:.0%} agreement (kappa {e11['has_value_kappa']:.2f}); same value: "
                            f"{e11['same_value']} of {e11['both_gave_a_value']}.")
    if (r / "e10_neo4j.json").exists():
        e10 = json.loads((r / "e10_neo4j.json").read_text())
        out["neo4j"] = (f"**Neo4j (E10):** {sum(e10['nodes_by_label'].values())} nodes, {sum(e10['relationships_by_type'].values())} relationships; "
                        f"provenance completeness {e10['provenance_completeness']:.0%} ({e10['server']}).")
    if (r / "e2_text_fidelity.json").exists():
        e2 = json.loads((r / "e2_text_fidelity.json").read_text())["summary"]
        out["text"] = pd.DataFrame([{"condition | engine": k, **v} for k, v in e2.items() if k.count("|") == 1])
    return out


# ---------------------------------------------------------------- layout
with gr.Blocks(title="VeriDoc-KG") as demo:
    gr.Markdown("# VeriDoc-KG\nEvidence-verified extraction from insurance workflow PDFs. Group 30, CSE 573.")
    summary = gr.Markdown(summary_md())
    with gr.Tab("Evidence viewer"):
        with gr.Row():
            doc = gr.Dropdown(doc_choices(), value="certificate", label="Document")
            page = gr.Slider(1, 36, value=5, step=1, label="Page")
        with gr.Row():
            with gr.Column(scale=5):
                img = gr.Image(type="pil", label="Page with evidence boxes", height=820)
            with gr.Column(scale=5):
                info = gr.Markdown()
                page_facts = gr.Dataframe(label="Facts on this page and their checks (value as read = OCR text; normalized = what is checked and loaded)",
                                          wrap=True, column_widths=widths(COLUMNS[2:]), pinned_columns=1)
                text = gr.Textbox(label="Extracted text", lines=10, max_lines=10)
        for c in (doc, page):
            c.change(view_page, [doc, page], [img, info, page_facts, text])
        demo.load(view_page, [doc, page], [img, info, page_facts, text])
    with gr.Tab("All facts"):
        gr.Dataframe(facts_frame(facts()), wrap=True, column_widths=widths(COLUMNS), pinned_columns=3,
                     label="Every candidate fact with the result of each check")
    with gr.Tab("Quarantine review"):
        gr.Markdown("A fact that fails a check is never deleted. It waits here with its reason code. "
                    "**Accept** loads it into the graph flagged *human-verified*; **Reject** discards it.")
        qtable = gr.Dataframe(quarantine_frame(), wrap=True, column_widths=widths(COLUMNS), label="Quarantine queue")
        with gr.Row():
            fid = gr.Dropdown(quarantine_ids(), label="Fact", scale=3)
            acc = gr.Button("Accept (human-verified)", variant="primary")
            rej = gr.Button("Reject")
            clr = gr.Button("Clear decision")
    with gr.Tab("Knowledge graph"):
        gplot = gr.Plot(graph_plot(), label="Entities (maroon), source documents (gold) and evidence counts")
        gr.Markdown("Full graph: `DATA/processed/kg/kg.ttl` (RDF) and `kg.cypher` (Neo4j). Every property has at least one Evidence node.")
    with gr.Tab("Deduction file"):
        dtable = gr.Dataframe(deduction_frame(), wrap=True, label="22-column payroll-deduction record")
        gr.File(str(config.PROCESSED / "deduction" / "deduction_file.csv"), label="Download CSV")
    with gr.Tab("Ask"):
        gr.Markdown("Ask about the case. The answer comes from verified facts in the graph, with the page it came from; "
                    "the table shows the pages a keyword search finds for the same question.")
        with gr.Row():
            q = gr.Textbox(label="Question", placeholder="What is the face amount of the employee's certificate?", scale=5)
            go = gr.Button("Ask", variant="primary", scale=1)
        how = gr.Radio(MATCHERS, value=MATCHERS[0], label="Match the question to a field with",
                       info="Keywords need the question to use the forms' own words. The local LLM picks a field from the ontology's property list; it never writes the answer.")
        answer = gr.Markdown()
        qpages = gr.Dataframe(label="Pages retrieved (graph evidence first, then BM25)", wrap=True)
        gr.Examples(["What is the face amount of the employee's certificate?", "What is the loan interest rate on the certificate?",
                     "What is the insured's name?", "What is the situs state of the group policy?"], inputs=q)
        go.click(ask, [q, how], [answer, qpages]); q.submit(ask, [q, how], [answer, qpages])
    with gr.Tab("Evaluations"):
        tabs = eval_tables()
        if "fields" in tabs:
            gr.Dataframe(tabs["fields"], label="Field extraction on the real documents vs. the field labels", wrap=True)
        for k in ("agreement", "neo4j"):
            if k in tabs:
                gr.Markdown(tabs[k])
        if "synthetic_test" in tabs:
            gr.Dataframe(tabs["synthetic_test"], label="Held-out synthetic forms (test set, generated after the rules were frozen): precision / recall", wrap=True)
        if "agree" in tabs:
            gr.Dataframe(tabs["agree"], label="Agreement-check policy: wrong values loaded vs correct values sent to review", wrap=True)
        if "qa" in tabs:
            gr.Dataframe(tabs["qa"], label="Question set: evidence recall@k by retrieval tier", wrap=True)
            gr.Dataframe(tabs["qa_answers"], label="Question set: answers, original wording and a teammate's wording", wrap=True)
        if "identity" in tabs:
            gr.Dataframe(tabs["identity"], label="Cross-document agreement under surface variation (E12)", wrap=True)
        if "scans" in tabs:
            gr.Dataframe(tabs["scans"], label="Scanned pages: Tesseract vs Docling against hand-checked transcriptions (E2b)", wrap=True)
        if "text" in tabs:
            gr.Dataframe(tabs["text"], label="OCR text fidelity on rasterized digital pages", wrap=True)
        if "stress" in tabs:
            gr.Dataframe(tabs["stress"], label="Verifier stress test: span-match threshold", wrap=True)
    outs = [qtable, summary, dtable, gplot]
    acc.click(lambda i: review(i, "accept"), fid, outs)
    rej.click(lambda i: review(i, "reject"), fid, outs)
    clr.click(lambda i: review(i, "clear"), fid, outs)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
