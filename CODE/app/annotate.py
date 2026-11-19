"""Second-annotator screen for the field labels (blind: the first annotator's values are never shown).

Run from the repo root:  python CODE/app/annotate.py     then open http://localhost:7861
For each field: look at the page, type the value exactly as the page shows it, press Save.
If the form has no value there (blank, redacted, crossed out), press "No value on the page".
Answers go to DATA/gold/fields_gold_second.json; EVALUATIONS/eval_agreement.py compares the two passes.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gradio as gr

from veridoc import config, extract
from veridoc.fields import FIELD_TYPES, norm_value
from veridoc.ontology import LABEL

FIRST = config.GOLD / "fields_gold_draft.json"
SECOND = config.GOLD / "fields_gold_second.json"
ITEMS = [(d, f) for d, f, _ in json.loads(FIRST.read_text())["items"]]      # document and field only; values are not loaded
# Where the field's label is printed (page numbers only: they say where to look, not what the value is).
PAGE = {"proposal": 1, "master_app": 1, "certificate": 5, "employee_app_blank": 1}
PAGE_OF = {("master_app", "signature_date"): 3, ("certificate", "payroll_frequency"): 32, ("certificate", "payroll_deduction_amount"): 32,
           ("employee_app_blank", "face_amount"): 2, ("employee_app_blank", "payroll_frequency"): 2,
           ("employee_app_blank", "payroll_deduction_amount"): 2,
           **{("certificate", f): 31 for f in ("ee_city", "ee_state", "ee_zip", "date_of_hire", "annual_salary", "job_title", "group_number")}}
HOW = {"date": "a date, as written", "money": "an amount, as written", "int": "a whole number", "percent": "a percentage",
       "state": "a state (name or two letters)", "zip": "a 5-digit zip code", "freq": "a frequency, e.g. Monthly"}


def load():
    return json.loads(SECOND.read_text()) if SECOND.exists() else {}


def key(i):
    return "|".join(ITEMS[i])


def show(i, page=None):
    i = max(0, min(int(i), len(ITEMS) - 1))
    doc, field = ITEMS[i]
    n_pages = len(extract.load_pages(doc))
    page = int(page or PAGE_OF.get((doc, field), PAGE[doc]))
    page = max(1, min(page, n_pages))
    png = extract.render(config.RAW / config.DOC_FILES[doc], page, dpi=110, out_dir=config.PROCESSED / "images_ui")
    done = load()
    prev = done.get(key(i))
    head = (f"### {i + 1} of {len(ITEMS)} &nbsp; ({len(done)} answered)\n**Document:** {config.DOC_TITLES[doc]} &nbsp; **Field:** `{field}`  \n"
            f"**Label on the page:** {LABEL.get(field, field)} &nbsp; **Expected:** {HOW.get(FIELD_TYPES.get(field, 'text'), 'text, as written')}  \n"
            f"**Look at page {PAGE_OF.get((doc, field), PAGE[doc])}** (use the slider to check other pages)"
            + (f"  \nYour saved answer: **{prev['value'] or 'no value on the page'}**" if prev else ""))
    return i, head, str(png), gr.update(value=page, maximum=n_pages), (prev or {}).get("value") or ""


def save(i, value, empty):
    doc, field = ITEMS[int(i)]
    value = "" if empty else (value or "").strip()
    if not empty and not value:
        return show(i)                                      # nothing typed: stay on the item
    done = load()
    done[key(i)] = {"doc": doc, "field": field, "value": value or None,
                    "normalized": (norm_value(FIELD_TYPES.get(field, "text"), value) or value.lower()) if value else None}
    SECOND.write_text(json.dumps(done, indent=1))
    return show(int(i) + 1)


with gr.Blocks(title="VeriDoc-KG: second annotation") as demo:
    gr.Markdown("# Field labels: second pass\nType what the page shows. Do not look at the pipeline's output or the first labels while doing this.")
    idx = gr.State(0)
    head = gr.Markdown()
    with gr.Row():
        with gr.Column(scale=6):
            img = gr.Image(type="filepath", label="Page", height=900)
        with gr.Column(scale=3):
            page = gr.Slider(1, 36, value=1, step=1, label="Page")
            val = gr.Textbox(label="Value as written on the page")
            b_save = gr.Button("Save value and go on", variant="primary")
            b_none = gr.Button("No value on the page (blank or redacted)")
            with gr.Row():
                b_back = gr.Button("Back")
                b_skip = gr.Button("Skip")
    outs = [idx, head, img, page, val]
    demo.load(lambda: show(len(load()) if len(load()) < len(ITEMS) else 0), None, outs)
    b_save.click(lambda i, v: save(i, v, False), [idx, val], outs)
    val.submit(lambda i, v: save(i, v, False), [idx, val], outs)
    b_none.click(lambda i: save(i, "", True), idx, outs)
    b_back.click(lambda i: show(int(i) - 1), idx, outs)
    b_skip.click(lambda i: show(int(i) + 1), idx, outs)
    page.release(lambda i, p: show(i, p)[:3], [idx, page], [idx, head, img])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
