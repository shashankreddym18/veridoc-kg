"""The VeriDoc-KG ontology, generated from the same tables the pipeline uses.

    cd CODE && python -m veridoc.ontology            # writes CODE/ontology/ontology.ttl
    cd CODE && python -m veridoc.ontology --check    # also checks the graph, the shapes and the deduction mapping against it

Classes and properties are declared once here (RDFS/OWL), so the extractor's field list, the SHACL
shapes, the knowledge graph and the deduction-file mapping cannot drift apart unnoticed.
The human-readable description is CODE/ontology/ONTOLOGY.md.
"""
from __future__ import annotations
import json, sys
from rdflib import BNode, Graph, Literal, Namespace, RDF, RDFS, OWL
from rdflib.collection import Collection
from rdflib.namespace import XSD, SH

from . import config
from .fields import FIELD_TYPES
from .verifier import EX, ENTITY_OF, XSD_OF
from .kg import RELATIONS, STAGES
from .deduction import COLUMN_FIELD

OUT = config.ONTOLOGY / "ontology.ttl"

CLASSES = {
    "Group": "The employer (group policyholder) that sponsors the worksite plan.",
    "Certificate": "One employee's certificate of coverage under the group policy.",
    "Insured": "The employee covered by a certificate.",
    "Rider": "An optional benefit attached to a certificate (here: accidental death benefit).",
    "Document": "A source PDF of the case.",
    "WorkflowStage": "A stage of the worksite workflow that produces a document (quote, master application, enrollment, policy, certificate).",
    "Evidence": "The place on a page that supports one property value: document, page, bounding box and text span.",
}
# label as printed on the documents (also used in the document)
LABEL = {
    "group_name": "Group Name / Group Policyholder", "group_number": "Group Number", "group_street": "PO Box or Street",
    "group_state": "State or Country", "situs_state": "Situs State", "eligible_employees": "Number of eligible employees/members",
    "proposed_effective_date": "Proposed effective date", "signature_date": "Date (signature block)", "presented_by": "Presented By",
    "proposal_date": "Date (proposal cover)", "face_amount": "Face Amount", "certificate_number": "Certificate Number",
    "certificate_date": "Certificate Date", "maturity_date": "Maturity Date", "risk_class": "Risk Class", "plan": "Plan",
    "min_paidup_amount": "Minimum Paid-Up Life Insurance Amount", "premium_frequency": "Initial Premium Payment Frequency",
    "loan_interest_rate": "Loan Interest Rate", "annual_premium": "Annual Certificate Premium (excluding Riders)",
    "payroll_frequency": "Your <frequency> Premium", "payroll_deduction_amount": "Your <frequency> Premium (amount)",
    "insured_name": "Insured", "date_of_birth": "Date of Birth", "issue_age": "Issue Age", "ee_city": "City",
    "ee_state": "State", "ee_zip": "Zip Code", "date_of_hire": "Date of Hire", "annual_salary": "Annual Salary/Compensation",
    "job_title": "Job Title/Occupation", "rider_date": "Rider Date", "rider_benefit_amount": "Benefit Amount",
    "rider_annual_premium": "Annual Premium (rider)", "rider_termination_date": "Termination Date",
}
OBJECT_PROPERTIES = {   # name: (domain, range, comment)
    **{rel: (a, b, "") for a, rel, b in RELATIONS},
    "produced_at": ("Document", "WorkflowStage", "The workflow stage that produces this document."),
    "precedes": ("WorkflowStage", "WorkflowStage", "Order of the workflow stages."),
    "supports_entity": ("Evidence", None, "The individual whose property this evidence supports."),
    "doc": ("Evidence", "Document", "The document the evidence is on."),
}
KEYED = ("Group", "Insured", "Certificate", "Rider")
KEY_PROPS = {
    "canonical_key": "Deterministic identity of the entity, built from canonical field values by fixed rules (veridoc/keys.py).",
    "key_basis": "Which key rule built canonical_key (e.g. group_number, name+state, certificate_number).",
}
EVIDENCE_DATA = {       # name: (range, comment)
    "supports_field": (XSD.string, "Name of the property this evidence supports."),
    "page": (XSD.integer, "1-based page number."),
    "span": (XSD.string, "The evidence text as read from the page."),
    "bbox": (XSD.string, "Bounding box [x0, y0, x1, y1] in PDF points, as JSON."),
    "status": (XSD.string, "verified (every check passed) or human_verified (restored by a reviewer)."),
}


def build() -> Graph:
    g = Graph(); g.bind("ex", EX); g.bind("owl", OWL)
    g.add((EX[""], RDF.type, OWL.Ontology))
    g.add((EX[""], RDFS.label, Literal("VeriDoc-KG worksite insurance ontology")))
    for c, comment in CLASSES.items():
        g.add((EX[c], RDF.type, OWL.Class)); g.add((EX[c], RDFS.label, Literal(c))); g.add((EX[c], RDFS.comment, Literal(comment)))
        for other in CLASSES:
            if other > c:
                g.add((EX[c], OWL.disjointWith, EX[other]))
    for p, (dom, rng, comment) in OBJECT_PROPERTIES.items():
        g.add((EX[p], RDF.type, OWL.ObjectProperty)); g.add((EX[p], RDFS.domain, EX[dom]))
        if rng:
            g.add((EX[p], RDFS.range, EX[rng]))
        if comment:
            g.add((EX[p], RDFS.comment, Literal(comment)))
    for field, ent in ENTITY_OF.items():
        g.add((EX[field], RDF.type, OWL.DatatypeProperty)); g.add((EX[field], RDFS.domain, EX[ent]))
        g.add((EX[field], RDFS.range, XSD_OF.get(FIELD_TYPES.get(field, "text"), XSD.string)))
        g.add((EX[field], RDFS.label, Literal(LABEL.get(field, field.replace("_", " ")))))
    for p, (rng, comment) in EVIDENCE_DATA.items():
        g.add((EX[p], RDF.type, OWL.DatatypeProperty)); g.add((EX[p], RDFS.domain, EX.Evidence)); g.add((EX[p], RDFS.range, rng))
        g.add((EX[p], RDFS.comment, Literal(comment)))
    g.add((EX.title, RDF.type, OWL.DatatypeProperty)); g.add((EX.title, RDFS.domain, EX.Document)); g.add((EX.title, RDFS.range, XSD.string))
    # deterministic identity (veridoc/keys.py): one canonical key per entity, declared as an OWL 2 key,
    # so two individuals with the same key are the same entity and nothing else merges them
    for p, comment in KEY_PROPS.items():
        g.add((EX[p], RDF.type, OWL.DatatypeProperty)); g.add((EX[p], RDFS.range, XSD.string)); g.add((EX[p], RDFS.comment, Literal(comment)))
        g.add((EX[p], RDFS.domain, EX.KeyedEntity))
    g.add((EX.KeyedEntity, RDF.type, OWL.Class)); g.add((EX.KeyedEntity, RDFS.comment, Literal("An entity that can carry a canonical key: Group, Insured, Certificate or Rider.")))
    for c in KEYED:
        g.add((EX[c], RDFS.subClassOf, EX.KeyedEntity))
        lst = BNode(); Collection(g, lst, [EX.canonical_key]); g.add((EX[c], OWL.hasKey, lst))
    return g


def check(onto: Graph) -> list:
    """Problems found when the other artifacts are compared with the ontology (empty list = consistent)."""
    problems = []
    classes = set(onto.subjects(RDF.type, OWL.Class))
    props = set(onto.subjects(RDF.type, OWL.DatatypeProperty)) | set(onto.subjects(RDF.type, OWL.ObjectProperty))
    for f in FIELD_TYPES:                                   # every extractor field is a property of some class
        if f not in ENTITY_OF:
            problems.append(f"field {f} is extracted but belongs to no class")
        if f not in LABEL:
            problems.append(f"field {f} has no printed label")
    shapes = Graph().parse(config.ONTOLOGY / "shapes.ttl")
    for c in shapes.objects(None, SH.targetClass):
        if c not in classes:
            problems.append(f"shape targets undeclared class {c}")
    for p in set(shapes.objects(None, SH.path)) | set(shapes.objects(None, SH.lessThan)):
        if p not in props:
            problems.append(f"shape uses undeclared property {p}")
    for shape, cls in shapes.subject_objects(SH.targetClass):  # a shape may only constrain properties of its own class
        for prop_node in shapes.objects(shape, SH.property):
            for p in shapes.objects(prop_node, SH.path):
                if (p, RDFS.domain, cls) not in onto:
                    problems.append(f"{shape} constrains {p}, whose domain is not {cls}")
    for col, f in COLUMN_FIELD.items():
        if f and f not in ENTITY_OF:
            problems.append(f"deduction column {col!r} maps to unknown field {f}")
    kg_path = config.PROCESSED / "kg" / "kg.ttl"
    if kg_path.exists():
        kg = Graph().parse(kg_path)
        for p in set(kg.predicates()) - {RDF.type}:
            if p not in props:
                problems.append(f"graph uses undeclared property {p}")
        for c in set(kg.objects(None, RDF.type)):
            if c not in classes:
                problems.append(f"graph uses undeclared class {c}")
        for s in set(kg.subjects(RDF.type, None)):          # the classes are disjoint: one class per individual
            types = set(kg.objects(s, RDF.type))
            if len(types) > 1:
                problems.append(f"{s} has more than one class: {sorted(map(str, types))}")
        for s, p, o in kg:                                  # domain check on every data and object property
            for dom in onto.objects(p, RDFS.domain):
                if not any(dom in onto.transitive_objects(t, RDFS.subClassOf) for t in kg.objects(s, RDF.type)):
                    problems.append(f"{s} has {p} but is not a {dom}")
            for rng in onto.objects(p, RDFS.range):
                if rng in classes and (o, RDF.type, rng) not in kg:
                    problems.append(f"{p} points at {o}, which is not a {rng}")
    return sorted(set(problems))


def main(do_check: bool) -> None:
    g = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(OUT, format="turtle")
    n = lambda t: len(set(g.subjects(RDF.type, t)))
    print(json.dumps({"file": str(OUT), "classes": n(OWL.Class), "object_properties": n(OWL.ObjectProperty),
                      "datatype_properties": n(OWL.DatatypeProperty), "triples": len(g)}))
    if do_check:
        problems = check(g)
        print("consistent: the graph, the shapes and the deduction mapping use only declared terms" if not problems
              else "PROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main("--check" in sys.argv)
