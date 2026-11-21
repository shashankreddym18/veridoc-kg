"""Load DATA/processed/kg/kg.cypher into Neo4j and report integrity numbers from the database.

Start Neo4j on a laptop (one of):
    brew install neo4j && neo4j start                                  # then set a password at http://localhost:7474
    docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/veridoc-kg neo4j:5
Then, from CODE/:
    pip install neo4j
    NEO4J_PASSWORD=veridoc-kg python load_neo4j.py
Writes EVALUATIONS/results/e10_neo4j.json. The load wipes the database first (it is a demo database).
"""
import json, os, sys
from veridoc import config

QUERIES = {
    "nodes_by_label": "MATCH (x) UNWIND labels(x) AS l RETURN l AS key, count(*) AS n ORDER BY key",
    "relationships_by_type": "MATCH ()-[r]->() RETURN type(r) AS key, count(*) AS n ORDER BY key",
    "evidence_without_document": "MATCH (v:Evidence) WHERE NOT EXISTS { (v)-[:FROM]->(:Document) } RETURN count(v) AS n",
    "evidence_without_entity": "MATCH (v:Evidence) WHERE NOT EXISTS { (v)-[:SUPPORTS]->() } RETURN count(v) AS n",
    # every extracted property on an entity must have at least one Evidence node for that field
    # (id, canonical_key and key_basis are derived identity, not extracted values)
    "properties_without_evidence": """
        MATCH (e) WHERE e:Certificate OR e:Group OR e:Insured OR e:Rider
        UNWIND [k IN keys(e) WHERE NOT k IN ['id', 'canonical_key', 'key_basis']] AS field
        OPTIONAL MATCH (v:Evidence {field: field})-[:SUPPORTS]->(e)
        WITH e, field, count(v) AS ev WHERE ev = 0
        RETURN count(*) AS n""",
    "properties_total": """
        MATCH (e) WHERE e:Certificate OR e:Group OR e:Insured OR e:Rider
        RETURN sum(size([k IN keys(e) WHERE NOT k IN ['id', 'canonical_key', 'key_basis']])) AS n""",
    "human_verified_evidence": "MATCH (v:Evidence {status:'human_verified'}) RETURN count(v) AS n",
}


def connect(GraphDatabase, uri, user, password):
    """Open a driver. A fresh Neo4j still has the default password neo4j/neo4j and refuses queries
    until it is changed, so change it to the requested password when that is the case."""
    from neo4j.exceptions import AuthError, ClientError
    drv = GraphDatabase.driver(uri, auth=(user, password))
    try:
        drv.verify_connectivity()
        return drv
    except (AuthError, ClientError):
        drv.close()
    boot = GraphDatabase.driver(uri, auth=(user, "neo4j"))
    with boot.session(database="system") as s:
        s.run("ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO $new", new=password)
    boot.close()
    drv = GraphDatabase.driver(uri, auth=(user, password))
    drv.verify_connectivity()
    return drv


def main():
    try:
        from neo4j import GraphDatabase
    except ImportError:
        sys.exit("pip install neo4j")
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user, password = os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "veridoc-kg")
    stmts = [s for s in (config.PROCESSED / "kg" / "kg.cypher").read_text().split(";\n") if s.strip()]
    out, errors = {}, {}
    with connect(GraphDatabase, uri, user, password) as drv, drv.session() as s:
        out["server"] = drv.get_server_info().agent
        s.run("MATCH (x) DETACH DELETE x").consume()
        for st in stmts:
            s.run(st.rstrip(";")).consume()
        for name, q in QUERIES.items():
            try:
                rows = [r.data() for r in s.run(q)]
                out[name] = {r["key"]: r["n"] for r in rows} if rows and "key" in rows[0] else (rows[0]["n"] if rows else 0)
            except Exception as e:                      # report the query that failed, keep the others
                errors[name] = str(e)[:300]
    out["statements_loaded"] = len(stmts)
    if "properties_without_evidence" in out and out.get("properties_total"):
        out["provenance_completeness"] = round(1 - out["properties_without_evidence"] / out["properties_total"], 3)
    if errors:
        out["query_errors"] = errors
    (config.RESULTS / "e10_neo4j.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
