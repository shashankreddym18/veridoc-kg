#!/bin/bash
# Runs, in order, every step that needs a laptop (model downloads, a local LLM) and logs to
# DATA/processed/laptop_run.log. A failed step is recorded and the script moves on.
# Safe to re-run: extracted pages and LLM responses are cached, so it resumes where it stopped.
#
#   cd veridoc-kg && caffeinate -i bash run_laptop.sh            # caffeinate keeps a Mac awake
#   MODEL=qwen2.5:7b bash run_laptop.sh                           # another Ollama model
#   SKIP_LLM=1 bash run_laptop.sh                                 # everything except the LLM steps
#   SKIP_DOCLING=1 bash run_laptop.sh                             # leave Docling out
cd "$(dirname "$0")" || exit 1
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="$PWD/CODE"
MODEL=${MODEL:-llama3.1:8b}
mkdir -p DATA/processed
LOG=DATA/processed/laptop_run.log
STATUS=DATA/processed/laptop_status.txt
: > "$STATUS"
echo "=== run started $(date) model=$MODEL" >> "$LOG"

step() {
  local name=$1; shift
  local t0=$SECONDS
  echo; echo ">>> $name"
  echo ">>> $name: $*" >> "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then echo "ok    $((SECONDS - t0))s  $name" >> "$STATUS"; else echo "FAIL  rc=$rc  $name" >> "$STATUS"; fi
  return "$rc"
}

ollama_up() { curl -s -m 2 http://localhost:11434/api/version > /dev/null; }
ensure_ollama() {
  ollama_up && return 0
  command -v brew > /dev/null && brew services start ollama > /dev/null 2>&1
  for _ in $(seq 30); do ollama_up && return 0; sleep 1; done
  (nohup ollama serve > /dev/null 2>&1 &)
  for _ in $(seq 20); do ollama_up && return 0; sleep 1; done
  echo "Ollama server did not start"; return 1
}

step "install packages"            pip install -q -r requirements.txt sentence-transformers
step "refresh native-text pages"   python CODE/run_extract.py --native-only
step "pipeline"                    python -m veridoc.pipeline
step "ontology: generate, check"   python -m veridoc.ontology --check
step "E5b NLI check"               python EVALUATIONS/eval_nli.py
step "E9 questions (graph tier)"   python EVALUATIONS/eval_qa.py

# eval_synthetic.py generates the forms when they are missing, then extracts and caches their pages
step "E7 dev: make forms, extract" python EVALUATIONS/eval_synthetic.py --set dev
step "CRF: train on dev"           python -m veridoc.crf
step "E7 dev"                      python EVALUATIONS/eval_synthetic.py --set dev
step "E7 test"                     python EVALUATIONS/eval_synthetic.py --set test
step "E8 agreement policy"         python EVALUATIONS/eval_agree_policy.py
step "E12 canonical forms"        python EVALUATIONS/eval_canonical.py
step "E1 E3 E4 E5 E6"              python EVALUATIONS/run_evals.py
[ -f DATA/gold/fields_gold_second.json ] && step "E11 label agreement" python EVALUATIONS/eval_agreement.py

if [ -z "$SKIP_LLM" ] && step "ollama server" ensure_ollama && step "ollama pull $MODEL" ollama pull "$MODEL"; then
  if step "LLM extraction: smoke test (2 pages)" python -m veridoc.fields_llm --model "$MODEL" --docs real --limit 2; then
    step "LLM extraction: real documents"        python -m veridoc.fields_llm --model "$MODEL" --docs real
    step "LLM reader on the question set"        python -m veridoc.qa_llm --model "$MODEL"
    step "LLM router: question -> field"         python -m veridoc.qa_router --model "$MODEL"
    if [ -f DATA/gold/qa_human.json ]; then        # the questions in a team member's words (CODE/app/reword.py)
      step "LLM reader on the reworded questions"  python -m veridoc.qa_llm --model "$MODEL" --set human
      step "LLM router on the reworded questions"  python -m veridoc.qa_router --model "$MODEL" --set human
    fi
    step "LLM extraction: synthetic test forms"  python -m veridoc.fields_llm --model "$MODEL" --docs synthetic-test
    step "E1 E3 E4 E5 E6 with LLM rows"          python EVALUATIONS/run_evals.py
    step "E7 test with LLM rows"                 python EVALUATIONS/eval_synthetic.py --set test
    step "E9 with LLM reader"                    python EVALUATIONS/eval_qa.py
    [ -f DATA/gold/qa_human.json ] && step "E9 on the reworded questions"  python EVALUATIONS/eval_qa.py --set human
    step "E5c LLM second opinion"                python EVALUATIONS/eval_judge.py --model "$MODEL"
  fi
fi

# Docling on the scanned documents (downloads its models on first use). SKIP_DOCLING=1 to leave it out.
if [ -z "$SKIP_DOCLING" ]; then
  step "install docling"                        pip install -q docling
  step "Docling: scanned documents"             python -m veridoc.docling_extract --docs master_app certificate
fi
step "E2b scanned text: Tesseract, Docling"     python EVALUATIONS/eval_docling.py

# Neo4j, only if it is installed (brew install neo4j). The loader sets the password on a fresh install.
if command -v neo4j > /dev/null; then
  neo4j status > /dev/null 2>&1 || neo4j start > /dev/null 2>&1
  for _ in $(seq 90); do nc -z localhost 7687 2> /dev/null && break; sleep 1; done
  step "install neo4j driver"                   pip install -q neo4j
  step "E10 Neo4j: load graph, integrity"       python CODE/load_neo4j.py
else
  echo "skip  Neo4j not installed (brew install neo4j)" >> "$STATUS"
fi

echo; echo "=== summary ($(date))"; cat "$STATUS"
echo "=== run finished $(date)" >> "$LOG"
