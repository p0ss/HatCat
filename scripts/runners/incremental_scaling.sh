#!/bin/bash
#
# Incremental Scaling Study
# Run configurations independently, starting with best candidates
#

set -e

CONCEPT_GRAPH_10="data/concept_graph/wordnet_v2_top10.json"
CONCEPT_GRAPH_100="data/concept_graph/wordnet_v2_top100.json"
MODEL="google/gemma-3-4b-pt"
OUTPUT_DIR="results/scaling_incremental"

mkdir -p "$OUTPUT_DIR"

# Helper function to run single config
run_config() {
    local n_concepts=$1
    local n_defs=$2
    local n_rels=$3
    local graph=$4
    local name=$5

    echo ""
    echo "======================================================================="
    echo "Running: $name"
    echo "  Concepts: $n_concepts, Definitions: $n_defs, Relationships: $n_rels"
    echo "======================================================================="

    poetry run python scripts/scaling_study.py \
        --concept-graph "$graph" \
        --model "$MODEL" \
        --output-dir "$OUTPUT_DIR" \
        --single \
        --n-concepts "$n_concepts" \
        --n-definitions "$n_defs" \
        --n-relationships "$n_rels"

    # Extract and display result
    local result_file="$OUTPUT_DIR/scaling_c${n_concepts}_d${n_defs}_r${n_rels}.json"
    if [ -f "$result_file" ]; then
        local val_acc=$(cat "$result_file" | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"{data['results']['mean_val_acc']:.1%}\")" 2>/dev/null || echo "N/A")
        local time=$(cat "$result_file" | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"{data['results']['elapsed_seconds']:.1f}s\")" 2>/dev/null || echo "N/A")
        echo "  Result: $val_acc validation accuracy in $time"
    fi
}

# Phase 1: Best candidate (max relationships, balanced samples)
echo "======================================================================="
echo "PHASE 1: BEST CANDIDATE"
echo "======================================================================="
echo "Testing maximum relationship usage with good sample size"
echo ""

run_config 10 10 100 "$CONCEPT_GRAPH_10" "10 concepts × 10 defs × 100 rels"

echo ""
echo "Press Enter to continue to Phase 2, or Ctrl+C to stop..."
read

# Phase 2: Scale concepts (keep best config)
echo ""
echo "======================================================================="
echo "PHASE 2: SCALE CONCEPTS"
echo "======================================================================="
echo "Testing if best config scales to more concepts"
echo ""

run_config 100 10 100 "$CONCEPT_GRAPH_100" "100 concepts × 10 defs × 100 rels"

echo ""
echo "Press Enter to continue to Phase 3, or Ctrl+C to stop..."
read

# Phase 3: Reduce samples (test diminishing returns)
echo ""
echo "======================================================================="
echo "PHASE 3: REDUCE SAMPLES"
echo "======================================================================="
echo "Testing if we can reduce samples without hurting performance"
echo ""

run_config 10 1 10 "$CONCEPT_GRAPH_10" "10 concepts × 1 def × 10 rels"
run_config 10 10 10 "$CONCEPT_GRAPH_10" "10 concepts × 10 defs × 10 rels"

echo ""
echo "Press Enter to continue to Phase 4, or Ctrl+C to stop..."
read

# Phase 4: Minimal config (establish baseline)
echo ""
echo "======================================================================="
echo "PHASE 4: MINIMAL BASELINE"
echo "======================================================================="
echo "Testing absolute minimum to establish floor"
echo ""

run_config 10 1 1 "$CONCEPT_GRAPH_10" "10 concepts × 1 def × 1 rel"
run_config 10 10 1 "$CONCEPT_GRAPH_10" "10 concepts × 10 defs × 1 rel"

echo ""
echo "======================================================================="
echo "INCREMENTAL SCALING COMPLETE"
echo "======================================================================="
echo ""
echo "Results saved in: $OUTPUT_DIR"
