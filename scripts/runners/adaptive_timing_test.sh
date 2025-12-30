#!/bin/bash
# Adaptive scaling timing test: 10, 100, 1000 concepts with relationship-first

set -e

mkdir -p results/adaptive_timing_test
mkdir -p logs

echo "======================================================================"
echo "ADAPTIVE SCALING TIMING TEST"
echo "Configuration: 1 def + adaptive rels, relationship-first mode"
echo "======================================================================"
echo

for n in 10 100 1000; do
    echo "======================================================================"
    echo "Testing $n concepts..."
    echo "======================================================================"

    case $n in
        10)
            graph="data/concept_graph/wordnet_v2_top10.json"
            ;;
        100)
            graph="data/concept_graph/wordnet_v2_top100.json"
            ;;
        1000)
            graph="data/concept_graph/wordnet_v2_top1000.json"
            ;;
    esac

    start_time=$(date +%s)

    poetry run python scripts/scaling_study.py \
        --concept-graph "$graph" \
        --model google/gemma-3-4b-pt \
        --output-dir "results/adaptive_timing_test/n${n}" \
        --single \
        --n-concepts "$n" \
        --n-definitions 1 \
        --n-relationships 999 \
        --relationship-first \
        2>&1 | tee "logs/adaptive_timing_n${n}.log"

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    echo
    echo "======================================================================"
    echo "Completed $n concepts in ${elapsed}s ($(($elapsed / 60))m)"
    echo "======================================================================"
    echo
done

echo
echo "======================================================================"
echo "TIMING TEST COMPLETE"
echo "======================================================================"
